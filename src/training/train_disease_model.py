from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models

from ..config import DISEASE_CLASS_NAMES, GRAPHS_DIR, REPORTS_DIR, get_disease_model_path, normalize_crop_name
from ..preprocessing.image_preprocessing import build_image_transforms
import os
import subprocess

# optional MLflow
try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except Exception:
    _MLFLOW_AVAILABLE = False


class DiseaseClassifier(nn.Module):
    def __init__(self, num_classes: int, freeze_backbone: bool = True):
        super().__init__()
        try:
            backbone_weights = models.ResNet18_Weights.DEFAULT
            self.backbone = models.resnet18(weights=backbone_weights)
        except Exception:
            self.backbone = models.resnet18(weights=None)

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs)


def _build_dataloaders(train_dir: Path, test_dir: Path, image_size: int, batch_size: int, aug_strength: str = "medium"):
    train_dataset = datasets.ImageFolder(train_dir, transform=build_image_transforms(image_size=image_size, training=True, aug_strength=aug_strength))
    if test_dir.exists() and any(test_dir.iterdir()):
        test_dataset = datasets.ImageFolder(test_dir, transform=build_image_transforms(image_size=image_size, training=False))
        return train_dataset, test_dataset

    train_length = max(1, int(len(train_dataset) * 0.8))
    validation_length = max(1, len(train_dataset) - train_length)
    train_dataset, test_dataset = random_split(train_dataset, [train_length, validation_length])
    return train_dataset, test_dataset


def train_disease_model(
    train_dir: Path,
    test_dir: Path,
    crop_type: str,
    model_path: Path | None = None,
    image_size: int = 224,
    batch_size: int = 16,
    epochs: int = 15,
    learning_rate: float = 1e-4,
    patience: int = 5,
    aug_strength: str = "medium",
    mlflow_enabled: bool = False,
) -> dict[str, Any]:
    if not train_dir.exists() or not any(train_dir.rglob("*")):
        raise FileNotFoundError(
            "Disease image data is missing. Place crop-specific disease folders under data/disease_images/train and data/disease_images/test."
        )

    normalized_crop = normalize_crop_name(crop_type)
    if model_path is None:
        model_path = get_disease_model_path(crop_type)

    train_dataset, test_dataset = _build_dataloaders(train_dir, test_dir, image_size=image_size, batch_size=batch_size, aug_strength=aug_strength)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if hasattr(train_dataset, "classes"):
        class_names = list(train_dataset.classes)
    elif hasattr(train_dataset, "dataset") and hasattr(train_dataset.dataset, "classes"):
        class_names = list(train_dataset.dataset.classes)
    else:
        class_names = DISEASE_CLASS_NAMES

    if class_names and not all(class_name == "healthy" or class_name.startswith(normalized_crop + "_") for class_name in class_names):
        raise ValueError(
            f"Training folders under {train_dir} do not match crop_type='{crop_type}'. Expected class folders to start with '{normalized_crop}_' or be 'healthy'."
        )

    model = DiseaseClassifier(num_classes=len(class_names), freeze_backbone=True)
    optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=learning_rate)

    # Compute class weights to handle imbalance
    try:
        # train_dataset may be ImageFolder or Subset
        if hasattr(train_dataset, 'targets'):
            targets = train_dataset.targets
        elif hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'targets'):
            targets = train_dataset.dataset.targets
        else:
            # fallback: extract labels by iterating (may be slower)
            targets = []
            for _, label in train_dataset:
                targets.append(int(label))

        import numpy as _np

        classes, counts = _np.unique(_np.array(targets), return_counts=True)
        # avoid division by zero
        counts = _np.where(counts == 0, 1, counts)
        weights = 1.0 / counts
        # map weights to class index order (assumes classes are 0..N-1)
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    except Exception:
        criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": []}
    best_state = None
    best_val_loss = float("inf")
    no_improve = 0

    # MLflow run
    mlflow_run = None
    if mlflow_enabled and _MLFLOW_AVAILABLE:
        mlflow_run = mlflow.start_run(run_name=f"train_{crop_type}")
        mlflow.log_params({"epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate, "image_size": image_size, "aug_strength": aug_strength})

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = float(sum(batch_losses) / max(1, len(batch_losses)))

        # validation loss
        model.eval()
        val_losses = []
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())

        val_loss = float(sum(val_losses) / max(1, len(val_losses)))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1

        # log per-epoch metrics to MLflow if available
        if mlflow_run is not None:
            mlflow.log_metric('train_loss', train_loss, step=epoch)
            mlflow.log_metric('val_loss', val_loss, step=epoch)

        if no_improve >= patience:
            break

    # Save disease loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(history["train_loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Validation loss")
    plt.title("Disease Model Training Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / f"disease_training_curves_{normalized_crop}.png", dpi=160)
    plt.close()

    # load best state
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            all_predictions.extend(predictions.tolist())
            all_targets.extend(labels.tolist())

    metrics = {
        "accuracy": float(accuracy_score(all_targets, all_predictions)) if all_targets else 0.0,
        "precision": float(precision_score(all_targets, all_predictions, average="weighted", zero_division=0)) if all_targets else 0.0,
        "recall": float(recall_score(all_targets, all_predictions, average="weighted", zero_division=0)) if all_targets else 0.0,
        "f1": float(f1_score(all_targets, all_predictions, average="weighted", zero_division=0)) if all_targets else 0.0,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "image_size": image_size,
            "architecture": "resnet18",
            "crop_type": crop_type,
            "metrics": metrics,
            "history": history,
        },
        model_path,
    )

    # Attempt to export ONNX artifact using the export script (best-effort, non-fatal)
    try:
        export_out = Path("models/exports")
        export_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(Path(os.sys.executable)),
            str(Path(__file__).parents[1] / "scripts" / "export_model.py"),
            "--bundle",
            str(model_path),
            "--out",
            str(export_out),
            "--formats",
            "onnx",
        ]
        subprocess.run(cmd, check=False)
        onnx_path = export_out / (Path(model_path).stem + ".onnx")
        if mlflow_run is not None and onnx_path.exists():
            mlflow.log_artifact(str(onnx_path))
    except Exception:
        # non-fatal: export failure should not break training
        pass

    # MLflow logging of artifacts and final metrics
    if mlflow_run is not None:
        try:
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(model_path))
            # log generated graphs
            cm_path = GRAPHS_DIR / f"disease_confusion_matrix_{normalized_crop}.png"
            curve_path = GRAPHS_DIR / f"disease_training_curves_{normalized_crop}.png"
            if curve_path.exists():
                mlflow.log_artifact(str(curve_path))
            if cm_path.exists():
                mlflow.log_artifact(str(cm_path))
        finally:
            mlflow.end_run()

    # Save confusion matrix and classification report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    disease_report_path = REPORTS_DIR / f"disease_metrics_{normalized_crop}.json"

    if all_targets:
        cm = confusion_matrix(all_targets, all_predictions)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.title("Disease Confusion Matrix")
        plt.tight_layout()
        plt.savefig(GRAPHS_DIR / f"disease_confusion_matrix_{normalized_crop}.png", dpi=160)
        plt.close()

        report = classification_report(all_targets, all_predictions, target_names=class_names, output_dict=True, zero_division=0)
        with disease_report_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)

    return {
        "model_path": str(model_path),
        "class_names": class_names,
        "metrics": metrics,
        "crop_type": crop_type,
        "report_path": str(disease_report_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train crop-specific disease models from prepared image folders.")
    parser.add_argument(
        "--crop",
        action="append",
        dest="crops",
        help="Crop name to train. Can be passed multiple times. Defaults to all supported crops with data folders.",
    )
    args = parser.parse_args()

    crops = args.crops or ["Coffee", "Wheat", "Maize", "Sorghum"]
    train_root = Path("data/disease_images/train")
    test_root = Path("data/disease_images/test")

    trained_models = []
    skipped_crops = []

    for crop in crops:
        normalized_crop = normalize_crop_name(crop)
        crop_train_dir = train_root / normalized_crop
        crop_test_dir = test_root / normalized_crop

        if not crop_train_dir.exists() or not any(crop_train_dir.rglob("*")):
            skipped_crops.append(crop)
            continue

        result = train_disease_model(crop_train_dir, crop_test_dir, crop)
        trained_models.append(result)
        print(json.dumps(result, indent=2))

    summary = {
        "trained": [item["crop_type"] for item in trained_models],
        "skipped": skipped_crops,
    }
    print(json.dumps(summary, indent=2))
