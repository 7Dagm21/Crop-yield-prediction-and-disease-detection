from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import datasets

from ..config import DISEASE_METRICS_PATH, get_disease_model_path, normalize_crop_name
from ..preprocessing.image_preprocessing import build_image_transforms
from ..training.train_disease_model import DiseaseClassifier


def evaluate_disease_model(
    test_dir: Path,
    crop_type: str,
    model_path: Path | None = None,
    image_size: int = 224,
) -> dict[str, Any]:
    if model_path is None:
        model_path = get_disease_model_path(crop_type)

    if not model_path.exists():
        raise FileNotFoundError(f"Train the disease model for {crop_type} first or download its checkpoint into {model_path}.")

    bundle = torch.load(model_path, map_location="cpu")
    test_dataset = datasets.ImageFolder(test_dir, transform=build_image_transforms(image_size=image_size, training=False))
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = DiseaseClassifier(num_classes=len(bundle["class_names"]))
    model.load_state_dict(bundle["model_state_dict"])
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

    disease_metrics_path = DISEASE_METRICS_PATH.with_name(f"disease_metrics_{normalize_crop_name(crop_type)}.json")
    with disease_metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return {"metrics": metrics, "predictions": all_predictions, "actual": all_targets, "metrics_path": str(disease_metrics_path)}
