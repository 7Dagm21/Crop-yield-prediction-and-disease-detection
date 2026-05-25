from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from torchvision import models
from src.preprocessing.image_preprocessing import build_image_transforms
from src.config import get_disease_model_path


def make_subset(dataset, max_per_class: int):
    from collections import defaultdict

    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    selected = []
    for label, idxs in class_indices.items():
        selected.extend(idxs[:max_per_class])
    return Subset(dataset, selected)


def run_tiny(train_dir: Path, test_dir: Path):
    train_dataset = datasets.ImageFolder(train_dir, transform=build_image_transforms(image_size=224, training=True))
    test_dataset = datasets.ImageFolder(test_dir, transform=build_image_transforms(image_size=224, training=False)) if test_dir.exists() else None

    train_sub = make_subset(train_dataset, max_per_class=32)
    test_sub = make_subset(test_dataset, max_per_class=8) if test_dataset is not None else None

    train_loader = DataLoader(train_sub, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_sub, batch_size=8, shuffle=False) if test_sub is not None else None

    class_names = train_dataset.classes
    print('Using classes:', class_names)

    # build a lightweight resnet18 without downloading pretrained weights
    try:
        backbone = models.resnet18(weights=None)
    except Exception:
        backbone = models.resnet18()
    in_features = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(256, len(class_names)),
    )
    model = backbone
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'val_loss': []}

    all_preds = []
    all_targets = []

    epochs = 1
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

        model.eval()
        val_losses = []
        all_preds = []
        all_targets = []
        if test_loader is not None:
            with torch.no_grad():
                for images, labels in test_loader:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_losses.append(loss.item())
                    preds = outputs.argmax(dim=1).tolist()
                    all_preds.extend(preds)
                    all_targets.extend(labels.tolist())
        val_loss = float(sum(val_losses) / max(1, len(val_losses))) if val_losses else 0.0

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        print(f'Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}')

    # metrics
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        if all_targets:
            metrics = {
                'accuracy': float(accuracy_score(all_targets, all_preds)),
                'precision': float(precision_score(all_targets, all_preds, average='weighted', zero_division=0)),
                'recall': float(recall_score(all_targets, all_preds, average='weighted', zero_division=0)),
                'f1': float(f1_score(all_targets, all_preds, average='weighted', zero_division=0)),
            }
        else:
            metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    except Exception:
        metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    model_path = get_disease_model_path('Coffee')
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'class_names': class_names, 'image_size': 224, 'architecture': 'resnet18', 'crop_type': 'Coffee', 'metrics': metrics, 'history': history}, model_path)

    print(json.dumps({'model_path': str(model_path), 'class_names': class_names, 'metrics': metrics}, indent=2))


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    train_dir = root / 'ethiopian cofee leaf dataset' / 'train aug'
    test_dir = root / 'ethiopian cofee leaf dataset' / 'test'
    run_tiny(train_dir, test_dir)
