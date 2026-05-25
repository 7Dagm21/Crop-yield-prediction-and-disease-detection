from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.train_disease_model import DiseaseClassifier, _build_dataloaders
from src.config import get_disease_model_path


def run_inplace(train_src: Path, test_src: Path):
    train_ds, test_ds = _build_dataloaders(train_src, test_src, image_size=224, batch_size=16)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    if hasattr(train_ds, 'classes'):
        class_names = list(train_ds.classes)
    elif hasattr(train_ds, 'dataset') and hasattr(train_ds.dataset, 'classes'):
        class_names = list(train_ds.dataset.classes)
    else:
        class_names = [str(i) for i in range(2)]

    print('Detected classes:', class_names)

    model = DiseaseClassifier(num_classes=len(class_names), freeze_backbone=True)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'val_loss': []}
    best_state = None
    best_val = float('inf')
    no_improve = 0
    patience = 2

    all_preds = []
    all_targets = []

    for epoch in range(3):
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
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
                preds = outputs.argmax(dim=1).tolist()
                all_preds.extend(preds)
                all_targets.extend(labels.tolist())
        val_loss = float(sum(val_losses) / max(1, len(val_losses)))

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1

        print(f'Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}')
        if no_improve >= patience:
            print('Early stopping')
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # compute metrics
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
    train_src = root / 'ethiopian cofee leaf dataset' / 'train aug'
    test_src = root / 'ethiopian cofee leaf dataset' / 'test'
    if not train_src.exists():
        raise FileNotFoundError('Train source folder not found: ' + str(train_src))
    run_inplace(train_src, test_src)
