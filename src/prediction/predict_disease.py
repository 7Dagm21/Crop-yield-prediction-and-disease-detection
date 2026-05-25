from __future__ import annotations

from io import BufferedIOBase
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

from ..config import DISEASE_CLASS_NAMES, DISEASE_MODEL_PATH, get_disease_model_path, normalize_crop_name
from ..preprocessing.image_preprocessing import build_image_transforms


class DiseaseClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs)

def predict_disease(image_path: str | Path | BufferedIOBase, crop_type: str, model_path: Path | None = None) -> dict[str, Any]:
    fallback_used = False
    if model_path is None:
        model_path = get_disease_model_path(crop_type)

    if not model_path.exists():
        if DISEASE_MODEL_PATH.exists():
            model_path = DISEASE_MODEL_PATH
            fallback_used = True
        else:
            raise FileNotFoundError(
                f"No trained disease model found for {crop_type}. Train a crop-specific disease model and try again."
            )

    bundle = torch.load(model_path, map_location="cpu")
    class_names = bundle.get("class_names", DISEASE_CLASS_NAMES)
    normalized_crop = normalize_crop_name(crop_type)
    if not fallback_used and class_names and not all(
        class_name == "healthy" or class_name.startswith(normalized_crop + "_") for class_name in class_names
    ):
        raise FileNotFoundError(
            f"The checkpoint at {model_path} does not match crop_type='{crop_type}'. Train the matching crop-specific model."
        )

    model = DiseaseClassifier(num_classes=len(class_names))
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()

    image_source = image_path
    if hasattr(image_source, "seek"):
        image_source.seek(0)

    image_tensor = build_image_transforms(image_size=bundle.get("image_size", 224), training=False)(
        Image.open(image_source).convert("RGB")
    ).unsqueeze(0)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities_tensor = torch.softmax(logits, dim=1)[0]

    best_index = int(torch.argmax(probabilities_tensor).item())
    probabilities = {
        class_names[index]: round(float(probabilities_tensor[index].item()), 4)
        for index in range(len(class_names))
    }

    return {
        "disease": class_names[best_index],
        "confidence": round(float(probabilities_tensor[best_index].item()), 4),
        "probabilities": probabilities,
        "method": "pytorch_resnet18",
        "model_source": "legacy_generic" if fallback_used else "crop_specific",
        "model_path": str(model_path),
    }
