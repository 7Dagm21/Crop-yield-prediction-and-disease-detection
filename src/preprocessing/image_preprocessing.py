from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms


IMAGE_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
IMAGE_NORMALIZE_STD = [0.229, 0.224, 0.225]


def build_image_transforms(image_size: int = 224, training: bool = False, aug_strength: str = "medium"):
    """Build image transforms.

    aug_strength: one of 'none', 'light', 'medium', 'heavy'
    """
    transform_steps = []
    if training:
        # Augmentation presets
        transform_steps.append(transforms.RandomResizedCrop(image_size))
        transform_steps.append(transforms.RandomHorizontalFlip())
        if aug_strength in ("medium", "heavy"):
            transform_steps.append(transforms.RandomRotation(15))
            transform_steps.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02))
        if aug_strength == "heavy":
            transform_steps.append(transforms.RandomVerticalFlip())
            # random erasing helps robustness
            transform_steps.append(transforms.RandomErasing(p=0.2))
    else:
        transform_steps.append(transforms.Resize((image_size, image_size)))

    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NORMALIZE_MEAN, std=IMAGE_NORMALIZE_STD),
        ]
    )
    return transforms.Compose(transform_steps)


def load_image_tensor(image_path: str | Path, image_size: int = 224):
    image = Image.open(image_path).convert("RGB")
    return build_image_transforms(image_size=image_size, training=False)(image)


def image_statistics(image_path: str | Path) -> dict[str, float]:
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return {
        "red": float(array[..., 0].mean()),
        "green": float(array[..., 1].mean()),
        "blue": float(array[..., 2].mean()),
        "brightness": float(array.mean()),
        "contrast": float(array.std()),
    }
