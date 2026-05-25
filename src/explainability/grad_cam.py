from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()
        self.activations = None
        self.gradients = None

        if target_layer is None:
            # try common resnet layer
            try:
                target_layer = self.model.backbone.layer4[-1].conv2
            except Exception:
                # fallback to last conv found
                target_layer = None

        if target_layer is None:
            raise ValueError("Could not infer target layer for GradCAM; pass target_layer explicitly")

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            # grad_out is a tuple
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        input_tensor: shape (1, C, H, W)
        returns heatmap as numpy array shape (H, W) normalized 0..1
        """
        self.model.zero_grad()
        outputs = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(torch.argmax(outputs, dim=1).item())

        score = outputs[0, class_idx]
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("GradCAM hooks did not capture activations/gradients")

        # global average pooling of gradients
        pooled_grads = torch.mean(self.gradients, dim=(0, 2, 3))  # channels

        # weight the channels
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_grads[i]

        heatmap = torch.sum(activations, dim=0)
        heatmap = torch.relu(heatmap)
        heatmap -= heatmap.min()
        if heatmap.max() != 0:
            heatmap /= heatmap.max()

        heatmap_np = heatmap.cpu().numpy()
        # upsample to input size
        import cv2

        h, w = input_tensor.shape[2], input_tensor.shape[3]
        heatmap_resized = cv2.resize(heatmap_np, (w, h))
        return heatmap_resized


def overlay_heatmap(pil_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
    """Overlay heatmap (H,W) on PIL image and return combined PIL image"""
    import matplotlib
    import matplotlib.cm as cm

    cmap = cm.get_cmap("jet")
    heatmap_colored = cmap(heatmap)[:, :, :3]
    heatmap_img = (heatmap_colored * 255).astype("uint8")
    heat_pil = Image.fromarray(heatmap_img).resize(pil_img.size)

    return Image.blend(pil_img.convert("RGBA"), heat_pil.convert("RGBA"), alpha)


def save_gradcam_overlay(image_path: Path, heatmap: np.ndarray, out_path: Path, alpha: float = 0.4):
    img = Image.open(image_path).convert("RGB")
    over = overlay_heatmap(img, heatmap, alpha=alpha)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    over.save(out_path)
