from __future__ import annotations

import argparse
from pathlib import Path
import torch

from src.prediction.predict_disease import DiseaseClassifier
from src.preprocessing.image_preprocessing import build_image_transforms


def export(bundle_path: Path, out_dir: Path, formats: list[str]):
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = torch.load(bundle_path, map_location="cpu")
    class_names = bundle.get("class_names")
    image_size = bundle.get("image_size", 224)

    model = DiseaseClassifier(num_classes=len(class_names))
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 3, image_size, image_size)

    if "script" in formats:
        try:
            # Try tracing first; tracing can fail for models with data-dependent control flow
            scripted = torch.jit.trace(model, dummy_input)
            script_path = out_dir / (bundle_path.stem + "_script.pt")
            scripted.save(script_path)
            print("Saved TorchScript at", script_path)
        except Exception as exc:  # pragma: no cover - environment-dependent
            print("TorchScript export failed (tracing). Skipping script export:", exc)
            try:
                scripted = torch.jit.script(model)
                script_path = out_dir / (bundle_path.stem + "_script_scripted.pt")
                scripted.save(script_path)
                print("Saved TorchScript (script) at", script_path)
            except Exception as exc2:
                print("TorchScript scripting also failed. Recommend using ONNX. Error:", exc2)

    if "onnx" in formats:
        onnx_path = out_dir / (bundle_path.stem + ".onnx")
        try:
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                input_names=["input"],
                output_names=["output"],
                opset_version=11,
            )
            print("Saved ONNX at", onnx_path)
        except Exception as exc:  # pragma: no cover - environment-dependent
            print("ONNX export failed:", exc)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, default=Path("models/disease/coffee_disease_model.pth"))
    p.add_argument("--out", type=Path, default=Path("models/exports"))
    p.add_argument("--formats", type=str, default="script,onnx")
    args = p.parse_args()
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    export(args.bundle, args.out, formats)


if __name__ == "__main__":
    main()
