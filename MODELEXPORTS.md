# Model export recommendations

This project exports models in multiple formats. Notes and best-practices:

- TorchScript:
  - TorchScript `trace` may fail or cause native crashes for models with data-dependent control flow
    or custom ops. The exporter will attempt `trace` then fall back to `script` and will skip on failure.
  - If TorchScript fails on your system, prefer ONNX for portability or serve the PyTorch bundle directly.

- ONNX:
  - ONNX export is the recommended portable format. Use `scripts/export_model.py --formats onnx`.
  - ONNX runtime (onnxruntime) can run ONNX models efficiently across platforms.

- Troubleshooting:
  - If exports fail, ensure your environment has compatible `torch` and `onnx` versions.
  - For reproducible exports, use the same Python/torch versions as used for training.

Files:

- `scripts/export_model.py` — best-effort exporter; will not crash the training pipeline.
