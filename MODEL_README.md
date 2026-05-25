# Model Card — Ethiopian Crop Intelligence

This document summarizes the disease and yield models included in this repository, how to run them, limitations, and reproduction steps.

## Models included

- Disease classifier (crop-specific): ResNet-18 backbone with a small custom head (Linear -> ReLU -> Dropout -> Linear). Models are saved under `models/disease/<crop>_disease_model.pth` and include a bundled `model_state_dict`, `class_names`, `image_size`, and simple `metrics` JSON.
- Yield predictor: Feed-forward MLP regression (tabular). See `src/training` for training scripts and saved artifacts under `models/yield`.

## Intended use

- Input: RGB images of crop leaves. For disease prediction the `crop_type` must match a trained checkpoint (e.g., `Coffee`, `Wheat`, `Maize`, `Sorghum`). Class names in bundles follow `healthy` or `<normalized_crop>_<label>`.
- Output: Predicted disease class and per-class probabilities.

## Quick usage examples

From the repository root, run the smoke prediction snippet (requires virtualenv activated):

```bash
python -c "from src.prediction.predict_disease import predict_disease; print(predict_disease('ethiopian cofee leaf dataset/train aug/Cerscospora/1550.jpg', 'Coffee'))"
```

Run the Streamlit dashboard:

```bash
.venv\Scripts\python.exe -m streamlit run main.py --server.port 8501
```

Run the FastAPI endpoint (if available after export):

```bash
.venv\Scripts\python.exe -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

## Reproducibility

- The training scripts save minimal `outputs/reports/*.json` and `outputs/graphs/*` but full experiment tracking (MLflow/W&B) is not enabled by default — see `src/training/train_disease_model.py` for parameters to reproduce a run.
- To reproduce a crop-specific disease model: run `train_disease_model(train_dir, test_dir, crop_type='Coffee', epochs=...)` from `src/training/train_disease_model.py`. Use consistent `random_seed` and dataset splits to ensure exact reproducibility.

## Known gaps and limitations (summary)

- Missing: automated dataset downloader and provenance files (dataset URLs and licenses are not bundled).
- Missing: experiment tracking (MLflow/W&B) and structured hyperparameter sweep artifacts.
- Missing: explainability outputs (Grad-CAM saliency maps) — utility will be added under `src/explainability`.
- Missing: model export for production (TorchScript/ONNX) — export utilities available under `scripts/export_model.py` when added.
- Missing: unit tests and CI integration — basic tests will be added under `tests/`.
- Limited performance: sample coffee model shows low balanced performance on the provided test set; see `outputs/reports` for available metrics.

## Model governance and license

- The repository does not include a unified license for the trained artifacts. Check the dataset source for licensing before redistributing models trained on third-party data.

## Contact / Next steps

- For production: add model export, a REST endpoint, explainability, and experiment tracking.
- Example next step: run `scripts/export_model.py --model models/disease/coffee_disease_model.pth --format onnx,script` once created.
