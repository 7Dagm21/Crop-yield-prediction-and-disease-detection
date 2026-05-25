# Ethiopian Crop Intelligence System

A PyTorch-based agriculture intelligence system for Ethiopian grains and coffee.

## Scope

This project is limited to these Ethiopian crops:

- Teff
- Wheat
- Maize
- Sorghum
- Barley
- Millet
- Chickpea
- Coffee

The system has two main modules:

- Crop yield prediction
  A concise guide to run and develop the Ethiopian Crop Intelligence System (yield + disease).

Prerequisites

- Python 3.10+ and a virtual environment
- Install project dependencies:

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Quick start

- Start the Streamlit dashboard (local dev):

```powershell
streamlit run main.py
```

Public deployment

- If you want the dashboard visible from any computer, deploy it to Streamlit Community Cloud and set the entry file to `main.py`.
- See `deployment/streamlit-cloud.md` for the exact steps and the repo size warning.

- Run the FastAPI server (inference):

```powershell
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Run tests

- Execute unit tests:

```powershell
pytest -q
```

Training (disease models)

- Expect ImageFolder layout under `data/disease_images/` with `train/` and `test/` subfolders per class.
- Train a disease model (example):

```powershell
python -m src.training.train_disease_model --data data/disease_images --crop coffee --epochs 5
```

Model export

- Export TorchScript/ONNX for a saved bundle:

```powershell
python scripts/export_model.py --bundle models/disease/coffee_disease_model.pth --out models/exports/
```

Repository notes

- Large training datasets and model checkpoints live in the workspace but should not be pushed to public remotes unless intended. Keep them in `data/` and `models/` locally.
- Common cache folders to ignore: `__pycache__`, `.pytest_cache`, `.ipynb_checkpoints`.

Project structure (key folders)

```
ethiopian-crop-intelligence-system/
├─ data/                      # datasets (local only)
├─ models/                    # saved model bundles and exports
├─ outputs/                   # training logs, grads, reports
├─ src/                       # application code (training, api, dashboard)
├─ scripts/                   # helpers (export, run wrappers)
├─ requirements.txt
└─ README.md
```

Next steps

- I will prepare a concise commit for these documentation and cleanup edits when you confirm. After commit we can push to GitHub.
  python -m src.training.compare_models
