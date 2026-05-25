from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from ..config import CROP_YIELD_DATA_PATH, REPORTS_DIR, YIELD_FEATURE_COLUMNS, YIELD_METRICS_PATH, YIELD_MODEL_PATH, YIELD_TARGET_COLUMN
from ..preprocessing.clean_yield_data import load_yield_data
from ..training.train_yield_model import YieldRegressor


def evaluate_yield_model(csv_path=None, model_path: Path = YIELD_MODEL_PATH) -> dict[str, Any]:
    bundle = torch.load(model_path, map_location="cpu", weights_only=False)
    data_frame = load_yield_data(csv_path or CROP_YIELD_DATA_PATH)
    feature_frame = data_frame[YIELD_FEATURE_COLUMNS]
    target_frame = data_frame[YIELD_TARGET_COLUMN].astype(np.float32)
    features = bundle["preprocessor"].transform(feature_frame).astype(np.float32)

    _, x_test, _, y_test = train_test_split(features, target_frame.to_numpy(), test_size=0.2, random_state=42)

    model = YieldRegressor(input_dim=bundle["input_dim"])
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()

    with torch.no_grad():
        predictions = model(torch.tensor(x_test, dtype=torch.float32)).numpy()

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with YIELD_METRICS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return {"metrics": metrics, "predictions": predictions.tolist(), "actual": y_test.tolist()}
