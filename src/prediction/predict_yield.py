from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import joblib

from ..config import MODELS_DIR, YIELD_FEATURE_COLUMNS, YIELD_MODEL_PATH
from ..preprocessing.clean_yield_data import load_yield_data
from ..training.train_yield_model import YieldRegressor, train_yield_model


def load_yield_bundle(model_path: Path = YIELD_MODEL_PATH) -> dict[str, Any]:
    if not model_path.exists():
        train_yield_model()
    bundle = torch.load(model_path, map_location="cpu", weights_only=False)
    # Try loading separate preprocessor first
    preprocessor_path = MODELS_DIR / "preprocessor.pkl"
    try:
        preprocessor = joblib.load(preprocessor_path)
        bundle["preprocessor"] = preprocessor
    except Exception:
        # keep existing preprocessor in bundle if present
        pass
    return bundle


def _historical_feature_defaults(crop_type: str, region: str | None = None) -> dict[str, float | str]:
    data_frame = load_yield_data()
    defaults: dict[str, float | str] = {
        "region": region or str(data_frame["region"].mode().iloc[0]),
        "rainfall_mm": float(data_frame.get("rainfall_mm", pd.Series([0.0])).median()),
        "temperature_c": float(data_frame.get("temperature_c", pd.Series([20.0])).median()),
        "humidity": float(data_frame.get("humidity", pd.Series([0.5])).median()),
        "fertilizer_kg_per_ha": float(data_frame.get("fertilizer_kg_per_ha", pd.Series([0.0])).median()),
        "area_harvested_ha": float(data_frame.get("area_harvested_ha", pd.Series([0.0])).median()),
    }

    crop_slice = data_frame[data_frame["crop_type"] == crop_type]
    if region is not None and "region" in crop_slice.columns:
        region_slice = crop_slice[crop_slice["region"] == region]
        if not region_slice.empty:
            crop_slice = region_slice

    if not crop_slice.empty:
        for column in ("area_harvested_ha", "rainfall_mm", "temperature_c", "humidity", "fertilizer_kg_per_ha"):
            if column in crop_slice.columns:
                defaults[column] = float(pd.to_numeric(crop_slice[column], errors="coerce").dropna().median())

    return defaults


def predict_yield(
    crop_type: str,
    year: int,
    area_harvested_ha: float | None = None,
    region: str | None = None,
    rainfall_mm: float | None = None,
    temperature_c: float | None = None,
    humidity: float | None = None,
    fertilizer_kg_per_ha: float | None = None,
) -> dict[str, float | str]:
    bundle = load_yield_bundle()
    preprocessor = bundle.get("preprocessor")
    if preprocessor is None:
        raise FileNotFoundError("Preprocessor missing. Train the yield model first to generate the preprocessor.")

    inferred_defaults = _historical_feature_defaults(crop_type, region)
    # Build input frame consistent with YIELD_FEATURE_COLUMNS
    row = {k: None for k in YIELD_FEATURE_COLUMNS}
    row.update({
        "crop_type": crop_type,
        "year": year,
        "area_harvested_ha": area_harvested_ha if area_harvested_ha is not None else inferred_defaults["area_harvested_ha"],
    })
    row["region"] = region or inferred_defaults["region"]
    row["rainfall_mm"] = rainfall_mm if rainfall_mm is not None else inferred_defaults["rainfall_mm"]
    row["temperature_c"] = temperature_c if temperature_c is not None else inferred_defaults["temperature_c"]
    row["humidity"] = humidity if humidity is not None else inferred_defaults["humidity"]
    row["fertilizer_kg_per_ha"] = (
        fertilizer_kg_per_ha if fertilizer_kg_per_ha is not None else inferred_defaults["fertilizer_kg_per_ha"]
    )

    input_frame = pd.DataFrame([row], columns=YIELD_FEATURE_COLUMNS)
    features = preprocessor.transform(input_frame).astype(np.float32)

    model = YieldRegressor(input_dim=bundle["input_dim"])
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()

    with torch.no_grad():
        prediction = float(model(torch.tensor(features, dtype=torch.float32)).item())

    rmse = float(bundle.get("metrics", {}).get("rmse", 0.5))
    lower_bound = max(0.0, prediction - 1.96 * rmse)
    upper_bound = prediction + 1.96 * rmse

    return {
        "predicted_yield": round(prediction, 2),
        "lower_bound": round(lower_bound, 2),
        "upper_bound": round(upper_bound, 2),
        "model_rmse": round(rmse, 2),
        "method": "pytorch_mlp",
    }
