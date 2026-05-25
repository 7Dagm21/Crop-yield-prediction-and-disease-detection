from __future__ import annotations

from itertools import cycle

from pathlib import Path
import pandas as pd
import requests
import tempfile
import shutil
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import (
    CROP_YIELD_DATA_PATH,
    YIELD_CATEGORICAL_COLUMNS,
    YIELD_FEATURE_COLUMNS,
    YIELD_NUMERIC_COLUMNS,
    YIELD_TARGET_COLUMN,
    REGIONS,
)


def _build_demo_yield_data() -> pd.DataFrame:
    crop_sequence = ["Teff", "Wheat", "Maize", "Coffee", "Sorghum", "Barley", "Millet", "Chickpea"]
    region_cycle = cycle(REGIONS)

    rows = []
    for index, crop_type in enumerate(crop_sequence, start=1):
        region = next(region_cycle)
        for year in range(2018, 2024):
            rows.append(
                {
                    "crop_type": crop_type,
                    "year": year,
                    "area_harvested_ha": 1200 + index * 140 + (year - 2018) * 55,
                    "region": region,
                    "rainfall_mm": 680 + index * 18 + (year - 2018) * 12,
                    "temperature_c": 18 + (index % 4) + (year - 2018) * 0.2,
                    "humidity": 0.42 + (index % 5) * 0.04,
                    "fertilizer_kg_per_ha": 35 + index * 4 + (year - 2018) * 1.5,
                    "yield_kg_per_ha": 1600 + index * 110 + (year - 2018) * 75,
                }
            )

    return pd.DataFrame(rows)


def load_yield_data(csv_path=None) -> pd.DataFrame:
    csv_file = Path(csv_path or CROP_YIELD_DATA_PATH)

    # If CSV missing, try downloading from configured public URL (set in config)
    from ..config import CROP_YIELD_DATA_URL

    if not pd.io.common.file_exists(csv_file):
        if CROP_YIELD_DATA_URL:
            try:
                csv_file.parent.mkdir(parents=True, exist_ok=True)
                tmpfd, tmpname = tempfile.mkstemp(suffix=".csv")
                with requests.get(CROP_YIELD_DATA_URL, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    with open(tmpname, "wb") as f:
                        shutil.copyfileobj(r.raw, f)
                os.close(tmpfd)
                shutil.move(tmpname, str(csv_file))
            except Exception:
                # fall back to demo data if download fails
                return _build_demo_yield_data()
        else:
            return _build_demo_yield_data()

    data_frame = pd.read_csv(csv_file)
    required_columns = list(YIELD_FEATURE_COLUMNS) + [YIELD_TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in data_frame.columns]
    # If missing optional climate/region columns, fill with sensible defaults
    for col in missing_columns:
        if col == "region":
            data_frame["region"] = REGIONS[0]
        elif col == "rainfall_mm":
            data_frame["rainfall_mm"] = data_frame.get("rainfall_mm", pd.Series(0.0))
        elif col == "temperature_c":
            data_frame["temperature_c"] = data_frame.get("temperature_c", pd.Series(20.0))
        elif col == "humidity":
            data_frame["humidity"] = data_frame.get("humidity", pd.Series(0.5))
        elif col == "fertilizer_kg_per_ha":
            data_frame["fertilizer_kg_per_ha"] = data_frame.get("fertilizer_kg_per_ha", pd.Series(0.0))
        else:
            # For any other missing required column, raise an error
            raise ValueError(f"Missing yield column: {col}")

    # Ensure types
    for num_col in YIELD_NUMERIC_COLUMNS:
        data_frame[num_col] = pd.to_numeric(data_frame[num_col], errors="coerce").fillna(0.0)

    return data_frame


def build_yield_preprocessor() -> ColumnTransformer:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            ("categorical", encoder, YIELD_CATEGORICAL_COLUMNS),
            ("numerical", StandardScaler(), YIELD_NUMERIC_COLUMNS),
        ]
    )
