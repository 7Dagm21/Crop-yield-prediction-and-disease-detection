from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CROP_TYPES, REGIONS, YIELD_TARGET_COLUMN


def generate_sample_yield_dataframe(rows: int = 240, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    crop_effects = {
        "Teff": 1.6,
        "Wheat": 2.6,
        "Maize": 4.2,
        "Coffee": 1.1,
        "Sorghum": 2.3,
        "Barley": 2.2,
        "Millet": 1.8,
        "Chickpea": 2.1,
    }
    region_effects = {
        "Oromia": 0.45,
        "Amhara": 0.35,
        "SNNPR": 0.30,
        "Tigray": -0.05,
        "Afar": -0.35,
        "Somali": -0.28,
        "Benishangul-Gumuz": 0.18,
        "Addis_Ababa": 0.22,
    }

    records: list[dict[str, float | str]] = []
    for _ in range(rows):
        crop_type = str(rng.choice(CROP_TYPES))
        region = str(rng.choice(REGIONS))

        crop_rainfall_center = {
            "Teff": 780,
            "Wheat": 860,
            "Maize": 940,
            "Coffee": 1280,
            "Sorghum": 700,
            "Barley": 820,
            "Millet": 650,
            "Chickpea": 520,
        }[crop_type]
        rainfall_mm = float(np.clip(rng.normal(crop_rainfall_center, 120), 300, 1700))
        humidity_percent = float(np.clip(rng.normal(58 if crop_type != "Coffee" else 72, 11), 18, 95))
        temperature_c = float(np.clip(rng.normal(22 if crop_type in {"Teff", "Wheat", "Sorghum", "Barley", "Millet"} else 24, 3.5), 10, 35))
        fertilizer_kg_per_hectare = float(np.clip(rng.normal(74, 18), 15, 160))

        yield_value = (
            crop_effects[crop_type]
            + region_effects[region]
            + (rainfall_mm - 700) * 0.0024
            + (humidity_percent - 55) * 0.015
            - abs(temperature_c - 22) * 0.06
            + fertilizer_kg_per_hectare * 0.018
            + rng.normal(0, 0.22)
        )

        records.append(
            {
                "crop_type": crop_type,
                "region": region,
                "rainfall_mm": round(rainfall_mm, 1),
                "humidity_percent": round(humidity_percent, 1),
                "temperature_c": round(temperature_c, 1),
                "fertilizer_kg_per_hectare": round(fertilizer_kg_per_hectare, 1),
                YIELD_TARGET_COLUMN: round(max(0.4, yield_value), 2),
            }
        )

    return pd.DataFrame.from_records(records)


def generate_supplemental_yield_dataframe(rows_per_crop: int = 80, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict[str, float | str]] = []

    crop_profiles = {
        "Teff": {"yield_center": 1.7, "area_center": 180000.0, "rainfall": 760, "temp": 22.0},
        "Wheat": {"yield_center": 2.7, "area_center": 240000.0, "rainfall": 860, "temp": 21.5},
        "Maize": {"yield_center": 4.4, "area_center": 310000.0, "rainfall": 940, "temp": 24.0},
        "Coffee": {"yield_center": 1.15, "area_center": 260000.0, "rainfall": 1280, "temp": 23.5},
        "Sorghum": {"yield_center": 2.35, "area_center": 210000.0, "rainfall": 700, "temp": 23.0},
        "Barley": {"yield_center": 2.25, "area_center": 150000.0, "rainfall": 820, "temp": 20.5},
        "Millet": {"yield_center": 1.9, "area_center": 120000.0, "rainfall": 650, "temp": 24.0},
        "Chickpea": {"yield_center": 2.15, "area_center": 95000.0, "rainfall": 520, "temp": 23.0},
    }

    region_effects = {
        "Oromia": 0.42,
        "Amhara": 0.30,
        "SNNPR": 0.28,
        "Tigray": -0.02,
        "Afar": -0.30,
        "Somali": -0.25,
        "Benishangul-Gumuz": 0.16,
        "Addis_Ababa": 0.14,
    }

    for crop_type, profile in crop_profiles.items():
        for _ in range(rows_per_crop):
            region = str(rng.choice(REGIONS))
            year = int(rng.integers(1993, 2027))
            area_harvested_ha = float(np.clip(rng.normal(profile["area_center"], profile["area_center"] * 0.18), 2500, 1200000))
            rainfall_mm = float(np.clip(rng.normal(profile["rainfall"], 110), 250, 1800))
            temperature_c = float(np.clip(rng.normal(profile["temp"], 2.8), 10, 35))
            humidity = float(np.clip(rng.normal(0.58 if crop_type != "Coffee" else 0.72, 0.08), 0.18, 0.95))
            fertilizer_kg_per_ha = float(np.clip(rng.normal(72, 16), 12, 180))

            yield_kg_per_ha = (
                profile["yield_center"]
                + region_effects[region]
                + (rainfall_mm - profile["rainfall"]) * 0.0022
                + (humidity - 0.55) * 1.3
                - abs(temperature_c - profile["temp"]) * 0.07
                + fertilizer_kg_per_ha * 0.017
                + rng.normal(0, 0.18)
            )

            records.append(
                {
                    "crop_type": crop_type,
                    "year": year,
                    "area_harvested_ha": round(area_harvested_ha, 1),
                    "region": region,
                    "rainfall_mm": round(rainfall_mm, 1),
                    "temperature_c": round(temperature_c, 1),
                    "humidity": round(humidity, 3),
                    "fertilizer_kg_per_ha": round(fertilizer_kg_per_ha, 1),
                    YIELD_TARGET_COLUMN: round(max(0.35, yield_kg_per_ha), 2),
                }
            )

    return pd.DataFrame.from_records(records)


def ensure_sample_yield_csv(csv_path: Path, rows: int = 240) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        generate_sample_yield_dataframe(rows=rows).to_csv(csv_path, index=False)
    return csv_path
