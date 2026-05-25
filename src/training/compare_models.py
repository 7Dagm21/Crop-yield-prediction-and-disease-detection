"""Compare LinearRegression, RandomForest, and PyTorch MLP on the yield dataset."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from ..config import CROP_YIELD_DATA_PATH, YIELD_FEATURE_COLUMNS, YIELD_TARGET_COLUMN, MODELS_DIR
from ..preprocessing.clean_yield_data import load_yield_data, build_yield_preprocessor

try:
    from .train_yield_model import train_yield_model, YieldRegressor
    import torch
except Exception:
    YieldRegressor = None
    train_yield_model = None
    torch = None


def compare(save_reports: bool = True) -> dict:
    df = load_yield_data(CROP_YIELD_DATA_PATH)
    X = df[YIELD_FEATURE_COLUMNS]
    y = df[YIELD_TARGET_COLUMN].astype(float).to_numpy()

    preprocessor = build_yield_preprocessor()
    X_proc = preprocessor.fit_transform(X).astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=0.2, random_state=42)

    results = {}

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)
    results['linear'] = {
        'mae': float(mean_absolute_error(y_test, pred_lr)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, pred_lr))),
        'r2': float(r2_score(y_test, pred_lr)),
    }

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    results['random_forest'] = {
        'mae': float(mean_absolute_error(y_test, pred_rf)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, pred_rf))),
        'r2': float(r2_score(y_test, pred_rf)),
    }

    # PyTorch model: train quickly or load existing
    if train_yield_model is not None:
        try:
            # train a quick model with fewer epochs for comparison
            bundle = train_yield_model(epochs=30)
            # use test predictions from bundle
            results['pytorch'] = bundle['metrics']
        except Exception:
            results['pytorch'] = {'mae': None, 'rmse': None, 'r2': None}
    else:
        results['pytorch'] = {'mae': None, 'rmse': None, 'r2': None}

    if save_reports:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = MODELS_DIR / 'model_comparison.json'
        with report_path.open('w', encoding='utf-8') as fh:
            json.dump(results, fh, indent=2)

    return results


if __name__ == '__main__':
    print(json.dumps(compare(), indent=2))
