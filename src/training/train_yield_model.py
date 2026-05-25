from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import joblib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from ..config import (
    CROP_YIELD_DATA_PATH,
    GRAPHS_DIR,
    REPORTS_DIR,
    YIELD_FEATURE_COLUMNS,
    YIELD_MODEL_PATH,
    YIELD_NUMERIC_COLUMNS,
    YIELD_TARGET_COLUMN,
)
from ..config import MODELS_DIR
from ..data_generation import generate_supplemental_yield_dataframe
from ..preprocessing.clean_yield_data import build_yield_preprocessor, load_yield_data


class YieldRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _save_training_plot(history: dict[str, list[float]], graph_path: Path) -> None:
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(history["train_loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Validation loss")
    plt.title("Yield Model Training Curves")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(graph_path, dpi=160)
    plt.close()


def _save_prediction_plot(actual: np.ndarray, predicted: np.ndarray, graph_path: Path) -> None:
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(actual, predicted, alpha=0.8, color="#136f63")
    line_min = min(actual.min(), predicted.min())
    line_max = max(actual.max(), predicted.max())
    plt.plot([line_min, line_max], [line_min, line_max], linestyle="--", color="#ba181b")
    plt.xlabel("Actual yield")
    plt.ylabel("Predicted yield")
    plt.title("Predicted vs Actual Yield")
    plt.tight_layout()
    plt.savefig(graph_path, dpi=160)
    plt.close()


def _ensure_all_crop_coverage(data_frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    observed_crops = set(data_frame["crop_type"].astype(str).unique())
    missing_crops = [crop for crop in ["Teff", "Wheat", "Maize", "Coffee", "Sorghum", "Barley", "Millet", "Chickpea"] if crop not in observed_crops]
    if not missing_crops:
        return data_frame

    supplemental_frame = generate_supplemental_yield_dataframe(rows_per_crop=60, seed=seed)
    supplemental_frame = supplemental_frame[supplemental_frame["crop_type"].isin(missing_crops)]
    if supplemental_frame.empty:
        return data_frame

    combined = pd.concat([data_frame, supplemental_frame], ignore_index=True)
    return combined


def train_yield_model(
    csv_path=None,
    model_path: Path = YIELD_MODEL_PATH,
    epochs: int = 120,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seed: int = 42,
    patience: int = 10,
) -> dict[str, Any]:
    _set_seed(seed)
    data_frame = load_yield_data(csv_path or CROP_YIELD_DATA_PATH)
    data_frame = _ensure_all_crop_coverage(data_frame, seed=seed)
    feature_frame = data_frame[YIELD_FEATURE_COLUMNS]
    target_frame = data_frame[YIELD_TARGET_COLUMN].astype(np.float32)

    preprocessor = build_yield_preprocessor()
    features = preprocessor.fit_transform(feature_frame).astype(np.float32)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target_frame.to_numpy(),
        test_size=0.2,
        random_state=seed,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        random_state=seed,
    )

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    model = YieldRegressor(input_dim=x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_state = None
    best_val_loss = float("inf")
    no_improve_epochs = 0

    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(x_train_tensor.size(0))
        batch_losses = []

        for start in range(0, x_train_tensor.size(0), batch_size):
            batch_indices = permutation[start : start + batch_size]
            batch_inputs = x_train_tensor[batch_indices]
            batch_targets = y_train_tensor[batch_indices]

            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            train_loss = criterion(model(x_train_tensor), y_train_tensor).item()
            val_loss = criterion(model(x_val_tensor), y_val_tensor).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(x_test_tensor)

    y_pred = y_pred_tensor.numpy()
    test_mae = float(mean_absolute_error(y_test, y_pred))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    test_r2 = float(r2_score(y_test, y_pred))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": int(x_train.shape[1]),
            "preprocessor": preprocessor,
            "feature_columns": list(YIELD_FEATURE_COLUMNS),
            "target_column": YIELD_TARGET_COLUMN,
            "metrics": {"mae": test_mae, "rmse": test_rmse, "r2": test_r2},
            "history": history,
            "seed": seed,
            "epochs": epochs,
        },
        model_path,
    )

    # Save preprocessor separately to avoid pickle/unpickling issues with torch bundles
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    preprocessor_path = MODELS_DIR / "preprocessor.pkl"
    try:
        joblib.dump(preprocessor, preprocessor_path)
    except Exception:
        # best-effort fallback
        torch.save(preprocessor, str(preprocessor_path) + ".pth")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORTS_DIR / "yield_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump({"mae": test_mae, "rmse": test_rmse, "r2": test_r2}, handle, indent=2)

    _save_training_plot(history, GRAPHS_DIR / "yield_training_curves.png")
    _save_prediction_plot(y_test, y_pred, GRAPHS_DIR / "yield_predicted_vs_actual.png")

    return {
        "model_path": str(model_path),
        "metrics": {"mae": test_mae, "rmse": test_rmse, "r2": test_r2},
        "history": history,
        "test_actual": y_test.tolist(),
        "test_predicted": y_pred.tolist(),
    }


if __name__ == "__main__":
    results = train_yield_model()
    print(json.dumps(results["metrics"], indent=2))
