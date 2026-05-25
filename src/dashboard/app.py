from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import streamlit as st

from ..config import CROP_TYPES, CROP_YIELD_DATA_PATH, DISEASE_SUPPORTED_CROPS, YIELD_METRICS_PATH
from ..config import REGIONS, REPORTS_DIR, get_disease_model_path, normalize_crop_name
from ..preprocessing.clean_yield_data import load_yield_data
from ..prediction.predict_disease import predict_disease
from ..prediction.predict_yield import predict_yield


st.set_page_config(
    page_title="Ethiopian Crop Intelligence System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_real_yield_data() -> pd.DataFrame:
    return load_yield_data(CROP_YIELD_DATA_PATH)


@st.cache_data
def load_yield_training_metrics() -> dict[str, float]:
    if not YIELD_METRICS_PATH.exists():
        return {}
    with YIELD_METRICS_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data
def load_disease_training_summary() -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []

    for crop_type in DISEASE_SUPPORTED_CROPS:
        normalized_crop = normalize_crop_name(crop_type)
        model_path = get_disease_model_path(crop_type)
        report_path = REPORTS_DIR / f"disease_metrics_{normalized_crop}.json"

        class_rows: list[dict[str, Any]] = []
        source = "missing"
        metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        class_count = 0

        if report_path.exists():
            with report_path.open("r", encoding="utf-8") as handle:
                report = json.load(handle)

            weighted = report.get("weighted avg", {})
            metrics = {
                "accuracy": float(report.get("accuracy", 0.0)),
                "precision": float(weighted.get("precision", 0.0)),
                "recall": float(weighted.get("recall", 0.0)),
                "f1": float(weighted.get("f1-score", 0.0)),
            }
            class_rows = [
                {
                    "class_name": class_name,
                    "precision": float(class_metrics.get("precision", 0.0)),
                    "recall": float(class_metrics.get("recall", 0.0)),
                    "f1": float(class_metrics.get("f1-score", 0.0)),
                    "support": float(class_metrics.get("support", 0.0)),
                }
                for class_name, class_metrics in report.items()
                if isinstance(class_metrics, dict) and class_name not in {"macro avg", "weighted avg"}
            ]
            class_count = len(class_rows)
            source = "report"
        elif model_path.exists():
            bundle = torch.load(model_path, map_location="cpu")
            bundle_metrics = bundle.get("metrics", {})
            class_names = bundle.get("class_names", [])
            metrics = {
                "accuracy": float(bundle_metrics.get("accuracy", 0.0)),
                "precision": float(bundle_metrics.get("precision", 0.0)),
                "recall": float(bundle_metrics.get("recall", 0.0)),
                "f1": float(bundle_metrics.get("f1", 0.0)),
            }
            class_count = len(class_names)
            source = "checkpoint"

        summaries.append(
            {
                "crop_type": crop_type,
                "status": "trained" if source != "missing" else "pending",
                "source": source,
                "model_path": str(model_path),
                "report_path": str(report_path),
                "class_count": class_count,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "class_rows": class_rows,
            }
        )

    return summaries


def _render_metrics_summary(data_frame: pd.DataFrame) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Records", f"{len(data_frame)}")
    col2.metric("Crops", f"{data_frame['crop_type'].nunique()}")
    col3.metric("Years", f"{data_frame['year'].nunique()}")


def render_home() -> None:
    st.title("AI-Based Ethiopian Crop Intelligence System")
    st.markdown(
        """
        A portfolio-ready agriculture dashboard for Ethiopian crop intelligence.

        It combines:
        - crop yield prediction from official Ethiopian and climate datasets
        - crop disease detection from leaf images
        - analytics for Ethiopian grain and coffee production
        """
    )

    left, right = st.columns([1.2, 0.8])
    with left:
        st.subheader("Why this project stands out")
        st.write(
            "It uses a yield pipeline built from Ethiopian crop and climate data, a crop-specific disease classifier, and a clean Streamlit dashboard for inspection and forecasting."
        )
        st.info("Use the sidebar to move between prediction, disease detection, analytics, and project info.")
    with right:
        st.markdown(
            """
            <div style="padding: 1.1rem 1.2rem; border-radius: 18px; background: linear-gradient(135deg, #0b3d2e, #1b4332); color: white; min-height: 220px; display: flex; flex-direction: column; justify-content: space-between;">
              <div>
                <div style="font-size: 0.9rem; opacity: 0.82;">Official source</div>
                <div style="font-size: 1.6rem; font-weight: 700; line-height: 1.15; margin-top: 0.35rem;">FAOSTAT-backed crop yield modeling for Ethiopia.</div>
              </div>
              <div style="font-size: 0.95rem; opacity: 0.9;">
                Built to support yield forecasting, disease screening, and dashboard storytelling in one package.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_yield_prediction() -> None:
    st.header("Crop Yield Prediction")
    st.caption("Estimate yield using crop type, year, and region. Harvested area and climate values are inferred from historical data unless you choose to override them.")

    data_frame = load_real_yield_data()
    year_min = int(data_frame["year"].min())
    current_year = date.today().year

    with st.form("yield_prediction_form"):
        crop_type = st.selectbox("Crop type", CROP_TYPES)
        region = st.selectbox("Region", REGIONS)
        year = st.number_input("Year", min_value=year_min, value=max(year_min, current_year), step=1)

        with st.expander("Optional advanced inputs", expanded=False):
            st.caption("Leave these blank to use inferred historical defaults for a simpler experience.")
            area_harvested_ha = st.number_input("Harvested area (ha) - optional", min_value=0.0, value=0.0, step=1.0)
            use_custom_area = st.checkbox("Use custom harvested area", value=False)
            rainfall_mm = st.number_input("Rainfall (mm) - optional", value=0.0, step=1.0)
            temperature_c = st.number_input("Temperature (°C) - optional", value=0.0, step=0.5)
            humidity = st.number_input("Humidity (0-1) - optional", value=0.0, step=0.01)
            fertilizer_kg_per_ha = st.number_input("Fertilizer (kg/ha) - optional", value=0.0, step=1.0)
            use_custom_climate = st.checkbox("Use custom climate values", value=False)
        submitted = st.form_submit_button("Predict yield")

    if submitted:
        result = predict_yield(
            crop_type=crop_type,
            year=int(year),
            area_harvested_ha=float(area_harvested_ha) if 'use_custom_area' in locals() and use_custom_area else None,
            region=region,
            rainfall_mm=float(rainfall_mm) if 'use_custom_climate' in locals() and use_custom_climate else None,
            temperature_c=float(temperature_c) if 'use_custom_climate' in locals() and use_custom_climate else None,
            humidity=float(humidity) if 'use_custom_climate' in locals() and use_custom_climate else None,
            fertilizer_kg_per_ha=float(fertilizer_kg_per_ha) if 'use_custom_climate' in locals() and use_custom_climate else None,
        )
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Predicted yield", f"{result['predicted_yield']:.2f} kg/ha")
        metric_col2.metric("Likely range", f"{result['lower_bound']:.2f} - {result['upper_bound']:.2f}")
        metric_col3.metric("Model RMSE", f"{result['model_rmse']:.2f}")
        st.success(f"Prediction method: {result['method']}")


def render_disease_detection() -> None:
    st.header("Disease Detection")
    st.caption("Choose the crop, upload a leaf image, and the system will return a crop-specific disease label plus confidence.")
    st.caption("Current disease model coverage: Coffee, Wheat, Maize, and Sorghum. Teff remains future work until a real image set is available.")

    crop_type = st.selectbox("Crop type", DISEASE_SUPPORTED_CROPS)
    uploaded_file = st.file_uploader("Upload a crop leaf image", type=["png", "jpg", "jpeg", "webp"])
    if uploaded_file is None:
        st.info("Upload a leaf image to get a prediction.")
        return

    try:
        uploaded_file.seek(0)
        result = predict_disease(uploaded_file, crop_type=crop_type)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    if result.get("model_source") == "legacy_generic":
        st.warning(
            "No crop-specific checkpoint is available yet, so this result comes from the legacy PlantVillage disease model. Train a crop-specific model for Coffee, Wheat, Maize, or Sorghum to replace it."
        )
    left, right = st.columns([0.9, 1.1])
    with left:
        st.image(uploaded_file, caption="Uploaded leaf image", width="stretch")
    with right:
        st.metric("Predicted disease", result["disease"])
        st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
        st.write(f"Prediction method: {result['method']}")
        probability_frame = pd.DataFrame(
            {"Disease": list(result["probabilities"].keys()), "Probability": list(result["probabilities"].values())}
        )
        st.plotly_chart(px.bar(probability_frame, x="Disease", y="Probability", title="Class confidence"), width="stretch")


def render_analytics() -> None:
    st.header("Analytics Dashboard")
    data_frame = load_real_yield_data()
    _render_metrics_summary(data_frame)

    yield_metrics = load_yield_training_metrics()
    if yield_metrics:
        st.subheader("Yield model training results")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("MAE", f"{yield_metrics.get('mae', 0.0):.2f}")
        metric_col2.metric("RMSE", f"{yield_metrics.get('rmse', 0.0):.2f}")
        metric_col3.metric("R²", f"{yield_metrics.get('r2', 0.0):.3f}")

    st.subheader("Yield analysis across all crop types")
    yield_summary = (
        data_frame.groupby("crop_type", as_index=False)
        .agg(
            records=("yield_kg_per_ha", "size"),
            average_yield=("yield_kg_per_ha", "mean"),
            average_area=("area_harvested_ha", "mean"),
        )
        .sort_values("average_yield", ascending=False)
    )

    yield_left, yield_right = st.columns(2)
    with yield_left:
        st.plotly_chart(
            px.bar(
                yield_summary,
                x="crop_type",
                y="average_yield",
                title="Average Yield by Crop",
                color="average_yield",
                text_auto=".1f",
            ),
            width="stretch",
        )
    with yield_right:
        st.plotly_chart(
            px.bar(
                yield_summary,
                x="crop_type",
                y="records",
                title="Training Records by Crop",
                color="records",
                text_auto=True,
            ),
            width="stretch",
        )

    st.dataframe(
        yield_summary.rename(
            columns={
                "crop_type": "Crop",
                "records": "Records",
                "average_yield": "Average yield (kg/ha)",
                "average_area": "Average harvested area (ha)",
            }
        ).reset_index(drop=True),
        width="stretch",
        hide_index=True,
    )

    yield_line = px.scatter(
        data_frame,
        x="year",
        y="yield_kg_per_ha",
        color="crop_type",
        title="Yield by Year",
        hover_data=["area_harvested_ha"],
    )
    crop_bar = px.bar(
        data_frame.groupby("crop_type", as_index=False)["yield_kg_per_ha"].mean(),
        x="crop_type",
        y="yield_kg_per_ha",
        title="Average Yield by Crop",
    )
    area_bar = px.bar(
        data_frame.groupby("crop_type", as_index=False)["area_harvested_ha"].mean(),
        x="crop_type",
        y="area_harvested_ha",
        title="Average Harvested Area by Crop",
    )
    corr_matrix = data_frame[["year", "area_harvested_ha", "yield_kg_per_ha"]].corr()
    heatmap = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index, colorscale="Viridis"))
    heatmap.update_layout(title="Correlation Heatmap")

    top_left, top_right = st.columns(2)
    with top_left:
        st.plotly_chart(yield_line, width="stretch")
    with top_right:
        st.plotly_chart(crop_bar, width="stretch")
    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        st.plotly_chart(area_bar, width="stretch")
    with bottom_right:
        st.plotly_chart(heatmap, width="stretch")

    st.subheader("Disease model performance by crop")
    disease_summary = pd.DataFrame(load_disease_training_summary())
    if disease_summary.empty:
        st.info("No disease model summaries are available yet.")
        return

    trained_count = int((disease_summary["status"] == "trained").sum())
    best_row = disease_summary.sort_values("accuracy", ascending=False).iloc[0]
    disease_metric_col1, disease_metric_col2, disease_metric_col3, disease_metric_col4 = st.columns(4)
    disease_metric_col1.metric("Trained crops", f"{trained_count}/{len(disease_summary)}")
    disease_metric_col2.metric("Best crop", str(best_row["crop_type"]))
    disease_metric_col3.metric("Best accuracy", f"{best_row['accuracy'] * 100:.1f}%")
    disease_metric_col4.metric("Average F1", f"{disease_summary['f1'].mean() * 100:.1f}%")

    disease_chart_frame = disease_summary[["crop_type", "accuracy", "f1"]].melt(
        id_vars="crop_type",
        value_vars=["accuracy", "f1"],
        var_name="metric",
        value_name="score",
    )
    st.plotly_chart(
        px.bar(
            disease_chart_frame,
            x="crop_type",
            y="score",
            color="metric",
            barmode="group",
            title="Disease model accuracy and F1 by crop",
            text_auto=".2f",
        ),
        width="stretch",
    )

    st.dataframe(
        disease_summary[["crop_type", "status", "source", "class_count", "accuracy", "precision", "recall", "f1"]].rename(
            columns={
                "crop_type": "Crop",
                "status": "Status",
                "source": "Source",
                "class_count": "Classes",
                "accuracy": "Accuracy",
                "precision": "Precision",
                "recall": "Recall",
                "f1": "F1",
            }
        ).reset_index(drop=True),
        width="stretch",
        hide_index=True,
    )

    for _, summary_row in disease_summary.iterrows():
        class_rows = summary_row["class_rows"]
        if not class_rows:
            continue
        with st.expander(f"{summary_row['crop_type']} class breakdown"):
            st.dataframe(
                pd.DataFrame(class_rows).rename(
                    columns={
                        "class_name": "Class",
                        "precision": "Precision",
                        "recall": "Recall",
                        "f1": "F1",
                        "support": "Support",
                    }
                ),
                width="stretch",
                hide_index=True,
            )


def render_about() -> None:
    st.header("About the Project")
    st.write(
        "The yield module uses Ethiopian crop and climate records, and the disease module is designed around crop-specific image folders for Ethiopian grains and coffee."
    )
    st.markdown(
        """
        **Included modules**
        - crop yield prediction with a PyTorch MLP trained on Ethiopian crop rows
        - disease detection with ResNet18 transfer learning
        - analytics and visual reporting
        - a documented, source-backed crop and disease dataset strategy
        """
    )
    st.warning("Dataset quality determines model quality: use crop-specific disease folders and Ethiopian yield sources for production training.")


def run_app() -> None:
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Yield Prediction", "Disease Detection", "Analytics", "About"],
    )

    if page == "Home":
        render_home()
    elif page == "Yield Prediction":
        render_yield_prediction()
    elif page == "Disease Detection":
        render_disease_detection()
    elif page == "Analytics":
        render_analytics()
    else:
        render_about()


if __name__ == "__main__":
    run_app()
