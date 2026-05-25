from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
GRAPHS_DIR = OUTPUTS_DIR / "graphs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

CROP_YIELD_DATA_PATH = DATA_DIR / "crop_yield" / "faostat_ethiopia_yield.csv"
YIELD_MODEL_PATH = MODELS_DIR / "yield_regression_model.pth"
DISEASE_MODEL_PATH = MODELS_DIR / "resnet18_disease_model.pth"
DISEASE_MODELS_DIR = MODELS_DIR / "disease"
DISEASE_MODEL_URL = ""  # Set to a public URL to enable runtime download (leave empty to disable)
YIELD_METRICS_PATH = REPORTS_DIR / "yield_metrics.json"
DISEASE_METRICS_PATH = REPORTS_DIR / "disease_metrics.json"
YIELD_GRAPH_PATH = GRAPHS_DIR / "yield_training_curves.png"
YIELD_PREDICTION_GRAPH_PATH = GRAPHS_DIR / "yield_predicted_vs_actual.png"
DISEASE_CONFUSION_MATRIX_PATH = GRAPHS_DIR / "disease_confusion_matrix.png"

CROP_TYPES = ["Teff", "Wheat", "Maize", "Coffee", "Sorghum", "Barley", "Millet", "Chickpea"]
DISEASE_SUPPORTED_CROPS = ["Coffee", "Wheat", "Maize", "Sorghum"]
# Expanded features for stronger modeling
YIELD_FEATURE_COLUMNS = [
	"crop_type",
	"year",
	"area_harvested_ha",
	"region",
	"rainfall_mm",
	"temperature_c",
	"humidity",
	"fertilizer_kg_per_ha",
]
YIELD_NUMERIC_COLUMNS = ["year", "area_harvested_ha", "rainfall_mm", "temperature_c", "humidity", "fertilizer_kg_per_ha"]
YIELD_CATEGORICAL_COLUMNS = ["crop_type", "region"]
YIELD_TARGET_COLUMN = "yield_kg_per_ha"
DISEASE_CLASS_NAMES = [
	"teff_healthy",
	"teff_leaf_rust",
	"wheat_healthy",
	"wheat_powdery_mildew",
	"wheat_septoria",
	"maize_healthy",
	"maize_leaf_blight",
	"maize_rust",
	"coffee_healthy",
	"coffee_leaf_rust",
	"sorghum_healthy",
	"sorghum_leaf_blight",
	"barley_healthy",
	"barley_powdery_mildew",
	"millet_healthy",
	"millet_leaf_spot",
	"chickpea_healthy",
	"chickpea_blight",
]

# Example Ethiopian regions
REGIONS = ["Oromia", "Amhara", "Tigray", "SNNPR", "Addis_Ababa"]


def normalize_crop_name(crop_type: str) -> str:
	return crop_type.strip().lower().replace(" ", "_")


def get_disease_model_path(crop_type: str) -> Path:
	return DISEASE_MODELS_DIR / f"{normalize_crop_name(crop_type)}_disease_model.pth"
