from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR      = PROJECT_ROOT / "src"
DATA_DIR     = PROJECT_ROOT / "data"
LOGS_DIR     = PROJECT_ROOT / "logs"
MODELS_DIR   = PROJECT_ROOT / "models"
NOTEBOOKS_DIR= PROJECT_ROOT / "notebooks"
PLOTS_DIR    = PROJECT_ROOT / "plots"
RESULTS_DIR  = PROJECT_ROOT / "results"
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"
TESTS_DIR    = PROJECT_ROOT / "tests"

for _dir in [DATA_DIR, LOGS_DIR, MODELS_DIR, NOTEBOOKS_DIR,
             PLOTS_DIR, RESULTS_DIR, SCRIPTS_DIR, TESTS_DIR]:
    _dir.mkdir(exist_ok=True)

ENV_FILE          = PROJECT_ROOT / ".env"
APP_ENTRYPOINT    = PROJECT_ROOT / "src" / "app.py"
MODEL_METRICS_FILE= RESULTS_DIR / "model_metrics.csv"

STREAMLIT_HOST = "localhost"
STREAMLIT_PORT = 8501

# ── Project metadata ───────────────────────────────────────────────────────────
PROJECT_TITLE = "Hydro-Alpha: USGS Streamflow → IDACORP Excess Return"
TARGET_TICKER  = "IDA"    # IDACORP — primary Snake River hydropower utility
BENCH_TICKER   = "XLU"   # Utilities sector ETF — benchmark to isolate sector alpha
FORWARD_DAYS   = 20       # prediction horizon (≈ 1 trading month)

# ── Registered models ──────────────────────────────────────────────────────────
MODELS = {
    "ridge": {
        "name": "Ridge Regression",
        "description": (
            "Linear baseline. Assumes a direct linear relationship between "
            "streamflow anomaly and forward excess return. "
            "Interpretable coefficients reveal which rivers drive the signal."
        ),
        "path": MODELS_DIR / "ridge.joblib",
    },
    "random_forest": {
        "name": "Random Forest",
        "description": (
            "Ensemble of 200 decision trees. Captures non-linear interactions "
            "between rivers and seasonal effects without explicit specification."
        ),
        "path": MODELS_DIR / "random_forest.joblib",
    },
    "xgboost": {
        "name": "XGBoost",
        "description": (
            "Gradient boosted trees. Best at capturing threshold effects "
            "(e.g. flow drops below a critical level → sharp margin compression)."
        ),
        "path": MODELS_DIR / "xgboost.joblib",
    },
}
