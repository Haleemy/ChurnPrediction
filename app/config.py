"""
config.py — Central configuration for the Churn Analyst project.
All paths, constants, and environment settings live here.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ── Root directories ──────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv(ROOT_DIR / ".env")

# ── LLM Configuration ─────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
MODEL_NAME: str = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))
LLM_MAX_RETRIES: int = 2  # Hard limit — free-tier rate limits

# ── Dataset Configuration ─────────────────────────────────────────────────────
DATA_PATH: Path = Path(os.getenv("DATA_PATH", str(DATA_DIR / "Customer-Churn.csv")))
TARGET_COLUMN: str = "Churn"
ID_COLUMN: str = "customerID"

# ── Model Configuration ────────────────────────────────────────────────────────
MODEL_PATH: Path = MODELS_DIR / "churn_pipeline.joblib"
METADATA_PATH: Path = MODELS_DIR / "model_metadata.json"
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
CV_FOLDS: int = 5

# ── Prediction Thresholds ─────────────────────────────────────────────────────
# These are determined during training via threshold analysis; defaults here
DEFAULT_THRESHOLD: float = 0.50
HIGH_RISK_THRESHOLD: float = 0.70
MEDIUM_RISK_THRESHOLD: float = 0.40

# ── Risk Labels ───────────────────────────────────────────────────────────────
RISK_LEVELS = {
    "High": (HIGH_RISK_THRESHOLD, 1.01),
    "Medium": (MEDIUM_RISK_THRESHOLD, HIGH_RISK_THRESHOLD),
    "Low": (0.0, MEDIUM_RISK_THRESHOLD),
}

def get_risk_level(score: float) -> str:
    for level, (low, high) in RISK_LEVELS.items():
        if low <= score < high:
            return level
    return "High"

# ── Feature Definitions ───────────────────────────────────────────────────────
NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE: Path = LOGS_DIR / "churn_analyst.log"
