from pathlib import Path

# Find the base directory
SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent

# Set up directory paths
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
RESULTS_DIR = STATIC_DIR / "results"
UPLOAD_DIR = BASE_DIR / "uploads"

# Create any missing directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Path to the trained models
YOLO_MODEL_PATH = MODELS_DIR / "conjunctiva_detector.pt"
KERAS_MODEL_PATH = MODELS_DIR / "anemia_model.h5"
