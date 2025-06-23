import os

# Root directory of the project
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Arcene files
ARCENE_TRAIN_DATA = os.path.join(RAW_DATA_DIR, "arcene_train.data")
ARCENE_TRAIN_LABELS = os.path.join(RAW_DATA_DIR, "arcene_train.labels")
ARCENE_TEST_DATA = os.path.join(RAW_DATA_DIR, "arcene_valid.data")
ARCENE_TEST_LABELS = os.path.join(RAW_DATA_DIR, "arcene_valid.labels")

# Outputs (optional)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")