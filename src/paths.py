import os

from pathlib import Path

# Automatically find the root by going up from this file's location
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Define standard subfolders
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEAN_DATA_DIR = DATA_DIR / "clean"
MODELS_DIR = DATA_DIR / "models"
## Define data
SEISMIC_DATA = RAW_DATA_DIR / "seismic-bumps.arff"

SEED = 202506