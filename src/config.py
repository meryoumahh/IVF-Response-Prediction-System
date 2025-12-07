"""
Centralized configuration for file paths.
All paths are resolved relative to the project root directory.
"""
import os

# Get the project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
PDF_REPORTS_DIR = os.path.join(RAW_DATA_DIR, "pdf_reports")

# Data files
RAW_PATIENTS_CSV = os.path.join(RAW_DATA_DIR, "patients.csv")
CLEAN_DATASET_EXP1_CSV = os.path.join(PROCESSED_DATA_DIR, "clean_dataset_experience1.csv")
CLEAN_DATASET_EXP2_CSV = os.path.join(PROCESSED_DATA_DIR, "clean_dataset_experience2.csv")

# Model directory and files
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_FILE = os.path.join(MODELS_DIR, "ivf_prediction_model.pkl")

# Output directories
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
