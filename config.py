# config.py
import os

# Model configurations
DEFAULT_MISTRAL_NAME = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
DEFAULT_COLQWEN_NAME = "vidore/colqwen2-v1.0"

# Path configurations
DATA_DIR = "/bee_pdf"
DATABASE_DIR = "/db_rag_advan"
MISTRAL_MODELS_DIR = "/Mistral"
COLQWEN_MODELS_DIR = "/ColQwen"
TEMP_UPLOAD_DIR = f"{DATA_DIR}/temp_uploads"
HEATMAP_DIR = f"{DATA_DIR}/heatmaps"
PDF_IMAGES_DIR = f"{DATA_DIR}/pdf_images"

# Database configuration
DB_PATH = os.path.join(DATABASE_DIR, 'image_analysis.db')

# Feature flags
USE_SECONDARY_GPU_FOR_MAPS = os.environ.get("USE_SECONDARY_GPU", "False").lower() == "true"
SKIP_SIMILARITY_MAPS = os.environ.get("SKIP_SIMILARITY_MAPS", "False").lower() == "true"

# App metadata
USERNAME = "c123ian"
APP_NAME = "polliknow-rag"