# common_image.py
import modal
import os

# Create shared volumes
BEE_VOLUME = modal.Volume.from_name("bee_pdf", create_if_missing=True)
DB_VOLUME = modal.Volume.from_name("db_data", create_if_missing=True)
MISTRAL_VOLUME = modal.Volume.lookup("Mistral", create_if_missing=False)
COLQWEN_VOLUME = modal.Volume.lookup("ColQwen", create_if_missing=False)

# Define paths
DATA_DIR = "/bee_pdf"
DATABASE_DIR = "/db_rag_advan"
MISTRAL_MODELS_DIR = "/Mistral"
COLQWEN_MODELS_DIR = "/ColQwen"
TEMP_UPLOAD_DIR = f"{DATA_DIR}/temp_uploads"
HEATMAP_DIR = f"{DATA_DIR}/heatmaps"
PDF_IMAGES_DIR = f"{DATA_DIR}/pdf_images"

# Create common image with all dependencies
common_image = modal.Image.debian_slim(python_version="3.10") \
    .apt_install("libgl1-mesa-glx","libglib2.0-0","libsm6","libxrender1","libxext6","poppler-utils") \
    .pip_install(
        "vllm==0.8.1",
        "mistral_common>=1.5.4",
        "python-fasthtml==0.4.3",
        "aiohttp",
        "faiss-cpu",
        "sentence-transformers",
        "pandas",
        "numpy",
        "huggingface_hub",
        "transformers>=4.48.3",
        "rerankers",
        "sqlite-minutils",
        "rank-bm25",
        "nltk",
        "sqlalchemy",
        "pdf2image",
        "colpali-engine[interpretability]>=0.3.2",
        "torch",
        "matplotlib"
    ) \
    .env({"VLLM_USE_V1": "0"})  # Set VLLM_USE_V1=0 to fix Mistral compatibility