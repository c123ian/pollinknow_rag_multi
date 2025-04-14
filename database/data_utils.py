# data_utils.py
import os
import logging
import torch
import pickle
import pandas as pd
from typing import Tuple, List, Dict, Any

from colpali_engine.models import ColQwen2, ColQwen2Processor

from config import (
    DATA_DIR, COLQWEN_MODELS_DIR, DEFAULT_COLQWEN_NAME,
    PDF_IMAGES_DIR
)

def ensure_colpali_model_loaded() -> Tuple[Any, Any]:
    """Load the ColPali/ColQwen2 model and return the model and processor
    
    Returns:
        Tuple of (model, processor)
    
    Raises:
        RuntimeError: If loading fails
    """
    # Verify CUDA is available
    if torch.cuda.is_available():
        logging.info(f"CUDA is available: {torch.cuda.get_device_name(0)}")
    else:
        logging.warning("WARNING: CUDA is NOT available, ColQwen will run on CPU which is slower")
    
    logging.info(f"Loading ColQwen2 model...")
    try:
        # First check if files exist directly in the volume root
        model_files_in_root = False
        volume_root = COLQWEN_MODELS_DIR
        required_files = ["tokenizer.json", "adapter_model.safetensors", "special_tokens_map.json"]
        
        if os.path.exists(volume_root):
            root_files = os.listdir(volume_root)
            if all(file in root_files for file in required_files):
                model_files_in_root = True
                logging.info(f"Found model files directly in volume root: {volume_root}")
        
        if model_files_in_root:
            # Use volume root directly as model path
            logging.info(f"Using local model from volume root: {volume_root}")
            colpali_model = ColQwen2.from_pretrained(
                volume_root,
                torch_dtype=torch.bfloat16,
                device_map="cuda" if torch.cuda.is_available() else "cpu"
            ).eval()
            colpali_processor = ColQwen2Processor.from_pretrained(volume_root)
        else:
            # Check for subdirectory structure
            model_path = os.path.join(COLQWEN_MODELS_DIR, os.path.basename(DEFAULT_COLQWEN_NAME))
            if os.path.exists(model_path) and os.path.isdir(model_path):
                logging.info(f"Using local model from volume subdirectory: {model_path}")
                colpali_model = ColQwen2.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="cuda" if torch.cuda.is_available() else "cpu"
                ).eval()
                colpali_processor = ColQwen2Processor.from_pretrained(model_path)
            else:
                # Fall back to HuggingFace download
                logging.info(f"Model not found in volume, downloading from HuggingFace: {DEFAULT_COLQWEN_NAME}")
                colpali_model = ColQwen2.from_pretrained(
                    DEFAULT_COLQWEN_NAME,
                    torch_dtype=torch.bfloat16,
                    device_map="cuda" if torch.cuda.is_available() else "cpu"
                ).eval()
                colpali_processor = ColQwen2Processor.from_pretrained(DEFAULT_COLQWEN_NAME)
        
        logging.info(f"ColQwen2 model loaded successfully on device: {colpali_model.device}")
        return colpali_model, colpali_processor
    except Exception as e:
        logging.error(f"Error loading ColQwen2 model: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Failed to load ColQwen2 model: {str(e)}")

def load_data() -> Tuple[List, pd.DataFrame, Dict, Any, Any]:
    """Load all data needed for document retrieval with explicit error handling
    
    Returns:
        Tuple of (colpali_embeddings, df, page_images, bm25_index, tokenized_docs)
    
    Raises:
        RuntimeError: If loading fails
    """
    # Path definitions
    COLPALI_EMBEDDINGS_PATH = os.path.join(DATA_DIR, "colpali_embeddings.pkl")
    DATA_PICKLE_PATH = os.path.join(DATA_DIR, "data.pkl")
    PDF_PAGE_IMAGES_PATH = os.path.join(DATA_DIR, "pdf_page_image_paths.pkl")
    BM25_INDEX_PATH = os.path.join(DATA_DIR, "bm25_index.pkl")
    TOKENIZED_PARAGRAPHS_PATH = os.path.join(DATA_DIR, "tokenized_paragraphs.pkl")
    
    # Load data frame with metadata
    if os.path.exists(DATA_PICKLE_PATH):
        try:
            df = pd.read_pickle(DATA_PICKLE_PATH)
            logging.info(f"Loaded DataFrame with {len(df)} documents")
        except Exception as e:
            logging.error(f"Error loading DataFrame: {e}")
            raise RuntimeError(f"Failed to load document metadata: {str(e)}")
    else:
        logging.error(f"DataFrame not found at {DATA_PICKLE_PATH}")
        raise FileNotFoundError(f"Document metadata file not found at {DATA_PICKLE_PATH}")
    
    # Load image paths
    if os.path.exists(PDF_PAGE_IMAGES_PATH):
        try:
            with open(PDF_PAGE_IMAGES_PATH, "rb") as f:
                page_images = pickle.load(f)
            logging.info(f"Loaded {len(page_images)} image paths")
        except Exception as e:
            logging.error(f"Error loading image paths: {e}")
            raise RuntimeError(f"Failed to load image paths: {str(e)}")
    else:
        logging.error(f"Image paths file not found at {PDF_PAGE_IMAGES_PATH}")
        raise FileNotFoundError(f"Image paths file not found at {PDF_PAGE_IMAGES_PATH}")
    
    # Load ColPali embeddings
    if os.path.exists(COLPALI_EMBEDDINGS_PATH):
        try:
            with open(COLPALI_EMBEDDINGS_PATH, "rb") as f:
                colpali_embeddings = pickle.load(f)
            logging.info(f"Loaded {len(colpali_embeddings)} ColPali embeddings")
        except Exception as e:
            logging.error(f"Error loading ColPali embeddings: {e}")
            raise RuntimeError(f"Failed to load ColPali embeddings: {str(e)}")
    else:
        logging.error(f"ColPali embeddings not found at {COLPALI_EMBEDDINGS_PATH}")
        raise FileNotFoundError(f"ColPali embeddings not found at {COLPALI_EMBEDDINGS_PATH}")
    
    # Load BM25 index (optional)
    try:
        if os.path.exists(BM25_INDEX_PATH) and os.path.exists(TOKENIZED_PARAGRAPHS_PATH):
            with open(BM25_INDEX_PATH, "rb") as f:
                bm25_index = pickle.load(f)
            with open(TOKENIZED_PARAGRAPHS_PATH, "rb") as f:
                tokenized_docs = pickle.load(f)
            logging.info("Loaded BM25 index successfully")
        else:
            logging.warning("BM25 index not found, will proceed with vector search only")
            bm25_index = None
            tokenized_docs = None
    except Exception as e:
        logging.error(f"Error loading BM25 index: {e}")
        # Don't raise exception for optional component
        bm25_index = None
        tokenized_docs = None
        
    return colpali_embeddings, df, page_images, bm25_index, tokenized_docs