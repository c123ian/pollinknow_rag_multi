# app_rag.py
import os
import logging
import uuid
from typing import List, Dict, Any
from PIL import Image

# Modal imports
import modal
from config import DATA_DIR, DATABASE_DIR, MISTRAL_MODELS_DIR, COLQWEN_MODELS_DIR, APP_NAME
import logging
from fasthtml.common import *
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware import Middleware
from starlette.requests import Request

# Import from our modules
from config import (
    DATA_DIR, DATABASE_DIR, MISTRAL_MODELS_DIR, COLQWEN_MODELS_DIR, 
    TEMP_UPLOAD_DIR, HEATMAP_DIR, PDF_IMAGES_DIR,
    USERNAME, APP_NAME, SKIP_SIMILARITY_MAPS, USE_SECONDARY_GPU_FOR_MAPS
)
from common_image import common_image, BEE_VOLUME, DB_VOLUME, MISTRAL_VOLUME, COLQWEN_VOLUME
from models.document_retrieval import DocumentRetriever
from models.llm_service import LLMService, format_image
from models.similarity_maps import SimilarityMapGenerator
from database.db_operations import DatabaseManager
from ui.components import (
    try_read_image, try_read_pdf_image, extract_classification,
    get_badge_color, batch_upload_form, carousel_ui
)
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create Modal app
app = modal.App(APP_NAME)

# Define volumes
try:
    mistral_volume = modal.Volume.lookup("Mistral", create_if_missing=False)
    logging.info("Successfully found Mistral volume")
except modal.exception.NotFoundError:
    raise Exception("Download Mistral models first with the appropriate script")

try:
    colqwen_volume = modal.Volume.lookup("ColQwen", create_if_missing=False)
    logging.info("Successfully found ColQwen volume")
except modal.exception.NotFoundError:
    raise Exception("Download ColQwen models first with the appropriate script")

bee_volume = modal.Volume.from_name("bee_pdf", create_if_missing=True)
db_volume = modal.Volume.lookup("db_data", create_if_missing=True)

# Define common image
image = modal.Image.debian_slim(python_version="3.10") \
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



# Global variables for data storage
colpali_model = None
colpali_processor = None
colpali_embeddings = None
df = None
page_images = {}
bm25_index = None
tokenized_docs = None

# Load models and data functions (imported from modules)
from database.data_utils import ensure_colpali_model_loaded, load_data

# Main VLLM service function
@app.function(
    image=common_image,
    gpu=modal.gpu.A100(count=1, size="80GB"),
    container_idle_timeout=10 * 60,
    timeout=24 * 60 * 60,
    allow_concurrent_inputs=20,
    volumes={
        MISTRAL_MODELS_DIR: mistral_volume,
        COLQWEN_MODELS_DIR: colqwen_volume,
        DATA_DIR: bee_volume,
        DATABASE_DIR: db_volume
    }
)
@modal.asgi_app()
def serve_vllm():
    """VLLM service endpoint for LLM processing"""
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    
    # Import vllm-specific modules
    from vllm import LLM
    from vllm.sampling_params import SamplingParams
    from models.llm_service import find_model_path

    # Create FastAPI app
    web_app = FastAPI(
        title=f"OpenAI-compatible LLM server",
        description="Multimodal LLM server for insect classification",
        version="0.1.0",
        docs_url="/docs",
    )

    # Find model path and initialize LLM
    model_path = find_model_path(MISTRAL_MODELS_DIR)
    if not model_path:
        raise Exception(f"Could not find model files in {MISTRAL_MODELS_DIR}")

    logging.info(f"Initializing LLM with model path: {model_path}")
    
    try:
        # Initialize with multimodal support
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.95,
            trust_remote_code=True,
            max_model_len=4096,
            tokenizer_mode="mistral",
            config_format="mistral",
            load_format="mistral",
            dtype="float16"
        )
        logging.info("LLM initialized successfully!")
    except Exception as init_error:
        logging.error(f"Error initializing LLM: {init_error}")
        import traceback
        traceback.print_exc()
        raise

    # Define endpoints (shortened for clarity)
    @web_app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        """Handle multimodal chat completions"""
        # Implementation omitted for brevity
        pass

    @web_app.get("/health")
    async def health_check():
        """Check if the server is running and the model is loaded"""
        return JSONResponse(content={"status": "healthy"})

    return web_app

# Main FastHTML app service
@app.function(
    image=common_image,
    gpu=modal.gpu.A100(count=1, size="80GB") if not SKIP_SIMILARITY_MAPS else None,
    container_idle_timeout=10 * 60,
    timeout=24 * 60 * 60,
    volumes={
        DATA_DIR: bee_volume,
        DATABASE_DIR: db_volume,
        COLQWEN_MODELS_DIR: colqwen_volume
    },
    secrets=[modal.Secret.from_name("my-custom-secret-3")]
)
@modal.asgi_app()
def serve_fasthtml():
    """Main FastHTML application for the web UI"""
    import nltk
    
    # Setup NLTK
    NLTK_DATA_DIR = "/tmp/nltk_data"
    os.makedirs(NLTK_DATA_DIR, exist_ok=True)
    nltk.data.path.append(NLTK_DATA_DIR)
    nltk.download("punkt", download_dir=NLTK_DATA_DIR)
    
    # Create necessary directories
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
    os.makedirs(HEATMAP_DIR, exist_ok=True)
    
    # Initialize database
    db_manager = DatabaseManager()
    
    # Load models and data
    global colpali_model, colpali_processor, colpali_embeddings, df, page_images, bm25_index, tokenized_docs
    ensure_colpali_model_loaded()
    colpali_embeddings, df, page_images, bm25_index, tokenized_docs = load_data()
    
    # Initialize the similarity map generator if enabled
    if not SKIP_SIMILARITY_MAPS:
        map_generator = SimilarityMapGenerator(
            model=colpali_model,
            processor=colpali_processor,
            use_secondary_gpu=USE_SECONDARY_GPU_FOR_MAPS
        )
    
    # Process with Mistral - now just orchestrates our components
    async def process_with_mistral(image, query, context_text="", analysis_id=""):
        """Process an image with Mistral using our modular components"""
        logging.info(f"Processing image {analysis_id} with Mistral using query: {query}")
        
        # Validate inputs
        if image is None:
            raise ValueError("No image provided for processing")
        
        try:
            # Create document retriever 
            document_retriever = DocumentRetriever(
                colpali_model, colpali_processor, colpali_embeddings, df, page_images, bm25_index, tokenized_docs
            )
            
            # Retrieve relevant documents
            retrieved_paragraphs, top_sources = await document_retriever.retrieve_documents(query)
            
            # Initialize LLM service
            llm_service = LLMService()
            
            # Use context from retrieved documents
            if not context_text and retrieved_paragraphs:
                context_text = "\n\n".join(retrieved_paragraphs)
                
            context_source = top_sources[0] if top_sources else None
            
            # Process the image with explicit multimodal prompt
            response = await llm_service.process_image_classification(
                image=image,
                query=query,
                context_text=context_text,
                context_source=context_source
            )
            
            return response, retrieved_paragraphs, top_sources
            
        except Exception as e:
            logging.error(f"Exception in process_with_mistral: {str(e)}")
            raise RuntimeError(f"Failed to process image: {str(e)}")
    
    # Initialize FastHTML app
    fasthtml_app, rt = fast_app(
        # Configuration omitted for brevity
    )
    
    # Define routes (process_batch, etc.) - implementation omitted for brevity
    @rt("/process-batch", methods=["POST"])
    async def process_batch(request: Request):
        """Process a batch of uploaded images for insect classification"""
        # Implementation omitted for brevity
        pass
    
    @rt("/")
    def get(session):
        """Main landing page"""
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        
        # Return UI components
        return (
            Title("Insect Classification"),
            Main(
                # UI components omitted for brevity
            )
        )
    
    return fasthtml_app

if __name__ == "__main__":
    # Entry point
    serve_vllm()
    serve_fasthtml()