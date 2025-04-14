# app_rag.py
import os
import logging
import uuid
import asyncio
from typing import List, Dict, Any
from PIL import Image

# Modal imports
import modal
from fasthtml.common import *
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

# Import from our modules
from config import (
    DATA_DIR, DATABASE_DIR, MISTRAL_MODELS_DIR, COLQWEN_MODELS_DIR, 
    TEMP_UPLOAD_DIR, HEATMAP_DIR, PDF_IMAGES_DIR,
    USERNAME, APP_NAME, SKIP_SIMILARITY_MAPS, USE_SECONDARY_GPU_FOR_MAPS
)
from common_image import common_image, BEE_VOLUME, DB_VOLUME, MISTRAL_VOLUME, COLQWEN_VOLUME
from models.document_retrieval import DocumentRetriever
from models.llm_service import LLMService
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
bee_volume = modal.Volume.from_name("bee_pdf", create_if_missing=True)
db_volume = modal.Volume.from_name("db_data", create_if_missing=True)

# Global variables for data storage
colpali_model = None
colpali_processor = None
colpali_embeddings = None
df = None
page_images = {}
bm25_index = None
tokenized_docs = None

# Load models and data functions
from database.data_utils import ensure_colpali_model_loaded, load_data

# Main FastHTML app service
@app.function(
    image=common_image,
    gpu=modal.gpu.A100(count=1, size="80GB") if not SKIP_SIMILARITY_MAPS else None,
    container_idle_timeout=10 * 60,
    timeout=24 * 60 * 60,
    volumes={
        DATA_DIR: bee_volume,
        DATABASE_DIR: db_volume,
        COLQWEN_MODELS_DIR: COLQWEN_VOLUME,
        MISTRAL_MODELS_DIR: MISTRAL_VOLUME
    },
    secrets=[modal.Secret.from_name("my-custom-secret-3")]
)
@modal.asgi_app()
def serve_fasthtml():
    """Main FastHTML application for the web UI"""
    import nltk
    import torch
    import matplotlib.pyplot as plt
    from colpali_engine.interpretability import get_similarity_maps_from_embeddings, plot_similarity_map
    
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
    
    # Explicitly load models and data at startup
    try:
        colpali_model, colpali_processor = ensure_colpali_model_loaded()
        colpali_embeddings, df, page_images, bm25_index, tokenized_docs = load_data()
        logging.info("Successfully loaded models and data")
    except Exception as e:
        logging.error(f"Error loading models and data: {e}")
        import traceback
        traceback.print_exc()
    
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
        hdrs=(
            # Explicit HTMX loading and other scripts
            Script(src="https://unpkg.com/htmx.org@1.9.6"),
            Script(src="https://cdn.tailwindcss.com"),
            Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/daisyui@4.11.1/dist/full.min.css"),
            # Add custom CSS for better interactive elements
            Style("""
                button, .btn, a[href] {
                    cursor: pointer !important;
                }
                
                button:hover, .btn:hover, a[href]:hover {
                    opacity: 0.9;
                }
                
                .file-input {
                    cursor: pointer !important;
                }
                
                /* Ensure nothing blocks interaction */
                #main-content, #analysis-results {
                    position: relative;
                    z-index: 1;
                }
                
                /* Custom styles for token maps */
                .token-tab.active {
                    background-color: #4CAF50;
                    color: white;
                }
                
                /* Custom animations */
                .fade-in {
                    animation: fadeIn 0.5s;
                }
                
                @keyframes fadeIn {
                    from { opacity: 0; }
                    to { opacity: 1; }
                }
                
                /* Only show loading indicator when active */
                #loading-indicator {
                    display: none;
                    pointer-events: none;  /* Allows clicking through when visible */
                }
                .htmx-request #loading-indicator {
                    display: flex !important;
                }
                
                /* Custom styling for token map display */
                .token-map {
                    max-height: 350px;
                    object-fit: contain;
                }
                
                .context-container {
                    background-color: #2a2a2a;
                    border: 1px solid #3a3a3a;
                    border-radius: 8px;
                    padding: 10px;
                    margin-bottom: 12px;
                }
                
                .context-header {
                    color: #aaa;
                    font-size: 14px;
                    margin-bottom: 8px;
                }
                
                /* Carousel styles */
                .carousel-item {
                    scroll-snap-align: start;
                }
                
                .carousel {
                    scroll-behavior: smooth;
                    scroll-snap-type: x mandatory;
                }
            """),
        ),
        middleware=[
            Middleware(
                SessionMiddleware,
                secret_key=os.environ.get('YOUR_KEY', 'default-secret-key'),
                session_cookie="secure_session",
                max_age=86400,
                same_site="strict",
                https_only=True
            )
        ]
    )
    
    # Function to generate similarity maps
    async def generate_similarity_maps(query, image_key):
        """Generate token similarity maps for a document"""
        if not image_key or image_key not in page_images:
            logging.error(f"Invalid image key: {image_key}")
            return []
            
        if SKIP_SIMILARITY_MAPS:
            logging.info(f"Similarity maps generation is disabled")
            return []
            
        try:
            # Use the similarity map generator
            return await map_generator.generate_maps(query, image_key, page_images[image_key])
        except Exception as e:
            logging.error(f"Error generating similarity maps: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    # Main process_batch function that will be called by the form
    @rt("/process-batch", methods=["POST"])
    async def process_batch(request: Request):
        """Process a batch of uploaded images for insect classification"""
        logging.info("Starting batch processing of insect images")
        
        # Parse the form data
        form = await request.form()
        
        # Extract all uploaded images
        image_files = []
        for key in form.keys():
            if key.startswith('image_'):
                if form.get(key):
                    image_files.append(form.get(key))
        
        # If no image_X fields, try the multiple file field
        if not image_files and form.get("image_files"):
            image_files = form.getlist("image_files")
        
        # Limit to 10 images
        image_files = image_files[:10]
        
        if not image_files:
            return Div("No insect images uploaded", cls="text-red-500 text-center p-4")
        
        logging.info(f"Processing batch of {len(image_files)} insect images")
        
        # Set the query for insect classification
        query = "Classify as bumblebee, honeybee, wasp, solitary bee, hoverfly or other"
        share_context = form.get("share_context", "false") == "true"
        logging.info(f"Using query: {query}, Share context: {share_context}")
        
        # For shared context, retrieve documents once
        shared_context = None
        shared_top_sources = None
        
        if share_context:
            try:
                # Create document retriever
                document_retriever = DocumentRetriever(
                    colpali_model, colpali_processor, colpali_embeddings, 
                    df, page_images, bm25_index, tokenized_docs
                )
                
                # Retrieve context once for all images
                shared_context, shared_top_sources = await document_retriever.retrieve_documents(query)
                logging.info(f"Retrieved {len(shared_context) if shared_context else 0} shared context paragraphs")
            except Exception as e:
                logging.error(f"Error retrieving shared context: {e}")
                shared_context, shared_top_sources = [], []
        
        # Create directory for temporary uploads if it doesn't exist
        os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
        
        # Process each image
        batch_results = []
        token_maps_by_image = {}
        
        for i, image_file in enumerate(image_files):
            if not image_file:
                continue
                
            try:
                # Generate a unique ID for this analysis
                analysis_id = str(uuid.uuid4())
                
                # Get filename or create default
                filename = getattr(image_file, 'filename', f"image_{i}.jpg")
                # Sanitize filename
                safe_filename = "".join(c for c in filename if c.isalnum() or c in "._-")
                image_path = os.path.join(TEMP_UPLOAD_DIR, f"{analysis_id}_{safe_filename}")
                
                logging.info(f"Processing insect image {i+1}/{len(image_files)}: {filename} → {image_path}")
                
                # Save the uploaded image
                content = await image_file.read()
                with open(image_path, "wb") as f:
                    f.write(content)
                
                # Verify the file was saved
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Failed to save image to {image_path}")
                    
                # Open the image for processing
                image = Image.open(image_path)
                
                # Get context - either shared or unique per image
                retrieved_paragraphs = []
                top_sources = []
                
                if not share_context:
                    # Create document retriever for unique context
                    document_retriever = DocumentRetriever(
                        colpali_model, colpali_processor, colpali_embeddings,
                        df, page_images, bm25_index, tokenized_docs
                    )
                    
                    # Retrieve unique context for this image
                    retrieved_paragraphs, top_sources = await document_retriever.retrieve_documents(query)
                    logging.info(f"Retrieved unique context for image {i+1}")
                else:
                    # Use the shared context
                    retrieved_paragraphs, top_sources = shared_context, shared_top_sources
                    logging.info(f"Using shared context for image {i+1}")
                
                # Generate context text
                context_text = "\n\n".join(retrieved_paragraphs) if retrieved_paragraphs else ""
                
                # Generate token maps for top document
                image_token_maps = {}
                if top_sources:
                    top_source = top_sources[0]
                    image_key = top_source.get('image_key')
                    
                    # Only generate token maps if we haven't already for this context
                    if image_key and image_key not in token_maps_by_image:
                        logging.info(f"Generating token maps for context document: {image_key}")
                        image_heatmaps = await generate_similarity_maps(query, image_key)
                        if image_heatmaps:
                            token_maps_by_image[image_key] = image_heatmaps
                            image_token_maps[image_key] = image_heatmaps
                    elif image_key:
                        # Reuse existing token maps
                        image_token_maps[image_key] = token_maps_by_image[image_key]
                
                # Process with Mistral using multimodal capabilities
                logging.info(f"Processing insect image {i+1} with Mistral")
                
                try:
                    response_text, _, _ = await process_with_mistral(
                        image, query, context_text, analysis_id
                    )
                    logging.info(f"Processed image {i+1} successfully")
                except Exception as llm_error:
                    logging.error(f"Error processing with LLM: {llm_error}")
                    response_text = f"Error processing image: {str(llm_error)}"
                
                # Save to database
                try:
                    await db_manager.save_analysis(
                        analysis_id=analysis_id,
                        image_path=image_path,
                        analysis_type="insect_classification",
                        query=query,
                        response=response_text,
                        top_sources=top_sources
                    )
                    logging.info(f"Saved analysis {analysis_id} to database")
                except Exception as db_error:
                    logging.error(f"Database error for insect image {i+1}: {db_error}")
                
                # Add to batch results
                batch_results.append({
                    "analysis_id": analysis_id,
                    "image_path": image_path,
                    "response": response_text,
                    "context_paragraphs": retrieved_paragraphs,
                    "top_sources": top_sources,
                    "token_maps": image_token_maps
                })
                
            except Exception as e:
                logging.error(f"Error processing insect image {i+1}: {e}")
                import traceback
                traceback.print_exc()
                
                # Add error result
                batch_results.append({
                    "error": True,
                    "message": f"Error: {str(e)}",
                    "analysis_id": f"error_{i}",
                    "image_path": ""
                })
        
        # Ensure volume is committed
        try:
            bee_volume.commit()
        except Exception as e:
            logging.error(f"Error committing volume: {e}")
        
        # Explicit log statement to show when rendering is about to start
        logging.info(f"Rendering carousel UI with {len(batch_results)} results")
        
        # Return carousel UI with all results
        return carousel_ui(batch_results)
    
    # Serve heatmap images
    @fasthtml_app.get("/heatmap-image/{filename}")
    async def get_heatmap_image(filename: str):
        """Serve heatmap images"""
        heatmap_path = os.path.join(HEATMAP_DIR, filename)
        logging.info(f"Looking for heatmap image: {heatmap_path}")
        
        if os.path.exists(heatmap_path):
            logging.info(f"Found heatmap at: {heatmap_path}")
            try:
                # Open and read the file directly
                with open(heatmap_path, "rb") as f:
                    content = f.read()
                    
                # Return as binary response
                return Response(
                    content=content,
                    media_type="image/png"
                )
            except Exception as e:
                logging.error(f"Error reading heatmap file: {e}")
                return Response(
                    content=f"Error reading heatmap: {str(e)}",
                    media_type="text/plain",
                    status_code=500
                )
        else:
            logging.error(f"Heatmap image not found: {heatmap_path}")
            return Response(
                content=f"Heatmap not found: {filename}",
                media_type="text/plain",
                status_code=404
            )
    
    # Serve images
    @fasthtml_app.get("/image/{image_key}")
    async def get_image(image_key: str):
        """Serve document images"""
        if image_key in page_images:
            image_path = page_images[image_key]
            if os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    content = f.read()
                return Response(
                    content=content,
                    media_type="image/png"
                )
        
        # Try alternative paths
        parts = image_key.split('_')
        if len(parts) >= 2:
            filename = '_'.join(parts[:-1])
            page_num = parts[-1]
            potential_paths = [
                os.path.join(PDF_IMAGES_DIR, filename, f"{page_num}.png"),
                os.path.join(PDF_IMAGES_DIR, f"{filename}", f"page_{page_num}.png"),
                os.path.join(PDF_IMAGES_DIR, f"{filename}_{page_num}.png")
            ]
            
            for path in potential_paths:
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        content = f.read()
                    return Response(
                        content=content,
                        media_type="image/png"
                    )
        
        return Response(
            content=f"Image not found for key: {image_key}",
            media_type="text/plain",
            status_code=404
        )
    
    # Add batch upload route
    @rt("/batch-upload", methods=["GET"])
    def get_batch_upload(session):
        """Show the batch upload form"""
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
            
        logging.info(f"Showing batch upload form for session: {session['session_id']}")
        return batch_upload_form()

    # Main route function
    @rt("/")
    def get(session):
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        
        logging.info(f"New session: {session['session_id']} - showing batch upload form")
        
        return (
            Title("Insect Classification"),
            Main(
                # Loading indicator with better visibility
                Div(
                    Div(cls="loading loading-spinner loading-lg text-warning"),
                    Div("Processing your insect images...", cls="text-white mt-4 text-lg"),
                    id="loading-indicator",
                    cls="htmx-indicator fixed top-0 left-0 w-full h-full bg-black bg-opacity-80 flex flex-col items-center justify-center z-50"
                ),
                
                # Page header 
                H1("Insect Classifier", cls="text-3xl font-bold mb-4 text-white text-center"),
                
                # Header info
                Div(
                    P("Upload insect images for instant AI classification", 
                      cls="text-white text-center mb-6"),
                    cls="w-full max-w-2xl"
                ),
                
                # Main content area - DIRECTLY SHOW BATCH FORM
                Div(
                    batch_upload_form(),
                    id="main-content",
                    cls="w-full max-w-2xl"
                ),
                
                # Results area - will be populated by process-batch
                Div(id="analysis-results", cls="w-full max-w-5xl mt-8"),
                
                cls="flex flex-col items-center min-h-screen bg-black p-4",
            )
        )

if __name__ == "__main__":
    # Entry point
    serve_fasthtml()
