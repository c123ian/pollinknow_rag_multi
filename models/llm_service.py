# models/llm_service.py
import os
import logging
import uuid
import datetime
import base64
import io
from typing import Dict, Any, List
from PIL import Image

from config import MISTRAL_MODELS_DIR, DEFAULT_MISTRAL_NAME

def format_image(image):
    """Convert PIL Image to data URI for multimodal API"""
    buffered = io.BytesIO()
    # Convert to RGB if it has alpha channel
    if image.mode == "RGBA":
        image = image.convert("RGB")
    image.save(buffered, format="JPEG", quality=90)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    image_data_uri = f'data:image/jpeg;base64,{img_base64}'
    return image_data_uri

def find_model_path(base_dir):
    """Find model directory with config.json"""
    for root, _, files in os.walk(base_dir):
        if "config.json" in files:
            return root
    return None

class LLMService:
    def __init__(self):
        """Initialize the LLM service to load VLLM"""
        from vllm import LLM
        
        # Find model path
        model_path = find_model_path(MISTRAL_MODELS_DIR)
        if not model_path:
            raise RuntimeError(f"Could not find model files in {MISTRAL_MODELS_DIR}")
            
        logging.info(f"Initializing LLM with model path: {model_path}")
        
        try:
            # Initialize with multimodal support
            self.llm = LLM(
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
            raise RuntimeError(f"Failed to initialize LLM: {str(init_error)}")
            
    async def process_image_classification(self, image: Image.Image, query: str, context_text: str = "", context_source: Dict = None):
        """Process an image classification with context
        
        Args:
            image: The PIL image to classify
            query: The classification query
            context_text: Optional context information as text
            context_source: Optional metadata about the context source
            
        Returns:
            The model's response
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If processing fails
        """
        from vllm.sampling_params import SamplingParams
        
        if image is None:
            raise ValueError("No image provided for classification")
            
        # Convert image to data URI for model input
        image_uri = format_image(image)
        
        # Create system prompt for insect classification
        system_prompt = (
            "You are an expert biologist. Classify the insect in the provided image into "
            "ONE of these categories (bumblebee, honeybee, wasp, solitary bee, hoverfly, other flies, "
            "butterfly & moths, other insect). Focus primarily on the visual characteristics "
            "in the insect image."
        )
        
        # Create task description with context if available
        task_text = "Classify this insect shown in the image."
        if context_text:
            task_text += " Additional reference information: " + context_text
            # Add source reference if available
            if context_source:
                task_text += f"\n\nThis information comes from {context_source.get('filename', 'unknown document')}, page {context_source.get('page', 'unknown')}."
        
        # Create message content structure
        message_content = [
            {"type": "text", "text": task_text},
            {"type": "image_url", "image_url": {"url": image_uri}}
        ]
        
        # Create messages array
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message_content}
        ]
        
        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=2000,
            stop=["User:", "System:"]
        )
        
        try:
            # Process with the LLM directly
            outputs = self.llm.chat(messages=messages, sampling_params=sampling_params)
            
            if outputs and len(outputs) > 0 and len(outputs[0].outputs) > 0:
                response_text = outputs[0].outputs[0].text
                return response_text.strip()
            else:
                raise RuntimeError("LLM produced no output")
                
        except Exception as e:
            logging.error(f"Error processing with LLM: {str(e)}")
            raise RuntimeError(f"Image processing failed: {str(e)}")