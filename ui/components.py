# ui/components.py
import os
import base64
import logging
from typing import List, Dict, Any
from PIL import Image
from fasthtml.common import *

# Import constants
from config import TEMP_UPLOAD_DIR, HEATMAP_DIR, PDF_IMAGES_DIR

def try_read_image(image_path):
    """Attempt to read an image and return it as an embedded element with error handling"""
    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as f:
                img_data = f.read()
                base64_img = base64.b64encode(img_data).decode('utf-8')
            
            # Determine image type from extension
            ext = os.path.splitext(image_path)[1].lower()
            if ext in ['.jpg', '.jpeg']:
                media_type = "image/jpeg"
            elif ext == '.png':
                media_type = "image/png"
            elif ext == '.gif':
                media_type = "image/gif"
            else:
                media_type = "image/jpeg"  # Default to JPEG
            
            # Display as embedded base64 image
            return Img(
                src=f"data:{media_type};base64,{base64_img}",
                cls="mx-auto max-h-96 w-full object-contain rounded-lg border border-zinc-700"
            )
        else:
            raise FileNotFoundError(f"Image not found at {image_path}")
    except Exception as e:
        logging.error(f"Error reading image {image_path}: {e}")
        return Div(
            P(f"Error displaying image: {str(e)}", cls="text-red-500 text-center"),
            cls="w-full h-64 bg-zinc-800 rounded-lg border border-zinc-700 flex items-center justify-center"
        )

def try_read_pdf_image(image_key, page_images):
    """Attempt to read a PDF image and return it as an embedded element with error handling"""
    try:
        if image_key in page_images:
            image_path = page_images[image_key]
            if os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    img_data = f.read()
                    base64_img = base64.b64encode(img_data).decode('utf-8')
                
                # Display as embedded base64 image
                return Img(
                    src=f"data:image/png;base64,{base64_img}",
                    cls="w-full rounded-lg border border-zinc-700 max-h-80 object-contain mx-auto"
                )
        
        # Try alternative paths
        parts = image_key.split('_')
        if len(parts) >= 2:
            filename = '_'.join(parts[:-1])
            page_num = int(parts[-1]) if parts[-1].isdigit() else 0
            potential_paths = [
                os.path.join(PDF_IMAGES_DIR, filename, f"{page_num}.png"),
                os.path.join(PDF_IMAGES_DIR, f"{filename}", f"page_{page_num}.png"),
                os.path.join(PDF_IMAGES_DIR, f"{filename}_{page_num}.png")
            ]
            
            for path in potential_paths:
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        img_data = f.read()
                        base64_img = base64.b64encode(img_data).decode('utf-8')
                    
                    return Img(
                        src=f"data:image/png;base64,{base64_img}",
                        cls="w-full rounded-lg border border-zinc-700 max-h-80 object-contain mx-auto"
                    )
        
        raise FileNotFoundError(f"PDF image not found for key {image_key}")
        
    except Exception as e:
        logging.error(f"Error embedding PDF image {image_key}: {e}")
        return Div(
            P(f"Error displaying context document: {str(e)}", cls="text-red-500 text-center"),
            cls="w-full h-64 bg-zinc-800 rounded-lg border border-zinc-700 flex items-center justify-center"
        )

def extract_classification(response_text):
    """Extract the one-word classification from the LLM response"""
    # Try to extract the classification after "Answer:" if present
    if "Answer:" in response_text:
        parts = response_text.split("Answer:")
        if len(parts) > 1:
            answer = parts[1].strip().lower()
            # Take only the first word
            classification = answer.split()[0] if answer.split() else "unknown"
            return classification
    
    # If no "Answer:" pattern, look for specific categories in the response
    categories = ["bumblebee", "honeybee", "wasp", "solitary bee", "hoverfly", "other flies", "butterfly", "moth", "other"]
    
    # Check for each category in the text
    response_lower = response_text.lower()
    for category in categories:
        if category in response_lower:
            if category == "flies":
                return "other flies"
            elif category in ["butterfly", "moth"]:
                return "butterfly & moths"
            return category
    
    # Default if no category found
    return "other insect"

def get_badge_color(classification):
    """Return the appropriate badge color for a classification"""
    colors = {
        "bumblebee": "badge-primary",      # Blue
        "honeybee": "badge-warning",       # Yellow
        "wasp": "badge-accent",            # Dark yellow/orange
        "hoverfly": "badge-success",       # Green
        "other flies": "badge-info",       # Light blue
        "butterfly & moths": "badge-secondary", # Purple
        "other insect": "badge-neutral",   # Gray
        "error": "badge-error"             # Red
    }
    
    # Normalize the classification
    classification = classification.lower()
    
    # Check for direct match
    if classification in colors:
        return colors[classification]
    
    # Default to neutral if no match
    return "badge-neutral"

# ui/components.py - Updated with fixes for interactive elements

def batch_upload_form():
    """Render form for uploading multiple images for batch insect classification"""
    
    return Form(
        Div(
            H2("Batch Insect Classification", cls="text-xl font-semibold text-white mb-4"),
            
            P("Upload multiple insect images (up to 10) for AI classification.", 
              cls="text-zinc-300 text-center mb-6"),
            
            # Image upload card - made more interactive
            Div(
                Div(
                    Label("Upload Insect Images:", cls="text-white font-medium mb-2"),
                    Input(
                        type="file",
                        name="image_files",
                        accept=".jpg,.jpeg,.png",
                        required=True,
                        multiple=True,
                        cls="file-input file-input-bordered file-input-warning w-full cursor-pointer",
                        onchange="handleMultipleFiles(this)"
                    ),
                    P(id="file-count", cls="text-sm text-zinc-400 mt-2"),
                    # Hidden container for individual file inputs
                    Div(id="file-inputs-container", cls="hidden"),
                    cls="grid place-items-center p-4"
                ),
                cls="card bg-zinc-800 border border-zinc-700 rounded-box w-full mb-4 relative z-10"  # Added z-index
            ),
            
            # Context sharing option - improved for interaction
            Div(
                Label(
                    Input(
                        type="checkbox", 
                        name="share_context", 
                        value="true", 
                        cls="checkbox checkbox-warning mr-2 cursor-pointer"
                    ),
                    "Share context across all images (faster)",
                    cls="flex items-center cursor-pointer text-white"
                ),
                P("Uses the same reference document for all insects instead of finding unique matches.",
                  cls="text-zinc-400 text-sm mt-1 ml-6"),
                cls="mb-6 relative z-10"  # Added z-index
            ),
            
            # Process button with improved interaction
            Button(
                Div(
                    "Classify Insects",
                    cls="flex items-center justify-center"
                ),
                id="batch-button",
                type="submit",
                cls="btn btn-warning w-full hover:btn-warning-focus relative z-10"  # Added hover state and z-index
            ),
            
            # JavaScript for handling multiple files - improved with debugging
            Script("""
            function handleMultipleFiles(input) {
                console.log("File input changed:", input.files.length, "files selected");
                const maxFiles = 10;
                if (input.files.length > maxFiles) {
                    alert(`Please select a maximum of ${maxFiles} files.`);
                    input.value = '';
                    document.getElementById('file-count').textContent = '';
                    return;
                }
                
                // Clear previous file inputs
                const container = document.getElementById('file-inputs-container');
                container.innerHTML = '';
                
                // Create hidden inputs for each file
                for (let i = 0; i < input.files.length; i++) {
                    const fileInput = document.createElement('input');
                    fileInput.type = 'file';
                    fileInput.name = `image_${i}`;
                    fileInput.style.display = 'none';
                    container.appendChild(fileInput);
                    
                    // Use DataTransfer to set the file
                    try {
                        const dataTransfer = new DataTransfer();
                        dataTransfer.items.add(input.files[i]);
                        fileInput.files = dataTransfer.files;
                        console.log(`Added file ${i+1}: ${input.files[i].name}`);
                    } catch (err) {
                        console.error(`Error adding file ${i+1}:`, err);
                    }
                }
                
                // Update count display
                document.getElementById('file-count').textContent = 
                    `${input.files.length} file${input.files.length !== 1 ? 's' : ''} selected`;
                    
                console.log("Files processed successfully");
            }
            """),
            
            # Additional initialization script for HTMX and events
            Script("""
            document.addEventListener('DOMContentLoaded', function() {
                console.log("DOM loaded, initializing form elements");
                
                // Make sure file inputs are clickable
                const fileInputs = document.querySelectorAll('input[type="file"]');
                fileInputs.forEach(input => {
                    input.addEventListener('click', function(e) {
                        console.log("File input clicked");
                        e.stopPropagation();
                    });
                });
                
                // Make sure checkboxes are clickable
                const checkboxes = document.querySelectorAll('input[type="checkbox"]');
                checkboxes.forEach(checkbox => {
                    checkbox.addEventListener('click', function(e) {
                        console.log("Checkbox clicked, value:", this.checked);
                        e.stopPropagation();
                    });
                });
                
                // Add submit handler with logging
                const form = document.getElementById('batch-upload-form');
                if (form) {
                    form.addEventListener('submit', function(e) {
                        console.log("Form submitted");
                        const formData = new FormData(this);
                        console.log("Files to upload:", formData.getAll('image_files').length);
                    });
                }
            });
            """),
            
            cls="bg-zinc-900 rounded-md p-6 w-full max-w-lg border border-zinc-700"
        ),
        action="/process-batch",
        method="post",
        enctype="multipart/form-data",
        id="batch-upload-form",
        hx_post="/process-batch",
        hx_target="#main-content",
        hx_indicator="#loading-indicator",
    )

def carousel_ui(batch_results):
    """Create a carousel UI to display multiple image analysis results"""
    # Create carousel items
    carousel_items = []
    carousel_indicators = []
    
    for i, result in enumerate(batch_results):
        # Extract data
        analysis_id = result.get("analysis_id", f"result_{i}")
        image_path = result.get("image_path", "")
        response = result.get("response", "No response generated")
        context_paragraphs = result.get("context_paragraphs", [])
        top_sources = result.get("top_sources", [])
        token_maps = result.get("token_maps", {})
        has_error = result.get("error", False)
        
        # Extract the one-word classification
        classification = "error" if has_error else extract_classification(response)
        badge_color = get_badge_color(classification)
        
        # Create a unique modal ID for this result
        modal_id = f"modal_{analysis_id}"
        
        # Create carousel item for the image WITH its classification elements
        carousel_items.append(
            Div(
                # Image display
                try_read_image(image_path),
                
                # Classification elements directly under each image
                Div(
                    # Badge and controls section
                    Div(
                        # Classification badge
                        Div(
                            classification,
                            cls=f"badge {badge_color} text-lg p-3 mr-4"
                        ),
                        
                        # Swap component
                        Label(
                            # Hidden checkbox controls the state
                            Input(type="checkbox", cls="cursor-pointer"),
                            # Swap-on shows when checked (thumbs down)
                            Div("👎", cls="swap-on"),
                            # Swap-off shows when unchecked (thumbs up)
                            Div("👍", cls="swap-off"),
                            cls="swap swap-flip text-3xl mx-2 cursor-pointer"
                        ),
                        
                        # View raw output modal button
                        Button(
                            "view raw output",
                            onclick=f"{modal_id}.showModal()",
                            cls="btn btn-sm btn-outline ml-2 cursor-pointer"
                        ),
                        
                        # Add dialog modal
                        Dialog(
                            Div(
                                H3("LLM Response", cls="text-lg font-bold"),
                                P(response, cls="py-4 whitespace-pre-wrap text-sm font-mono bg-black p-2 rounded overflow-auto max-h-96"),
                                Div(
                                    Form(
                                        Button("Close", cls="btn cursor-pointer"),
                                        method="dialog"
                                    ),
                                    cls="modal-action"
                                ),
                                cls="modal-box"
                            ),
                            id=modal_id,
                            cls="modal"
                        ),
                        
                        cls="flex items-center mt-4 mb-2 justify-center"
                    ),
                    
                    # Collapsible context section - only if context exists
                    (Div(
                        # Collapse title
                        Div(
                            "View Context",
                            cls="collapse-title font-semibold cursor-pointer"
                        ),
                        
                        # Collapse content
                        Div(
                            # Context document
                            (Div(
                                H4("Context Document", cls="text-lg font-semibold text-white mb-2"),
                                # Need to pass the page_images to try_read_pdf_image
                                # This will need modification in the actual code
                                Img(
                                    src=f"/image/{top_sources[0].get('image_key', '')}",
                                    cls="w-full rounded-lg border border-zinc-700 max-h-80 object-contain mx-auto"
                                ),
                                cls="mb-4"
                            ) if top_sources else ""),
                            
                            # Context paragraphs if available
                            (Div(
                                H4("Retrieved Context", cls="text-lg font-semibold text-white mb-2"),
                                P(context_paragraphs[0] if context_paragraphs else "No context available", 
                                cls="text-white text-sm bg-zinc-700 p-3 rounded-md"),
                                cls="mb-4"
                            ) if context_paragraphs else ""),
                            
                            # Token map
                            (Div(
                                H4("Token Similarity Map", cls="text-lg font-semibold text-white mb-2"),
                                (Img(
                                    src=f"/heatmap-image/{token_maps.get(top_sources[0].get('image_key', ''), [])[0]['path']}" 
                                        if top_sources and token_maps.get(top_sources[0].get('image_key', ''), []) else "",
                                    cls="w-full max-h-80 object-contain rounded-md"
                                ) if top_sources and token_maps.get(top_sources[0].get('image_key', ''), []) else 
                                Div("No token maps available", cls="text-zinc-400 text-center p-4")),
                                cls="mb-4"
                            ) if top_sources and token_maps else ""),
                            
                            cls="collapse-content text-sm"
                        ),
                        
                        tabindex="0",
                        cls="bg-zinc-800 text-white hover:bg-zinc-700 collapse rounded-md cursor-pointer",
                    ) if top_sources else ""),
                    
                    cls="w-full px-4 pb-4"
                ),
                
                id=f"item{i+1}",
                cls="carousel-item w-full flex flex-col"
            )
        )
        
        # Create indicator buttons
        carousel_indicators.append(
            A(
                str(i+1),
                href=f"#item{i+1}",
                cls=f"btn btn-xs {'' if i > 0 else 'btn-active'} cursor-pointer"
            )
        )
    
    # Create the complete carousel component
    return Div(
        H2("Insect Classification Results", cls="text-2xl font-bold text-white mb-4 text-center"),
        
        # Main carousel container with images and their classification elements
        Div(
            *carousel_items,
            cls="carousel w-full rounded-lg overflow-hidden mb-4"
        ),
        
        # Carousel indicators
        Div(
            *carousel_indicators,
            cls="flex justify-center w-full gap-2 py-2"
        ),
        
        # Process another batch button
        Div(
            Button(
                "Classify More Insects",
                hx_get="/batch-upload",
                hx_target="#main-content",
                cls="btn btn-warning w-full max-w-xs cursor-pointer"
            ),
            cls="mt-8 text-center w-full"
        ),
        
        # Enhanced JavaScript with debugging - for carousel navigation  
        Script("""
        document.addEventListener('DOMContentLoaded', function() {
            console.log("Initializing carousel UI");
            
            // Update active indicator on hash change
            const updateActiveIndicator = function() {
                const id = window.location.hash.substring(1);
                if (!id) return;
                
                console.log("Updating active indicator for:", id);
                
                // Update active indicator
                document.querySelectorAll('[href^="#item"]').forEach(indicator => {
                    if (indicator.getAttribute('href') === '#' + id) {
                        indicator.classList.add('btn-active');
                    } else {
                        indicator.classList.remove('btn-active');
                    }
                });
            };
            
            // Listen for hash changes
            window.addEventListener('hashchange', updateActiveIndicator);
            
            // Initial update - if no hash, set to first item
            if (!window.location.hash) {
                console.log("Setting initial carousel position to item1");
                window.location.hash = 'item1';
            } else {
                updateActiveIndicator();
            }
            
            // Make all modal buttons clickable
            document.querySelectorAll('[onclick*="showModal"]').forEach(btn => {
                btn.addEventListener('click', function(e) {
                    console.log("Modal button clicked:", this.innerText);
                    e.stopPropagation();
                });
            });
            
            // Initialize collapse elements
            document.querySelectorAll('.collapse').forEach(collapse => {
                collapse.addEventListener('click', function() {
                    console.log("Collapse element clicked");
                });
            });
            
            console.log("Carousel UI initialized");
        });
        """),
        
        id="batch-results",
        cls="w-full flex flex-col items-center bg-zinc-900 rounded-md p-6 fade-in"
    )
