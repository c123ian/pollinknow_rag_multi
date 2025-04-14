# models/similarity_maps.py
import os
import torch
import logging
import matplotlib.pyplot as plt
from PIL import Image
from colpali_engine.interpretability import get_similarity_maps_from_embeddings, plot_similarity_map

from config import HEATMAP_DIR

class SimilarityMapGenerator:
    def __init__(self, model, processor, use_secondary_gpu=False):
        """Initialize similarity map generator with model, processor and GPU settings
        
        Args:
            model: ColPali/ColQwen model
            processor: Corresponding processor
            use_secondary_gpu: Whether to use a secondary GPU (if available)
        """
        self.model = model
        self.processor = processor
        
        # Determine device based on availability and settings
        if use_secondary_gpu and torch.cuda.device_count() > 1:
            self.device = "cuda:1"
            logging.info(f"Using secondary GPU (cuda:1) for similarity maps")
        else:
            self.device = self.model.device
            logging.info(f"Using primary device {self.device} for similarity maps")
            
        # Create output directory
        os.makedirs(HEATMAP_DIR, exist_ok=True)
        
    async def generate_maps(self, query, image_key, image_path):
        """Generate token similarity maps for a document based on query
        
        Args:
            query: The search query
            image_key: Identifier key for the document image
            image_path: Path to the document image file
            
        Returns:
            List of dictionaries with token map information or raises Exception
        """
        # Validate inputs
        if not query or not image_key or not image_path:
            raise ValueError("Missing required parameters for similarity map generation")
            
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Document image not found at: {image_path}")
            
        try:
            # Load the image
            image = Image.open(image_path)
            
            # Process query and image with ColPali
            processed_query = self.processor.process_queries([query]).to(self.device)
            batch_images = self.processor.process_images([image]).to(self.device)
            
            # Forward passes to get embeddings
            with torch.no_grad():
                query_embeddings = self.model(**processed_query)
                image_embeddings = self.model(**batch_images)
            
            # Get the number of image patches
            try:
                # Try with spatial_merge_size parameter if available
                if hasattr(self.model, 'spatial_merge_size'):
                    n_patches = self.processor.get_n_patches(
                        image_size=image.size,
                        patch_size=self.model.patch_size,
                        spatial_merge_size=self.model.spatial_merge_size
                    )
                else:
                    # Simpler version without spatial_merge_size
                    n_patches = self.processor.get_n_patches(
                        image_size=image.size,
                        patch_size=self.model.patch_size
                    )
            except Exception as e:
                raise RuntimeError(f"Failed to calculate patch size: {str(e)}")
            
            # Get image mask
            image_mask = self.processor.get_image_mask(batch_images)
            
            # Generate similarity maps
            batched_similarity_maps = get_similarity_maps_from_embeddings(
                image_embeddings=image_embeddings,
                query_embeddings=query_embeddings,
                n_patches=n_patches,
                image_mask=image_mask
            )
            
            # Get the similarity map for this image
            similarity_maps = batched_similarity_maps[0]  # (query_length, n_patches_x, n_patches_y)
            
            # Get tokens for the query
            query_tokens = self.processor.tokenizer.tokenize(query)
            
            # Filter to meaningful tokens
            token_sims = []
            stopwords = set(["<bos>", "<eos>", "<pad>", "a", "an", "the", "in", "on", "at", "of", "for", "with", "by", "to", "from"])
            
            for token_idx, token in enumerate(query_tokens):
                if token_idx >= similarity_maps.shape[0]:
                    continue
                    
                # Skip stopwords and short tokens
                if token in stopwords or len(token) <= 1:
                    continue
                    
                token_clean = token.replace("Ġ", "").replace("▁", "")
                if token_clean and len(token_clean) > 1:
                    max_sim = similarity_maps[token_idx].max().item()
                    token_sims.append((token_idx, token, max_sim))
            
            # Sort by similarity score and take top tokens
            token_sims.sort(key=lambda x: x[2], reverse=True)
            top_tokens = token_sims[:6]  # Get top 6 tokens
            
            # Generate and save heatmaps
            image_heatmaps = []
            for token_idx, token, score in top_tokens:
                # Skip if score is very low
                if score < 0.1:
                    continue
                    
                # Generate heatmap
                fig, ax = plot_similarity_map(
                    image=image,
                    similarity_map=similarity_maps[token_idx],
                    figsize=(8, 8),
                    show_colorbar=False,
                )
                
                # Clean token for display
                token_display = token.replace("Ġ", "").replace("▁", "")
                ax.set_title(f"Token: '{token_display}', Score: {score:.2f}", fontsize=12)
                
                # Save heatmap
                heatmap_filename = f"{image_key}_token_{token_idx}.png"
                heatmap_path = os.path.join(HEATMAP_DIR, heatmap_filename)
                fig.savefig(heatmap_path, bbox_inches='tight', dpi=150)
                plt.close(fig)
                
                # Add to list
                image_heatmaps.append({
                    "token": token_display,
                    "score": score,
                    "path": heatmap_filename,
                    "token_idx": token_idx
                })
            
            logging.info(f"Generated {len(image_heatmaps)} token heatmaps")
            
            # If no heatmaps were generated, raise exception
            if not image_heatmaps:
                raise ValueError("No significant token similarities found in document")
                
            return image_heatmaps
            
        except Exception as e:
            logging.error(f"Error generating similarity maps: {str(e)}")
            # Re-raise with clear error message
            raise RuntimeError(f"Failed to generate similarity maps: {str(e)}")