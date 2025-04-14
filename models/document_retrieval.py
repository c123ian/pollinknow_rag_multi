# models/document_retrieval.py
import torch
import logging
import numpy as np
from typing import List, Dict, Tuple, Any
from nltk.tokenize import word_tokenize

class DocumentRetriever:
    def __init__(self, colpali_model, colpali_processor, colpali_embeddings, 
                 dataframe, page_images, bm25_index=None, tokenized_docs=None):
        """Initialize document retriever with models and data
        
        Args:
            colpali_model: Loaded ColPali/ColQwen model
            colpali_processor: Model processor
            colpali_embeddings: Pre-computed embeddings
            dataframe: DataFrame with document metadata
            page_images: Dictionary mapping image keys to file paths
            bm25_index: Optional BM25 index for text search
            tokenized_docs: Optional tokenized documents for BM25
        """
        self.model = colpali_model
        self.processor = colpali_processor
        self.embeddings = colpali_embeddings
        self.df = dataframe
        self.page_images = page_images
        self.bm25_index = bm25_index
        self.tokenized_docs = tokenized_docs
        
    async def retrieve_documents(self, query: str, top_k: int = 5) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Retrieve most relevant documents for a query
        
        Args:
            query: The search query
            top_k: Number of top results to retrieve
            
        Returns:
            Tuple of (retrieved_paragraphs, top_sources_metadata)
            
        Raises:
            ValueError: If no documents or embeddings are available
            RuntimeError: If retrieval process fails
        """
        if self.embeddings is None or self.df is None or len(self.df) == 0:
            raise ValueError("No documents or embeddings available for retrieval")
            
        retrieved_paragraphs = []
        top_sources_data = []
        
        try:
            # Process query with ColPali
            processed_query = self.processor.process_queries([query]).to(self.model.device)
            with torch.no_grad():
                query_embeddings = self.model(**processed_query)
            
            # Calculate similarities with all pages
            similarities = []
            for idx, page_emb in enumerate(self.embeddings):
                # Convert page embedding to tensor with matching dtype
                page_tensor = torch.tensor(page_emb, device=self.model.device, dtype=query_embeddings.dtype)
                
                # Score using ColPali's scoring method
                score = float(self.processor.score_multi_vector(
                    query_embeddings,
                    page_tensor.unsqueeze(0)  # Add batch dimension
                )[0])
                
                similarities.append((idx, score))
            
            # Sort by similarity score
            similarities.sort(key=lambda x: x[1], reverse=True)
            vector_top_indices = [idx for idx, _ in similarities[:top_k]]
            
            # Try BM25 keyword search if available
            keyword_top_indices = []
            bm25_scores = None
            if self.bm25_index is not None and self.tokenized_docs is not None:
                try:
                    # Tokenize query and get BM25 scores
                    tokenized_query = word_tokenize(query.lower())
                    bm25_scores = self.bm25_index.get_scores(tokenized_query)
                    keyword_top_indices = np.argsort(bm25_scores)[-top_k:][::-1].tolist()
                except Exception as e:
                    logging.error(f"Error in BM25 scoring: {e}")
                    # Don't fail the whole retrieval process, just don't use BM25
                    keyword_top_indices = []
            
            # Combine results (hybrid retrieval)
            all_indices = list(set(vector_top_indices + keyword_top_indices))
            
            # Check if we have any results
            if not all_indices:
                raise ValueError("No relevant documents found for query")
            
            # Get data for reranking
            docs_for_reranking = []
            doc_indices = []
            
            for idx in all_indices:
                if idx < len(self.df):
                    # Get document info
                    filename = self.df.iloc[idx]['filename']
                    page_num = self.df.iloc[idx]['page']
                    image_key = self.df.iloc[idx]['image_key']
                    text = self.df.iloc[idx]['text']
                    
                    # Get vector score
                    vector_score = 0.0
                    for v_idx, score in similarities:
                        if v_idx == idx:
                            vector_score = score
                            break
                    
                    # Get keyword score (if available)
                    keyword_score = 0.0
                    if bm25_scores is not None and len(bm25_scores) > idx:
                        keyword_score = float(bm25_scores[idx] / max(bm25_scores) if max(bm25_scores) > 0 else 0)
                    
                    # Combine scores (weighted)
                    alpha = 0.7  # Weight for vector search
                    combined_score = alpha * vector_score + (1 - alpha) * keyword_score
                    
                    # Store for reranking
                    docs_for_reranking.append(text)
                    doc_indices.append(idx)
                    
                    # Add to results
                    retrieved_paragraphs.append(text)
                    top_sources_data.append({
                        'filename': filename,
                        'page': page_num,
                        'score': combined_score,
                        'vector_score': vector_score,
                        'keyword_score': keyword_score,
                        'image_key': image_key,
                        'idx': idx
                    })
            
            # Make sure we have documents to rerank
            if not docs_for_reranking:
                raise ValueError("Document retrieval failed: no documents match the query")
            
            # Rerank results if we have documents
            try:
                # Use a cross-encoder reranker
                from rerankers import Reranker
                ranker = Reranker('cross-encoder/ms-marco-MiniLM-L-6-v2', model_type="cross-encoder", verbose=0)
                ranked_results = ranker.rank(query=query, docs=docs_for_reranking)
                top_ranked = ranked_results.top_k(min(3, len(docs_for_reranking)))
                
                # Get the final top documents after reranking
                final_retrieved_paragraphs = []
                final_top_sources = []
                
                for ranked_doc in top_ranked:
                    ranked_idx = docs_for_reranking.index(ranked_doc.text)
                    doc_idx = doc_indices[ranked_idx]
                    source_info = next((s for s in top_sources_data if s['idx'] == doc_idx), None)
                    if source_info:
                        source_info['reranker_score'] = ranked_doc.score
                        final_top_sources.append(source_info)
                        final_retrieved_paragraphs.append(ranked_doc.text)
                
                if not final_retrieved_paragraphs:
                    raise ValueError("Reranking produced no results")
                    
                return final_retrieved_paragraphs, final_top_sources
                
            except Exception as e:
                logging.warning(f"Reranking failed: {e}. Falling back to score-based ranking.")
                # If reranking fails, sort by combined score
                sorted_indices = sorted(range(len(top_sources_data)), 
                                       key=lambda i: top_sources_data[i]['score'], 
                                       reverse=True)
                sorted_paragraphs = [retrieved_paragraphs[i] for i in sorted_indices[:3]]
                sorted_sources = [top_sources_data[i] for i in sorted_indices[:3]]
                
                return sorted_paragraphs, sorted_sources
            
        except Exception as e:
            logging.error(f"Document retrieval failed: {str(e)}")
            raise RuntimeError(f"Document retrieval failed: {str(e)}")