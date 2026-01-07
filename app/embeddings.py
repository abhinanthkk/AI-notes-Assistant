"""
Embedding generation module using sentence-transformers.
Generates embeddings for text chunks and queries.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union, Dict, Any
import os


class EmbeddingModel:
    """
    Wrapper class for sentence-transformers model.
    Handles model loading and embedding generation.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence-transformers model
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence-transformers model."""
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            raise Exception(f"Failed to load embedding model {self.model_name}: {str(e)}")
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            Numpy array of embeddings (shape: (n, embedding_dim))
        """
        if self.model is None:
            self._load_model()
        
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            raise ValueError("Cannot encode empty text list")
        
        try:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            return embeddings
        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}")
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension
        """
        if self.model is None:
            self._load_model()
        
        # all-MiniLM-L6-v2 produces 384-dimensional embeddings
        # Test with a dummy string to get actual dimension
        test_embedding = self.encode("test")
        return test_embedding.shape[1]


# Global model instance (lazy loading)
_embedding_model = None


def get_embedding_model() -> EmbeddingModel:
    """
    Get or create the global embedding model instance.
    
    Returns:
        EmbeddingModel instance
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model


def generate_chunk_embeddings(chunks: List[Dict[str, Any]]) -> np.ndarray:
    """
    Generate embeddings for a list of text chunks.
    
    Args:
        chunks: List of chunk dictionaries with 'chunk_text' key
        
    Returns:
        Numpy array of embeddings
    """
    model = get_embedding_model()
    
    chunk_texts = [chunk['chunk_text'] for chunk in chunks]
    embeddings = model.encode(chunk_texts)
    
    return embeddings


def generate_query_embedding(query: str) -> np.ndarray:
    """
    Generate embedding for a query string.
    
    Args:
        query: Query text string
        
    Returns:
        Numpy array of embedding (shape: (1, embedding_dim))
    """
    model = get_embedding_model()
    embedding = model.encode(query)
    
    # Ensure 2D shape for consistency
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)
    
    return embedding

