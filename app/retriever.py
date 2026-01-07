"""
FAISS-based retrieval module for semantic search.
Handles index building and similarity search.
"""

# Import faiss with error handling
try:
    import faiss
except ImportError:
    raise ImportError(
        "faiss-cpu is not installed. Please install it with: pip install faiss-cpu"
    )

import numpy as np
from typing import List, Dict, Tuple, Any
import os


class FAISSRetriever:
    """
    FAISS-based retriever for semantic search.
    """
    
    def __init__(self, embedding_dim: int = 384):
        """
        Initialize the FAISS retriever.
        
        Args:
            embedding_dim: Dimension of embeddings (default: 384 for all-MiniLM-L6-v2)
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.chunks_metadata = []
        self.is_built = False
    
    def build_index(self, embeddings: np.ndarray, chunks_metadata: List[Dict[str, Any]]):
        """
        Build FAISS index from embeddings and metadata.
        
        Args:
            embeddings: Numpy array of embeddings (shape: (n, embedding_dim))
            chunks_metadata: List of chunk metadata dictionaries
        """
        if embeddings.shape[0] == 0:
            raise ValueError("Cannot build index with empty embeddings")
        
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )
        
        if len(chunks_metadata) != embeddings.shape[0]:
            raise ValueError(
                f"Metadata count mismatch: {len(chunks_metadata)} chunks but "
                f"{embeddings.shape[0]} embeddings"
            )
        
        # Create FAISS index (L2 distance for normalized embeddings)
        # Using InnerProduct index since embeddings are normalized
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Normalize embeddings (should already be normalized, but ensure it)
        embeddings_normalized = embeddings.astype('float32')
        faiss.normalize_L2(embeddings_normalized)
        
        # Add embeddings to index
        self.index.add(embeddings_normalized)
        
        # Store metadata
        self.chunks_metadata = chunks_metadata.copy()
        self.is_built = True
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for most similar chunks.
        
        Args:
            query_embedding: Query embedding (shape: (1, embedding_dim))
            top_k: Number of top results to return
            
        Returns:
            List of result dictionaries with chunk info and similarity score
        """
        if not self.is_built or self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        if query_embedding.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {query_embedding.shape[1]}"
            )
        
        # Normalize query embedding
        query_normalized = query_embedding.astype('float32')
        faiss.normalize_L2(query_normalized)
        
        # Search
        top_k = min(top_k, len(self.chunks_metadata))
        distances, indices = self.index.search(query_normalized, top_k)
        
        # Build results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.chunks_metadata):
                chunk_meta = self.chunks_metadata[idx].copy()
                chunk_meta['similarity_score'] = float(distance)
                chunk_meta['rank'] = i + 1
                results.append(chunk_meta)
        
        return results
    
    def get_index_size(self) -> int:
        """
        Get the number of chunks in the index.
        
        Returns:
            Number of chunks
        """
        if not self.is_built or self.index is None:
            return 0
        return self.index.ntotal
    
    def is_empty(self) -> bool:
        """
        Check if index is empty.
        
        Returns:
            True if index is empty
        """
        return not self.is_built or self.index is None or self.index.ntotal == 0

