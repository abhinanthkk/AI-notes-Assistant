"""
Storage module for persisting embeddings, FAISS indexes, and metadata.
Handles saving and loading of all persistent data.
"""

import os
import json
from typing import List, Dict, Optional, Tuple
from app.retriever import FAISSRetriever


class StorageManager:
    """
    Manages persistent storage of PDFs, embeddings, indexes, and metadata.
    """
    
    def __init__(self, base_dir: str = "storage"):
        """
        Initialize storage manager.
        
        Args:
            base_dir: Base directory for storage
        """
        self.base_dir = base_dir
        self.pdfs_dir = os.path.join(base_dir, "pdfs")
        self.indexes_dir = os.path.join(base_dir, "indexes")
        self.embeddings_dir = os.path.join(base_dir, "embeddings")
        
        # Create directories if they don't exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create storage directories if they don't exist."""
        os.makedirs(self.pdfs_dir, exist_ok=True)
        os.makedirs(self.indexes_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)
    
    def save_pdf(self, pdf_path: str, pdf_name: str) -> str:
        """
        Save uploaded PDF to storage.
        
        Args:
            pdf_path: Source path of PDF
            pdf_name: Name to save PDF as
            
        Returns:
            Path where PDF was saved
        """
        import shutil
        
        dest_path = os.path.join(self.pdfs_dir, pdf_name)
        shutil.copy2(pdf_path, dest_path)
        return dest_path
    
    def save_index_data(
        self,
        retriever: FAISSRetriever,
        chunks_metadata: List[Dict[str, any]],
        index_name: str = "main_index"
    ):
        """
        Save FAISS index and metadata to disk.
        
        Args:
            retriever: FAISSRetriever instance with built index
            chunks_metadata: List of chunk metadata
            index_name: Name for the index file
        """
        if not retriever.is_built:
            raise ValueError("Cannot save unbuilt index")
        
        # Save FAISS index
        index_path = os.path.join(self.indexes_dir, f"{index_name}.index")
        faiss.write_index(retriever.index, index_path)
        
        # Save metadata
        metadata_path = os.path.join(self.indexes_dir, f"{index_name}_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_metadata, f, indent=2, ensure_ascii=False)
        
        # Save index info
        info_path = os.path.join(self.indexes_dir, f"{index_name}_info.json")
        info = {
            'embedding_dim': retriever.embedding_dim,
            'num_chunks': retriever.get_index_size(),
            'index_name': index_name
        }
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2)
    
    def load_index_data(
        self,
        index_name: str = "main_index"
    ) -> Optional[Tuple[FAISSRetriever, List[Dict[str, any]]]]:
        """
        Load FAISS index and metadata from disk.
        
        Args:
            index_name: Name of the index file
            
        Returns:
            Tuple of (FAISSRetriever, chunks_metadata) or None if not found
        """
        index_path = os.path.join(self.indexes_dir, f"{index_name}.index")
        metadata_path = os.path.join(self.indexes_dir, f"{index_name}_metadata.json")
        info_path = os.path.join(self.indexes_dir, f"{index_name}_info.json")
        
        # Check if files exist
        if not all(os.path.exists(p) for p in [index_path, metadata_path, info_path]):
            return None
        
        try:
            # Load index info
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            
            embedding_dim = info.get('embedding_dim', 384)
            
            # Create retriever and load index
            retriever = FAISSRetriever(embedding_dim=embedding_dim)
            retriever.index = faiss.read_index(index_path)
            retriever.is_built = True
            
            # Load metadata
            with open(metadata_path, 'r', encoding='utf-8') as f:
                chunks_metadata = json.load(f)
            
            retriever.chunks_metadata = chunks_metadata
            
            return retriever, chunks_metadata
        
        except Exception as e:
            print(f"Error loading index: {str(e)}")
            return None
    
    def get_stored_pdfs(self) -> List[str]:
        """
        Get list of stored PDF filenames.
        
        Returns:
            List of PDF filenames
        """
        if not os.path.exists(self.pdfs_dir):
            return []
        
        pdf_files = [
            f for f in os.listdir(self.pdfs_dir)
            if f.lower().endswith('.pdf')
        ]
        return sorted(pdf_files)
    
    def delete_index(self, index_name: str = "main_index"):
        """
        Delete stored index files.
        
        Args:
            index_name: Name of the index to delete
        """
        files_to_delete = [
            os.path.join(self.indexes_dir, f"{index_name}.index"),
            os.path.join(self.indexes_dir, f"{index_name}_metadata.json"),
            os.path.join(self.indexes_dir, f"{index_name}_info.json")
        ]
        
        for file_path in files_to_delete:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {str(e)}")
    
    def index_exists(self, index_name: str = "main_index") -> bool:
        """
        Check if index exists on disk.
        
        Args:
            index_name: Name of the index
            
        Returns:
            True if index exists
        """
        index_path = os.path.join(self.indexes_dir, f"{index_name}.index")
        metadata_path = os.path.join(self.indexes_dir, f"{index_name}_metadata.json")
        return os.path.exists(index_path) and os.path.exists(metadata_path)

