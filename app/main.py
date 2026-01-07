"""
Main Streamlit application for AI Notes Q&A Assistant.
Provides UI for PDF upload, processing, and question answering.
"""

import streamlit as st
import os
import sys
import tempfile
from typing import List, Dict
import time
import numpy as np

# Add parent directory to path to allow imports
# This works for both local development and Streamlit Cloud
_current_file = os.path.abspath(__file__)
_current_dir = os.path.dirname(_current_file)
_project_root = os.path.dirname(_current_dir)

# Add project root to Python path (ensure it's a string and exists)
if _project_root and os.path.exists(_project_root) and _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Also try adding the current working directory as fallback
_cwd = os.getcwd()
if _cwd and _cwd not in sys.path:
    sys.path.insert(0, _cwd)

# Import modules
from app.pdf_reader import extract_text_from_pdfs
from app.chunker import chunk_pages
from app.embeddings import generate_chunk_embeddings, generate_query_embedding, get_embedding_model
from app.retriever import FAISSRetriever
from app.storage import StorageManager


# Page configuration
st.set_page_config(
    page_title="AI Notes Q&A Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'storage_manager' not in st.session_state:
    st.session_state.storage_manager = StorageManager()

if 'retriever' not in st.session_state:
    st.session_state.retriever = None

if 'chunks_metadata' not in st.session_state:
    st.session_state.chunks_metadata = []

if 'processed_pdfs' not in st.session_state:
    st.session_state.processed_pdfs = []


def load_existing_index():
    """Load existing index from storage if available."""
    storage = st.session_state.storage_manager
    result = storage.load_index_data()
    
    if result:
        retriever, chunks_metadata = result
        st.session_state.retriever = retriever
        st.session_state.chunks_metadata = chunks_metadata
        st.session_state.processed_pdfs = list(set(
            chunk['pdf_name'] for chunk in chunks_metadata
        ))
        return True
    return False


def process_pdfs(uploaded_files: List) -> bool:
    """
    Process uploaded PDFs: extract, chunk, embed, and index.
    
    Args:
        uploaded_files: List of uploaded file objects
        
    Returns:
        True if processing successful, False otherwise
    """
    if not uploaded_files:
        st.error("No PDF files uploaded.")
        return False
    
    storage = st.session_state.storage_manager
    
    # Save uploaded PDFs
    pdf_paths = []
    for uploaded_file in uploaded_files:
        # Save to temporary location first
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        # Copy to storage
        saved_path = storage.save_pdf(tmp_path, uploaded_file.name)
        pdf_paths.append(saved_path)
        
        # Clean up temp file
        os.unlink(tmp_path)
    
    # Extract text from PDFs
    with st.spinner("Extracting text from PDFs..."):
        try:
            all_pdfs_data = extract_text_from_pdfs(pdf_paths)
        except Exception as e:
            st.error(f"Error extracting text: {str(e)}")
            return False
    
    # Check if any text was extracted
    total_pages = sum(len(pages) for pages in all_pdfs_data.values())
    if total_pages == 0:
        st.error("No text could be extracted from the uploaded PDFs.")
        return False
    
    # Chunk text
    with st.spinner("Chunking text..."):
        all_chunks = []
        for pdf_name, pages_data in all_pdfs_data.items():
            if not pages_data:
                continue
            
            chunks = chunk_pages(
                pages_data,
                pdf_name,
                chunk_size=500,
                overlap=100
            )
            all_chunks.extend(chunks)
    
    if not all_chunks:
        st.error("No text chunks created. PDFs may be empty or unreadable.")
        return False
    
    # Generate embeddings
    with st.spinner("Generating embeddings..."):
        try:
            embeddings = generate_chunk_embeddings(all_chunks)
        except Exception as e:
            st.error(f"Error generating embeddings: {str(e)}")
            return False
    
    # Build or update FAISS index
    with st.spinner("Building search index..."):
        try:
            # Get embedding dimension
            model = get_embedding_model()
            embedding_dim = model.get_embedding_dimension()
            
            # Create or update retriever
            if st.session_state.retriever is None:
                retriever = FAISSRetriever(embedding_dim=embedding_dim)
            else:
                retriever = st.session_state.retriever
                # For simplicity, rebuild entire index
                # In production, you might want to add incrementally
            
            # Combine with existing chunks if any
            if st.session_state.chunks_metadata:
                # Rebuild with all chunks
                # For simplicity, regenerate all embeddings
                # In production, you would cache existing embeddings
                all_existing_chunks = st.session_state.chunks_metadata + all_chunks
                all_chunks = all_existing_chunks
                embeddings = generate_chunk_embeddings(all_chunks)
            
            # Build index
            retriever.build_index(embeddings, all_chunks)
            
            # Save to storage
            storage.save_index_data(retriever, all_chunks)
            
            # Update session state
            st.session_state.retriever = retriever
            st.session_state.chunks_metadata = all_chunks
            st.session_state.processed_pdfs = list(set(
                chunk['pdf_name'] for chunk in all_chunks
            ))
            
        except Exception as e:
            st.error(f"Error building index: {str(e)}")
            return False
    
    return True


def search_query(query: str, top_k: int = 5) -> List[Dict]:
    """
    Search for relevant chunks given a query.
    
    Args:
        query: User query string
        top_k: Number of results to return
        
    Returns:
        List of result dictionaries
    """
    if st.session_state.retriever is None or st.session_state.retriever.is_empty():
        return []
    
    try:
        # Generate query embedding
        query_embedding = generate_query_embedding(query)
        
        # Search
        results = st.session_state.retriever.search(query_embedding, top_k=top_k)
        
        return results
    except Exception as e:
        st.error(f"Error searching: {str(e)}")
        return []


# Main UI
def main():
    """Main application function."""
    
    # Title and description
    st.title("📚 AI Notes Q&A Assistant")
    st.markdown("""
    Upload PDF notes and ask questions to find the most relevant information.
    The app uses semantic search with embeddings and FAISS for fast retrieval.
    """)
    
    # Load existing index on startup
    if st.session_state.retriever is None:
        with st.spinner("Loading existing index..."):
            load_existing_index()
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Document Management")
        
        # Show current status
        if st.session_state.retriever and not st.session_state.retriever.is_empty():
            st.success(f"✅ Index loaded: {st.session_state.retriever.get_index_size()} chunks")
            if st.session_state.processed_pdfs:
                st.info(f"📄 PDFs: {len(st.session_state.processed_pdfs)}")
                for pdf_name in st.session_state.processed_pdfs:
                    st.text(f"  • {pdf_name}")
        else:
            st.info("No index loaded. Upload PDFs to get started.")
        
        st.divider()
        
        # PDF Upload
        st.subheader("Upload PDFs")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files to process"
        )
        
        if st.button("🔄 Process & Index PDFs", type="primary"):
            if uploaded_files:
                if process_pdfs(uploaded_files):
                    st.success("PDFs processed and indexed successfully!")
                    st.rerun()
            else:
                st.warning("Please upload at least one PDF file.")
        
        st.divider()
        
        # Reset option
        if st.button("🗑️ Clear Index", type="secondary"):
            if st.session_state.storage_manager.index_exists():
                st.session_state.storage_manager.delete_index()
                st.session_state.retriever = None
                st.session_state.chunks_metadata = []
                st.session_state.processed_pdfs = []
                st.success("Index cleared!")
                st.rerun()
    
    # Main content area
    if st.session_state.retriever is None or st.session_state.retriever.is_empty():
        st.info("👆 Upload and process PDFs using the sidebar to get started.")
    else:
        # Search interface
        st.header("🔍 Ask a Question")
        
        query = st.text_input(
            "Enter your question:",
            placeholder="e.g., What is the main topic discussed?",
            help="Ask any question about the content in your uploaded PDFs"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            search_button = st.button("🔎 Search", type="primary", use_container_width=True)
        with col2:
            top_k = st.number_input("Results", min_value=1, max_value=20, value=5, step=1)
        
        # Perform search
        if search_button and query:
            with st.spinner("Searching..."):
                results = search_query(query, top_k=top_k)
            
            if results:
                st.success(f"Found {len(results)} relevant result(s)")
                st.divider()
                
                # Display results
                for i, result in enumerate(results, 1):
                    with st.container():
                        # Result card
                        col1, col2 = st.columns([4, 1])
                        
                        with col1:
                            st.markdown(f"### Result #{result['rank']}")
                            st.markdown(f"**PDF:** {result['pdf_name']}")
                            st.markdown(f"**Page:** {result['page_number']}")
                        
                        with col2:
                            score = result['similarity_score']
                            st.metric("Similarity", f"{score:.3f}")
                        
                        # Chunk text
                        st.markdown("**Relevant Text:**")
                        st.text_area(
                            "",
                            value=result['chunk_text'],
                            height=150,
                            key=f"result_{i}",
                            label_visibility="collapsed"
                        )
                        
                        st.divider()
            elif query:
                st.warning("No relevant results found. Try rephrasing your question.")
        
        # Show statistics
        with st.expander("📊 Index Statistics"):
            if st.session_state.retriever:
                st.metric("Total Chunks", st.session_state.retriever.get_index_size())
                st.metric("Total PDFs", len(st.session_state.processed_pdfs))
                
                if st.session_state.chunks_metadata:
                    total_words = sum(chunk.get('word_count', 0) for chunk in st.session_state.chunks_metadata)
                    st.metric("Total Words", f"{total_words:,}")


if __name__ == "__main__":
    main()

