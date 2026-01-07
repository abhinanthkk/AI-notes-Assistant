"""
Text chunking module with overlap support.
Chunks text into smaller pieces for embedding generation.
"""

from typing import List, Dict
from app.utils import count_words, split_into_sentences


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> List[str]:
    """
    Chunk text into smaller pieces with overlap.
    
    Args:
        text: Text to chunk
        chunk_size: Target chunk size in words (default: 500)
        overlap: Overlap size in words (default: 100)
        
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    # Split into sentences first to maintain coherence
    sentences = split_into_sentences(text)
    
    if not sentences:
        return [text] if text.strip() else []
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_words = count_words(sentence)
        
        # If adding this sentence would exceed chunk size
        if current_word_count + sentence_words > chunk_size and current_chunk:
            # Save current chunk
            chunk_text = ' '.join(current_chunk)
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
            
            # Start new chunk with overlap
            # Keep last sentences that fit in overlap window
            overlap_words = 0
            overlap_sentences = []
            
            for s in reversed(current_chunk):
                s_words = count_words(s)
                if overlap_words + s_words <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_words += s_words
                else:
                    break
            
            current_chunk = overlap_sentences
            current_word_count = overlap_words
            current_chunk.append(sentence)
            current_word_count += sentence_words
        else:
            current_chunk.append(sentence)
            current_word_count += sentence_words
    
    # Add final chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if chunk_text.strip():
            chunks.append(chunk_text.strip())
    
    return chunks


def chunk_pages(
    pages_data: List[Dict[str, any]],
    pdf_name: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> List[Dict[str, any]]:
    """
    Chunk pages from a PDF into smaller pieces.
    
    Args:
        pages_data: List of page dictionaries with 'page_number' and 'text'
        pdf_name: Name of the PDF file
        chunk_size: Target chunk size in words
        overlap: Overlap size in words
        
    Returns:
        List of chunk dictionaries with metadata
    """
    all_chunks = []
    
    for page_data in pages_data:
        page_number = page_data['page_number']
        page_text = page_data['text']
        
        if not page_text or not page_text.strip():
            continue
        
        # Chunk the page text
        chunks = chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
        
        # Create chunk metadata
        for chunk_index, chunk_text in enumerate(chunks):
            all_chunks.append({
                'pdf_name': pdf_name,
                'page_number': page_number,
                'chunk_index': chunk_index,
                'chunk_text': chunk_text,
                'word_count': count_words(chunk_text)
            })
    
    return all_chunks

