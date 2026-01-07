"""
Utility functions for text processing and cleaning.
"""

import re
from typing import List


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def count_words(text: str) -> int:
    """
    Count words in text.
    
    Args:
        text: Text string
        
    Returns:
        Word count
    """
    if not text:
        return 0
    return len(text.split())


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    
    Args:
        text: Text string
        
    Returns:
        List of sentences
    """
    if not text:
        return []
    
    # Simple sentence splitting on punctuation
    sentences = re.split(r'[.!?]+\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

