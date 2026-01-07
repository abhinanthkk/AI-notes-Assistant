"""
PDF text extraction module using pdfplumber.
Extracts text page-wise from PDF files.
"""

import pdfplumber
from typing import List, Dict, Tuple
import os


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, any]]:
    """
    Extract text from PDF file page by page.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        List of dictionaries with 'page_number' and 'text' keys
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        Exception: If PDF extraction fails
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    pages_data = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            
            if total_pages == 0:
                raise ValueError(f"PDF file is empty: {pdf_path}")
            
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                
                # Handle empty pages
                if text is None:
                    text = ""
                
                pages_data.append({
                    'page_number': page_num,
                    'text': text.strip(),
                    'total_pages': total_pages
                })
    
    except Exception as e:
        raise Exception(f"Error extracting text from PDF {pdf_path}: {str(e)}")
    
    return pages_data


def extract_text_from_pdfs(pdf_paths: List[str]) -> Dict[str, List[Dict[str, any]]]:
    """
    Extract text from multiple PDF files.
    
    Args:
        pdf_paths: List of paths to PDF files
        
    Returns:
        Dictionary mapping PDF filename to list of page data
    """
    all_pdfs_data = {}
    
    for pdf_path in pdf_paths:
        filename = os.path.basename(pdf_path)
        try:
            pages_data = extract_text_from_pdf(pdf_path)
            all_pdfs_data[filename] = pages_data
        except Exception as e:
            # Log error but continue with other PDFs
            print(f"Warning: Failed to extract text from {filename}: {str(e)}")
            all_pdfs_data[filename] = []
    
    return all_pdfs_data

