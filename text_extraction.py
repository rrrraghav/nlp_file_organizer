"""
Text extraction utilities for text-based file formats.

Supports:
- DOCX: via python-docx
- Plain text: UTF-8 and Latin-1 (.txt)
"""

from pathlib import Path
from typing import Optional

# Optional imports with graceful fallback
try:
    import docx
    HAS_DOCX = True
except ImportError:
    docx = None
    HAS_DOCX = False


def extract_text_from_docx_bytes(b: bytes) -> str:
    """
    Extract text from DOCX bytes.
    
    Args:
        b: DOCX file bytes
        
    Returns:
        Extracted text string, empty if extraction fails
    """
    if not HAS_DOCX:
        return ""
    
    try:
        tmp_path = Path(".tmp_docx.docx")
        tmp_path.write_bytes(b)
        doc = docx.Document(str(tmp_path))
        paragraphs = [p.text for p in doc.paragraphs]
        tmp_path.unlink(missing_ok=True)
        return "\n".join(paragraphs)
    except Exception as e:
        return ""


def extract_text_from_bytes(filename: str, b: bytes) -> str:
    """
    Extract text from file bytes based on file extension.
    
    Supports only text-based files: .txt and .docx
    
    Args:
        filename: Original filename (used to determine file type)
        b: File bytes
        
    Returns:
        Extracted text string
    """
    ext = filename.lower().split('.')[-1] if '.' in filename else ""
    
    # DOCX files
    if ext == "docx":
        return extract_text_from_docx_bytes(b)
    
    # Plain text files (.txt or no extension)
    # Try UTF-8 first, then Latin-1
    try:
        return b.decode('utf-8')
    except UnicodeDecodeError:
        try:
            return b.decode('latin-1')
        except UnicodeDecodeError:
            return ""
