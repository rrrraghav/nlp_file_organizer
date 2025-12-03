"""
Configuration constants and utilities.
"""

import pickle
from pathlib import Path
from typing import List


# Directory paths
MODELS_DIR = Path("models")
TEXT_DATA_DIR = Path("text-data")
ORGANIZED_DIR = Path("organized")

# Model parameters (must match training)
RNN_MAX_LEN = 200
RNN_MAX_WORDS = 20000


def infer_class_names() -> List[str]:
    """
    Infer class names from saved file or directory structure.
    
    Priority:
    1. models/class_names.pkl (saved class names)
    2. Directory names in text-data/
    3. Generic fallback (class_0, class_1, ...)
    
    Returns:
        List of class name strings
    """
    # Try to load saved class names first
    class_names_path = MODELS_DIR / "class_names.pkl"
    if class_names_path.exists():
        try:
            with open(class_names_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    
    # Otherwise fall back to directory names under text-data
    if TEXT_DATA_DIR.exists() and TEXT_DATA_DIR.is_dir():
        dirs = sorted([d.name for d in TEXT_DATA_DIR.iterdir() if d.is_dir()])
        if dirs:
            return dirs
    
    # Final fallback: generic class names
    return [f"class_{i}" for i in range(16)]

