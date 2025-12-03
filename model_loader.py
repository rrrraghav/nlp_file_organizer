"""
Model loading utilities with error handling and caching.

Loads pre-trained models from disk:
- Logistic Regression: vectorizer.pkl + lr_model.pkl
- RNN: tokenizer.pkl + rnn_model.h5
- RoBERTa: HuggingFace model (local or remote)
"""

import pickle
from pathlib import Path
from typing import Tuple, Optional

# Optional imports
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    HAS_KERAS = True
except ImportError:
    load_model = None
    pad_sequences = None
    HAS_KERAS = False

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    pipeline = None
    HAS_TRANSFORMERS = False


def load_vectorizer_and_lr(models_dir: Path) -> Tuple[Optional[object], Optional[object]]:
    """
    Load TF-IDF vectorizer and Logistic Regression model.
    
    Args:
        models_dir: Directory containing model files
        
    Returns:
        Tuple of (vectorizer, lr_model), either may be None if file missing
    """
    vec_path = models_dir / "vectorizer.pkl"
    lr_path = models_dir / "lr_model.pkl"
    vec, lr = None, None
    
    if vec_path.exists():
        try:
            with open(vec_path, 'rb') as f:
                vec = pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load vectorizer: {e}")
    
    if lr_path.exists():
        try:
            with open(lr_path, 'rb') as f:
                lr = pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load LR model: {e}")
    
    return vec, lr


def load_tokenizer_and_rnn(models_dir: Path) -> Tuple[Optional[object], Optional[object]]:
    """
    Load tokenizer and RNN model.
    
    Args:
        models_dir: Directory containing model files
        
    Returns:
        Tuple of (tokenizer, rnn_model), either may be None if file missing or Keras unavailable
    """
    tok_path = models_dir / "tokenizer.pkl"
    rnn_path = models_dir / "rnn_model.h5"
    tokenizer, rnn_model = None, None
    
    if tok_path.exists():
        try:
            with open(tok_path, 'rb') as f:
                tokenizer = pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load tokenizer: {e}")
    
    if rnn_path.exists() and HAS_KERAS and load_model is not None:
        try:
            rnn_model = load_model(str(rnn_path))
        except Exception as e:
            print(f"Warning: Failed to load RNN model: {e}")
            rnn_model = None
    
    return tokenizer, rnn_model


def load_roberta(models_dir: Path, preset: Optional[str] = None) -> Optional[object]:
    """
    Load RoBERTa/HuggingFace model pipeline.
    
    Args:
        models_dir: Directory containing model files
        preset: Optional model identifier (local path or HuggingFace model ID)
        
    Returns:
        Pipeline object or None if unavailable
    """
    if not HAS_TRANSFORMERS or pipeline is None:
        return None
    
    try:
        if preset:
            # Load from preset (local path or HF model ID)
            return pipeline("text-classification", model=str(preset), 
                          tokenizer=str(preset), return_all_scores=True)
        
        # Try local roberta directory
        local = models_dir / "roberta"
        if local.exists():
            return pipeline("text-classification", model=str(local), 
                          tokenizer=str(local), return_all_scores=True)
    except Exception as e:
        print(f"Warning: Failed to load RoBERTa model: {e}")
        return None
    
    return None

