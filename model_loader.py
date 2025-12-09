"""
Model loading utilities with error handling and caching.

Loads pre-trained models from disk:
- Logistic Regression: lr_tuned.pkl (Pipeline containing vectorizer + model)
- RNN: tokenizer.pkl + rnn_classifier.keras
- RoBERTa: HuggingFace model (local or remote)
"""

import pickle
from pathlib import Path
from typing import Tuple, Optional
import torch

# Optional imports
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    joblib = None
    HAS_JOBLIB = False

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
    
    Tries to load lr_tuned.pkl (Pipeline) first, then falls back to separate files.
    
    Args:
        models_dir: Directory containing model files
        
    Returns:
        Tuple of (vectorizer, lr_model), either may be None if file missing
    """
    vec, lr = None, None
    
    # Try loading the tuned pipeline first (lr_tuned.pkl contains both vectorizer and model)
    lr_tuned_path = models_dir / "lr_tuned.pkl"
    if lr_tuned_path.exists() and HAS_JOBLIB:
        try:
            lr_pipeline = joblib.load(str(lr_tuned_path))
            # Extract vectorizer and model from pipeline
            if hasattr(lr_pipeline, 'named_steps'):
                vec = lr_pipeline.named_steps.get('tfidf', None)
                lr = lr_pipeline.named_steps.get('clf', None)
            elif hasattr(lr_pipeline, 'steps'):
                # Fallback for older pipeline format
                for name, step in lr_pipeline.steps:
                    if name == 'tfidf':
                        vec = step
                    elif name == 'clf':
                        lr = step
            else:
                # If it's not a pipeline, treat it as the model itself
                lr = lr_pipeline
        except Exception as e:
            print(f"Warning: Failed to load LR pipeline from lr_tuned.pkl: {e}")
    
    # Fallback: try loading separate files
    if vec is None or lr is None:
        vec_path = models_dir / "vectorizer.pkl"
        lr_path = models_dir / "lr_model.pkl"
        
        if vec_path.exists() and vec is None:
            try:
                with open(vec_path, 'rb') as f:
                    vec = pickle.load(f)
            except Exception as e:
                print(f"Warning: Failed to load vectorizer: {e}")
        
        if lr_path.exists() and lr is None:
            try:
                if HAS_JOBLIB:
                    lr = joblib.load(str(lr_path))
                else:
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
    rnn_path = models_dir / "rnn_classifier.keras"
    tokenizer, rnn_model = None, None
    
    if tok_path.exists():
        try:
            with open(tok_path, 'rb') as f:
                tokenizer = pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load tokenizer: {e}")
    else:
        print(f"Warning: tokenizer.pkl not found at {tok_path}. RNN model requires a tokenizer.")
    
    if rnn_path.exists() and HAS_KERAS and load_model is not None:
        try:
            rnn_model = load_model(str(rnn_path))
        except Exception as e:
            print(f"Warning: Failed to load RNN model: {e}")
            rnn_model = None
    else:
        # Fallback: try old .h5 format
        rnn_path_h5 = models_dir / "rnn_model.h5"
        if rnn_path_h5.exists() and HAS_KERAS and load_model is not None:
            try:
                rnn_model = load_model(str(rnn_path_h5))
            except Exception as e:
                print(f"Warning: Failed to load RNN model from .h5: {e}")
                rnn_model = None
    
    return tokenizer, rnn_model


def load_roberta(models_dir: Path, preset: Optional[str] = None) -> Optional[object]:
    """
    Load RoBERTa/HuggingFace model pipeline.

    Args:
        models_dir: Directory containing model files (project-local).
        preset: Optional model identifier (local path or HuggingFace model ID).
                Can be a Path object or a string with an absolute path.

    Returns:
        Pipeline object or None if unavailable
    """
    if not HAS_TRANSFORMERS or pipeline is None:
        print("transformers not available -> cannot load RoBERTa pipeline")
        return None

    device = 0 if torch.cuda.is_available() else -1  # GPU if available, else CPU

    try:
        # If a preset is provided, treat it either as a local path or an HF id
        if preset:
            # accept Path or string
            p = Path(preset)
            if p.exists():
                # local model folder -> pass as model and tokenizer to pipeline
                return pipeline(
                    "text-classification",
                    model=str(p),
                    tokenizer=str(p),
                    return_all_scores=True,
                    device=device,
                )
            else:
                # preset is probably a HF model id (e.g. "distilroberta-base")
                return pipeline(
                    "text-classification",
                    model=preset,
                    tokenizer=preset,
                    return_all_scores=True,
                    device=device,
                )

        # No preset: try the most common local locations
        # 1) models/roberta
        local = Path(models_dir) / "roberta"
        if local.exists():
            return pipeline(
                "text-classification",
                model=str(local),
                tokenizer=str(local),
                return_all_scores=True,
                device=device,
            )

        # 2) models/RoBERTa-data/final_roberta_model (your provided structure)
        alt = Path(models_dir) / "RoBERTa-data" / "final_roberta_model"
        if alt.exists():
            return pipeline(
                "text-classification",
                model=str(alt),
                tokenizer=str(alt),
                return_all_scores=True,
                device=device,
            )

    except Exception as e:
        # keep a helpful message for debugging
        print(f"Warning: Failed to load RoBERTa model: {e}")
        return None

    # nothing found
    return None


