"""
Classification functions for different model types.

Each classifier takes text input and returns a list of (label, confidence) tuples
sorted by confidence (highest first).
"""

import numpy as np
from typing import List, Tuple, Optional

# Optional imports
try:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    HAS_KERAS = True
except ImportError:
    pad_sequences = None
    HAS_KERAS = False


def classify_with_lr(text: str, vec, lr_model, class_names: List[str]) -> List[Tuple[str, float]]:
    """
    Classify text using Logistic Regression model.
    
    Args:
        text: Input text to classify
        vec: TF-IDF vectorizer
        lr_model: Trained Logistic Regression model
        class_names: List of class label names
        
    Returns:
        List of (label, confidence) tuples sorted by confidence (descending)
        Returns empty list if models unavailable or classification fails
    """
    if vec is None or lr_model is None:
        return []
    
    try:
        # Transform text to TF-IDF features
        Xv = vec.transform([text])
        
        # Get probabilities
        if hasattr(lr_model, 'predict_proba'):
            probs = lr_model.predict_proba(Xv)[0]
        else:
            # Fallback: use decision function and softmax
            scores = lr_model.decision_function(Xv)
            # Handle both 1D and 2D outputs
            if scores.ndim == 1:
                scores = scores.reshape(1, -1)
            # Apply softmax
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            probs = (exp_scores / exp_scores.sum(axis=1, keepdims=True))[0]
        
        # Sort by probability (descending)
        idxs = np.argsort(probs)[::-1]
        return [(class_names[i] if i < len(class_names) else str(i), float(probs[i])) 
                for i in idxs]
    except Exception as e:
        # Return empty list on error (error will be shown in UI)
        print(f"Error in LR classification: {e}")
        return []


def classify_with_rnn(text: str, tokenizer, rnn_model, class_names: List[str], 
                     max_len: int = 200) -> List[Tuple[str, float]]:
    """
    Classify text using RNN model.
    
    Args:
        text: Input text to classify
        tokenizer: Keras tokenizer
        rnn_model: Trained RNN model
        class_names: List of class label names
        max_len: Maximum sequence length (must match training)
        
    Returns:
        List of (label, confidence) tuples sorted by confidence (descending)
        Returns empty list if models unavailable or classification fails
    """
    if tokenizer is None or rnn_model is None or not HAS_KERAS or pad_sequences is None:
        return []
    
    try:
        # Tokenize and pad sequence
        seq = tokenizer.texts_to_sequences([text])
        pad = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
        
        # Get predictions
        probs = rnn_model.predict(pad, verbose=0)[0]
        
        # Sort by probability (descending)
        idxs = np.argsort(probs)[::-1]
        return [(class_names[i] if i < len(class_names) else str(i), float(probs[i])) 
                for i in idxs]
    except Exception as e:
        # Return empty list on error (error will be shown in UI)
        print(f"Error in RNN classification: {e}")
        return []


def classify_with_roberta(text: str, roberta_pipeline) -> List[Tuple[str, float]]:
    """
    Classify text using RoBERTa/HuggingFace model.
    
    Args:
        text: Input text to classify
        roberta_pipeline: HuggingFace text classification pipeline
        
    Returns:
        List of (label, confidence) tuples sorted by confidence (descending)
        Returns empty list if pipeline unavailable or classification fails
    """
    if roberta_pipeline is None:
        return []
    
    try:
        res = roberta_pipeline(text)
        
        # Handle different response formats
        if isinstance(res, list) and len(res) > 0:
            # When return_all_scores=True, result is [[{label, score}, ...]]
            if isinstance(res[0], list):
                scores = res[0]
                sorted_scores = sorted(scores, key=lambda x: x['score'], reverse=True)
                return [(s['label'], float(s['score'])) for s in sorted_scores]
            # Otherwise it's [{label, score}, ...]
            else:
                sorted_scores = sorted(res, key=lambda x: x.get('score', 0.0), reverse=True)
                return [(s['label'], float(s.get('score', 0.0))) for s in sorted_scores]
        
        return []
    except Exception as e:
        # Return empty list on error (error will be shown in UI)
        print(f"Error in RoBERTa classification: {e}")
        return []

