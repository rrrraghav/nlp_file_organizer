"""
Streamlit UI + model-bridge for nlp_file_organizer

Place this file in the project root and run:
    pip install -r requirements.txt
    streamlit run streamlit_nlp_file_organizer.py

Expect to have the following on disk (or adapt paths below):
- models/vectorizer.pkl          # sklearn TfidfVectorizer from lr.ipynb
- models/lr_model.pkl            # sklearn LogisticRegression saved with pickle
- models/tokenizer.pkl           # keras Tokenizer used for RNN
- models/rnn_model.h5           # keras RNN model saved with model.save(...)
- models/roberta/                # optional: HuggingFace saved model folder for RoBERTa or name of HF model
- text-data/                     # optional local dataset folder used to infer class names

This script extracts text from uploaded text-based files (.txt/.docx), runs 1+ models,
shows per-model predictions and confidence, and can move the file into an "organized"
folder by the predicted label.

Note: The script is defensive: if a model or asset is missing it will continue to
run the models that are available.
"""

import streamlit as st
import io
from pathlib import Path
from typing import List, Dict, Tuple

# Import our modular components
from config import MODELS_DIR, TEXT_DATA_DIR, ORGANIZED_DIR, RNN_MAX_LEN, infer_class_names
from text_extraction import extract_text_from_bytes, HAS_DOCX
from model_loader import load_vectorizer_and_lr, load_tokenizer_and_rnn, load_roberta
from classifiers import classify_with_lr, classify_with_rnn, classify_with_roberta

# Get class names
CLASS_NAMES = infer_class_names()


# -------------------- Model loaders (with Streamlit caching) --------------------

@st.cache_resource
def load_vectorizer_and_lr_cached():
    """Load LR models with Streamlit caching."""
    return load_vectorizer_and_lr(MODELS_DIR)


@st.cache_resource
def load_tokenizer_and_rnn_cached():
    """Load RNN models with Streamlit caching."""
    return load_tokenizer_and_rnn(MODELS_DIR)


@st.cache_resource
def load_roberta_cached(preset: str = None):
    """Load RoBERTa model with Streamlit caching."""
    return load_roberta(MODELS_DIR, preset)


# -------------------- Helper functions --------------------

def get_model_status_info(vec, lr, tokenizer, rnn_model, roberta_pipe) -> Dict[str, str]:
    """
    Get status information about loaded models for debugging.
    
    Returns:
        Dictionary mapping model names to status messages
    """
    status = {}
    
    # LR status
    if vec is not None and lr is not None:
        status['LR'] = "✓ Loaded"
    else:
        missing = []
        if vec is None:
            missing.append("vectorizer.pkl")
        if lr is None:
            missing.append("lr_model.pkl")
        status['LR'] = f"✗ Missing: {', '.join(missing)}"
    
    # RNN status
    if tokenizer is not None and rnn_model is not None:
        status['RNN'] = "✓ Loaded"
    else:
        missing = []
        if tokenizer is None:
            missing.append("tokenizer.pkl")
        if rnn_model is None:
            missing.append("rnn_model.h5")
        status['RNN'] = f"✗ Missing: {', '.join(missing)}"
    
    # RoBERTa status
    if roberta_pipe is not None:
        status['RoBERTa'] = "✓ Loaded"
    else:
        status['RoBERTa'] = "✗ Not available (check models/roberta/ or use HF model ID)"
    
    return status


# -------------------- Streamlit app --------------------

def main():
    st.title("NLP File Organizer — Drop a file to classify")
    st.markdown("Upload a text-based file (.txt or .docx). The app will extract text and run one or more models to predict the document class.")

    # Load available models
    vec, lr = load_vectorizer_and_lr_cached()
    tokenizer, rnn_model = load_tokenizer_and_rnn_cached()
    roberta_pipe = load_roberta_cached()

    # Sidebar with model selection and status
    with st.sidebar:
        st.header("Models")
        
        # Show model status
        status = get_model_status_info(vec, lr, tokenizer, rnn_model, roberta_pipe)
        with st.expander("Model Status", expanded=True):
            for model_name, model_status in status.items():
                st.write(f"**{model_name}**: {model_status}")
        
        # Model selection checkboxes
        use_lr = st.checkbox("Logistic Regression (LR)", value=(vec is not None and lr is not None))
        use_rnn = st.checkbox("RNN", value=(tokenizer is not None and rnn_model is not None))
        use_roberta = st.checkbox("RoBERTa", value=(roberta_pipe is not None))
        
        # RoBERTa model input
        roberta_model_input = st.text_input("Or load HF model id (e.g. 'distilroberta-base'):", value="")
        if roberta_model_input.strip():
            # Try to load an HF model on demand
            try:
                rp = load_roberta_cached(roberta_model_input.strip())
                if rp:
                    roberta_pipe = rp
                    use_roberta = True
                    st.success(f"Loaded model: {roberta_model_input}")
                else:
                    st.error(f"Failed to load model: {roberta_model_input}")
            except Exception as e:
                st.error(f"Failed to load HF model: {e}")

    # File uploader (text-based files only)
    uploaded = st.file_uploader("Drop a text-based file here", 
                                type=["txt", "docx"], 
                                accept_multiple_files=False)

    if uploaded is not None:
        raw = uploaded.read()
        
        # Extract text
        st.subheader("Extracted text preview")
        text = extract_text_from_bytes(uploaded.name, raw)
        
        if not text.strip():
            st.warning("⚠️ No text could be extracted from the file.")
            if uploaded.name.lower().endswith('.docx'):
                if not HAS_DOCX:
                    st.info("💡 Install python-docx for DOCX text extraction: `pip install python-docx`")
                else:
                    st.info("💡 The DOCX file may be empty or corrupted.")
            else:
                st.info("💡 The text file may be empty or in an unsupported encoding.")
        else:
            st.code(text[:5000] + ("..." if len(text) > 5000 else ""))
            st.caption(f"Extracted {len(text)} characters")

        # Classification button
        if st.button("Classify", type="primary"):
            if not text.strip():
                st.error("Cannot classify: no text extracted from file.")
            else:
                results = {}
                errors = {}
                
                # Run selected models
                if use_lr:
                    try:
                        results['LR'] = classify_with_lr(text, vec, lr, CLASS_NAMES)
                        if not results['LR']:
                            errors['LR'] = "Model returned no predictions. Check model compatibility."
                    except Exception as e:
                        errors['LR'] = f"Classification error: {str(e)}"
                        results['LR'] = []
                
                if use_rnn:
                    try:
                        results['RNN'] = classify_with_rnn(text, tokenizer, rnn_model, CLASS_NAMES, RNN_MAX_LEN)
                        if not results['RNN']:
                            errors['RNN'] = "Model returned no predictions. Check model compatibility."
                    except Exception as e:
                        errors['RNN'] = f"Classification error: {str(e)}"
                        results['RNN'] = []
                
                if use_roberta:
                    try:
                        results['RoBERTa'] = classify_with_roberta(text, roberta_pipe)
                        if not results['RoBERTa']:
                            errors['RoBERTa'] = "Model returned no predictions. Check model compatibility."
                    except Exception as e:
                        errors['RoBERTa'] = f"Classification error: {str(e)}"
                        results['RoBERTa'] = []

                # Display results
                if not results:
                    st.warning("⚠️ No models ran. Check that models are present in models/ and that you selected at least one model.")
                else:
                    for name, res in results.items():
                        st.subheader(f"{name} results")
                        
                        # Show error if any
                        if name in errors:
                            st.error(f"❌ {errors[name]}")
                        
                        # Show results if available
                        if not res:
                            st.warning("(model not found or failed to run)")
                            # Show debug info
                            with st.expander("Debug info"):
                                if name == 'LR':
                                    st.write(f"Vectorizer: {vec is not None}, LR Model: {lr is not None}")
                                elif name == 'RNN':
                                    st.write(f"Tokenizer: {tokenizer is not None}, RNN Model: {rnn_model is not None}")
                                elif name == 'RoBERTa':
                                    st.write(f"Pipeline: {roberta_pipe is not None}")
                        else:
                            # Present top 5 predictions
                            topk = res[:5]
                            rows = [{"label": label, "score": f"{score:.4f}"} for label, score in topk]
                            st.table(rows)
                            
                            # Show bar chart
                            import pandas as pd
                            chart_df = pd.DataFrame({
                                'Label': [r['label'] for r in rows],
                                'Score': [float(r['score']) for r in rows]
                            })
                            st.bar_chart(chart_df.set_index('Label'))

                    # File organization
                    successful_results = {k: v for k, v in results.items() if v}
                    if successful_results:
                        move_choice = st.selectbox(
                            "Choose which model to use for moving the file:",
                            options=list(successful_results.keys()),
                            index=0
                        )
                        
                        if move_choice:
                            chosen = successful_results[move_choice]
                            if chosen and len(chosen) > 0:
                                predicted_label = chosen[0][0]
                                confidence = chosen[0][1]
                                
                                st.info(f"Top prediction: **{predicted_label}** (confidence: {confidence:.2f})")
                                
                                if st.button(f"Move file to organized/{predicted_label}"):
                                    dest_dir = ORGANIZED_DIR / predicted_label
                                    dest_dir.mkdir(parents=True, exist_ok=True)
                                    dest_path = dest_dir / uploaded.name
                                    
                                    # Write file
                                    with open(dest_path, 'wb') as f:
                                        f.write(raw)
                                    st.success(f"✅ Saved to {dest_path}")

    # Sidebar footer
    st.sidebar.markdown("---")
    st.sidebar.write("**Class names:**")
    st.sidebar.write(CLASS_NAMES)


if __name__ == '__main__':
    main()
