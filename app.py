import streamlit as st
import joblib
import numpy as np
from src.utils import preprocess_tweet, LABELS, LR_DIR, LSTM_DIR, GRU_DIR, BERT_DIR

st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("Sentiment Analysis — LR, LSTM, GRU, BERT")

MODEL_CHOICES = ["LogisticRegression", "LSTM", "GRU", "BERT"]
model_choice = st.sidebar.selectbox("Select Model", MODEL_CHOICES)

@st.cache_resource
def load_lr():
    return joblib.load(LR_DIR / "pipeline.joblib")

@st.cache_resource
def load_lstm():
    from tensorflow.keras.models import load_model
    tok = joblib.load(LSTM_DIR / "tokenizer.joblib")
    mdl = load_model(str(LSTM_DIR / "model_final.keras"))
    return tok, mdl

@st.cache_resource
def load_gru():
    from tensorflow.keras.models import load_model
    tok = joblib.load(GRU_DIR / "tokenizer.joblib")
    mdl = load_model(str(GRU_DIR / "model_final.h5"))
    return tok, mdl

@st.cache_resource
def load_bert():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(str(BERT_DIR))
    mdl = AutoModelForSequenceClassification.from_pretrained(str(BERT_DIR))
    mdl.eval()
    return tok, mdl

st.write(f"Using Model: **{model_choice}**")

text = st.text_area("Enter tweet text", height=120)

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter a valid text.")
        st.stop()

    t = preprocess_tweet(text)

    if model_choice == "LogisticRegression":
        pipe = load_lr()
        probs = pipe.predict_proba([t])[0]
        pred = int(np.argmax(probs))

    elif model_choice == "LSTM":
        tok, mdl = load_lstm()
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        seq = tok.texts_to_sequences([t])
        seq = pad_sequences(seq, maxlen=80, padding='post', truncating='post')
        probs = mdl.predict(seq)[0]
        pred = int(np.argmax(probs))

    elif model_choice == "GRU":
        tok, mdl = load_gru()
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        seq = tok.texts_to_sequences([t])
        seq = pad_sequences(seq, maxlen=80, padding='post', truncating='post')
        probs = mdl.predict(seq)[0]
        pred = int(np.argmax(probs))

    else:
        tok, mdl = load_bert()
        import torch
        enc = tok(t, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            logits = mdl(**enc).logits
            probs = torch.softmax(logits, dim=-1).numpy().squeeze()
        pred = int(np.argmax(probs))

    st.subheader(f"Prediction: **{LABELS[pred].upper()}**")
    st.write(f"Confidence: {float(probs[pred]):.3f}")

    st.table({
        "class": [LABELS[i] for i in range(3)],
        "probability": [float(p) for p in probs]
    })
