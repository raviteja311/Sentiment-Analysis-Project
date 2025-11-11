
from pathlib import Path
import re
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR  = ROOT / "data"
MODELS_DIR = ROOT / "models"

LR_DIR   = MODELS_DIR / "lr"
LSTM_DIR = MODELS_DIR / "lstm"
GRU_DIR  = MODELS_DIR / "gru"
BERT_DIR = MODELS_DIR / "bert"

for p in (DATA_DIR, MODELS_DIR, LR_DIR, LSTM_DIR, GRU_DIR, BERT_DIR):
    p.mkdir(parents=True, exist_ok=True)


LABELS = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

def preprocess_tweet(text: str) -> str:
    
    if not isinstance(text, str):
        return ""

  
    text = re.sub(r"http\S+", " <url> ", text)
    text = re.sub(r"@\w+", " <user> ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text

def compute_metrics(y_true, y_pred):
    """
    Compute accuracy, macro F1, per-class F1, and confusion matrix.
    Works for LR, LSTM, GRU, BERT predictions.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2]).tolist()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_per_class": f1_per_class,
        "confusion_matrix": cm,
    }


def save_json(obj, path):
    """
    Save dictionary as JSON file.
    Useful for storing metrics or model configs.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    """
    Load JSON file back into a Python dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
