import joblib
import numpy as np
from pathlib import Path
from datasets import load_dataset
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.utils import preprocess_tweet, compute_metrics, LSTM_DIR

LSTM_DIR.mkdir(parents=True, exist_ok=True)

MAX_VOCAB = 30000
MAX_LEN = 100
EMBED_DIM = 200
BATCH = 128
EPOCHS = 15

def texts_and_labels(ds, split):
    texts = [preprocess_tweet(t) for t in ds[split]["text"]]
    labels = np.array(ds[split]["label"])
    return texts, labels

def build_model(vocab_size):
    model = Sequential([
        Embedding(vocab_size, EMBED_DIM, input_length=MAX_LEN),
        Bidirectional(LSTM(256, return_sequences=True)),
        Dropout(0.4),
        Bidirectional(LSTM(128)),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(3, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def main():
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    train_texts, train_labels = texts_and_labels(ds, "train")
    val_texts, val_labels = texts_and_labels(ds, "validation")
    test_texts, test_labels = texts_and_labels(ds, "test")

    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)
    joblib.dump(tokenizer, LSTM_DIR / "tokenizer.joblib")

    def seqs(texts):
        s = tokenizer.texts_to_sequences(texts)
        return pad_sequences(s, maxlen=MAX_LEN, padding="post", truncating="post")

    X_train = seqs(train_texts)
    X_val = seqs(val_texts)
    X_test = seqs(test_texts)

    y_train = tf.keras.utils.to_categorical(train_labels, 3)
    y_val = tf.keras.utils.to_categorical(val_labels, 3)

    model = build_model(MAX_VOCAB)

    class_weights = {0: 1.3, 1: 1.0, 2: 1.2}

    checkpoint = ModelCheckpoint(str(LSTM_DIR / "best.keras"), save_best_only=True, monitor="val_loss")
    es = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)

    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=EPOCHS,
              batch_size=BATCH,
              class_weight=class_weights,
              callbacks=[checkpoint, es])

    model.save(str(LSTM_DIR / "model_final.keras"))

    preds = np.argmax(model.predict(X_test), axis=1)
    print("Test metrics:", compute_metrics(test_labels, preds))

if __name__ == "__main__":
    main()
