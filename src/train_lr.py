import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from src.utils import preprocess_tweet, compute_metrics, LR_DIR

LR_DIR.mkdir(parents=True, exist_ok=True)

def ds_to_df(ds, split):
    texts = [preprocess_tweet(t) for t in ds[split]['text']]
    labels = ds[split]['label']
    return pd.DataFrame({"text": texts, "label": labels})

def main():
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    train_df = ds_to_df(ds, "train")
    val_df = ds_to_df(ds, "validation")
    test_df = ds_to_df(ds, "test")

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9
        )),
        ('clf', LogisticRegression(
            max_iter=2000,
            class_weight='balanced',
            solver='saga',
            multi_class='multinomial'
        ))
    ])

    params = {
        'tfidf__max_features': [10000],
        'clf__C': [1.0, 2.0, 0.5]
    }

    gs = GridSearchCV(
        pipe,
        params,
        cv=3,
        scoring='f1_macro',
        verbose=1,
        n_jobs=-1
    )

    gs.fit(train_df['text'], train_df['label'])
    best = gs.best_estimator_

    val_preds = best.predict(val_df['text'])
    test_preds = best.predict(test_df['text'])

    print("Validation:", compute_metrics(val_df['label'], val_preds))
    print("Test:", compute_metrics(test_df['label'], test_preds))

    model_path = LR_DIR / "pipeline.joblib"
    joblib.dump(best, model_path)
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    main()
