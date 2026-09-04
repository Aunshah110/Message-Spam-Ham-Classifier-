from pathlib import Path

import joblib
from sklearn.svm import LinearSVC


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "spam_classifier.pkl"
WORD_VECTORIZER_PATH = MODEL_DIR / "word_tfidf_vectorizer.pkl"
CHAR_VECTORIZER_PATH = MODEL_DIR / "char_tfidf_vectorizer.pkl"


def train_final_model(X_train, y_train):
    """Train the final Linear SVM spam classifier."""

    model = LinearSVC(
        C=1.0,
        class_weight={0: 1, 1: 1.2},
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def save_model(
    model,
    word_vectorizer,
    char_vectorizer
):
    """Save model and vectorizers."""

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(model, MODEL_PATH)

    joblib.dump(
        word_vectorizer,
        WORD_VECTORIZER_PATH
    )

    joblib.dump(
        char_vectorizer,
        CHAR_VECTORIZER_PATH
    )

    print("\nSaved files:")
    print(f"Model       : {MODEL_PATH}")
    print(f"Word TF-IDF : {WORD_VECTORIZER_PATH}")
    print(f"Char TF-IDF : {CHAR_VECTORIZER_PATH}")


def load_model():
    """Load trained model and vectorizers."""

    model = joblib.load(MODEL_PATH)

    word_vectorizer = joblib.load(
        WORD_VECTORIZER_PATH
    )

    char_vectorizer = joblib.load(
        CHAR_VECTORIZER_PATH
    )

    return (
        model,
        word_vectorizer,
        char_vectorizer
    )