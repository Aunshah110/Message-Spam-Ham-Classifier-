from pathlib import Path

import joblib
from scipy.sparse import hstack


# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "spam_classifier.pkl"
WORD_VECTORIZER_PATH = MODEL_DIR / "word_tfidf_vectorizer.pkl"
CHAR_VECTORIZER_PATH = MODEL_DIR / "char_tfidf_vectorizer.pkl"


# --------------------------------------------------
# Load trained artifacts
# --------------------------------------------------

def load_artifacts():
    """Load the trained model and TF-IDF vectorizers."""

    required_files = {
        "Model": MODEL_PATH,
        "Word vectorizer": WORD_VECTORIZER_PATH,
        "Character vectorizer": CHAR_VECTORIZER_PATH,
    }

    for name, path in required_files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{name} not found: {path}"
            )

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


# --------------------------------------------------
# Predict message
# --------------------------------------------------

def predict_message(
    message,
    model,
    word_vectorizer,
    char_vectorizer
):
    """
    Predict whether a message is HAM or SPAM.

    Returns:
        label: HAM or SPAM
        score: SVM decision score
    """

    if not isinstance(message, str):
        raise ValueError(
            "Message must be a string."
        )

    message = message.strip()

    if not message:
        raise ValueError(
            "Message cannot be empty."
        )

    # ----------------------------------------------
    # Word-level TF-IDF
    # ----------------------------------------------

    word_features = word_vectorizer.transform(
        [message]
    )

    # ----------------------------------------------
    # Character-level TF-IDF
    # ----------------------------------------------

    char_features = char_vectorizer.transform(
        [message]
    )

    # ----------------------------------------------
    # Combine features
    # ----------------------------------------------

    combined_features = hstack(
        [word_features, char_features]
    )

    # ----------------------------------------------
    # Prediction
    # ----------------------------------------------

    prediction = model.predict(
        combined_features
    )[0]

    decision_score = model.decision_function(
        combined_features
    )[0]

    label = "SPAM" if prediction == 1 else "NOT SPAM"

    return label, decision_score


# --------------------------------------------------
# Display prediction
# --------------------------------------------------

def display_prediction(
    message,
    label,
    decision_score
):
    """Display prediction results."""

    print("\n" + "-" * 50)
    print(f"Message : {message}")
    print(f"Prediction : {label}")
    print(f"SVM Score  : {decision_score:.4f}")
    print("-" * 50)


# --------------------------------------------------
# Main application
# --------------------------------------------------

def main():

    print("=" * 50)
    print("        SMS SPAM CLASSIFIER - V2")
    print("=" * 50)

    try:
        (
            model,
            word_vectorizer,
            char_vectorizer
        ) = load_artifacts()

    except FileNotFoundError as error:
        print(f"\nError: {error}")
        return

    print("\nModel loaded successfully.")
    print("Type 'exit' to close the classifier.")

    while True:

        message = input(
            "\nEnter a message: "
        )

        if message.strip().lower() == "exit":
            print("\nClassifier closed.")
            break

        try:

            label, decision_score = predict_message(
                message,
                model,
                word_vectorizer,
                char_vectorizer
            )

            display_prediction(
                message,
                label,
                decision_score
            )

        except ValueError as error:
            print(f"\nError: {error}")


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":
    main()