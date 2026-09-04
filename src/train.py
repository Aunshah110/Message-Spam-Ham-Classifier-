from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import LinearSVC

from preprocessing import load_dataset, preprocess_dataset
from vectorization import (
    split_data,
    create_vectorizers,
    transform_text,
)
from model_manager import save_model


# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "Datasets" / "SMSSpamCollection"


# --------------------------------------------------
# Configuration
# --------------------------------------------------

RANDOM_STATE = 42


# --------------------------------------------------
# Model training
# --------------------------------------------------

def train_model(X_train, y_train):
    """Train the Linear SVM classifier."""

    model = LinearSVC(
        C=1.0,
        class_weight={0: 1, 1: 1.2},
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    return model


# --------------------------------------------------
# Model evaluation
# --------------------------------------------------

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Ham", "Spam"],
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# --------------------------------------------------
# Main training pipeline
# --------------------------------------------------

def main():

    print("=" * 50)
    print("SMS SPAM CLASSIFIER - MODEL V2")
    print("=" * 50)

    # ----------------------------------------------
    # 1. Load dataset
    # ----------------------------------------------

    print("\n[1/6] Loading dataset...")

    df = load_dataset(DATA_PATH)

    print(f"Records loaded: {len(df)}")


    # ----------------------------------------------
    # 2. Preprocess dataset
    # ----------------------------------------------

    print("\n[2/6] Preparing dataset...")

    df = preprocess_dataset(df)

    print(f"Records after preprocessing: {len(df)}")


    # ----------------------------------------------
    # 3. Train/test split
    # ----------------------------------------------

    print("\n[3/6] Splitting dataset...")

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(df)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")


    # ----------------------------------------------
    # 4. TF-IDF transformation
    # ----------------------------------------------

    print("\n[4/6] Creating TF-IDF features...")

    (
        word_vectorizer,
        char_vectorizer
    ) = create_vectorizers()

    (
        X_train_tfidf,
        X_test_tfidf
    ) = transform_text(
        X_train,
        X_test,
        word_vectorizer,
        char_vectorizer
    )

    print(
        f"Training feature shape: "
        f"{X_train_tfidf.shape}"
    )

    print(
        f"Testing feature shape : "
        f"{X_test_tfidf.shape}"
    )


    # ----------------------------------------------
    # 5. Train and evaluate model
    # ----------------------------------------------

    print("\n[5/6] Training Linear SVM...")

    model = train_model(
        X_train_tfidf,
        y_train
    )

    print("Training completed.")

    evaluate_model(
        model,
        X_test_tfidf,
        y_test
    )


    # ----------------------------------------------
    # 6. Save model and vectorizers
    # ----------------------------------------------

    print("\n[6/6] Saving model artifacts...")

    save_model(
        model,
        word_vectorizer,
        char_vectorizer
    )

    print("\nTraining pipeline completed successfully.")


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":
    main()