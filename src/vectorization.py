from pathlib import Path

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42


def split_data(df):
    """Split messages and labels into training and testing sets."""

    X = df["message"]
    y = df["target"]

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )


def create_vectorizers():
    """Create word-level and character-level TF-IDF vectorizers."""

    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.98,
        sublinear_tf=True
    )

    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=100000,
        sublinear_tf=True
    )

    return word_vectorizer, char_vectorizer


def transform_text(
    X_train,
    X_test,
    word_vectorizer,
    char_vectorizer
):
    """Fit vectorizers on training data and transform both datasets."""

    # Word-level features
    X_train_word = word_vectorizer.fit_transform(X_train)
    X_test_word = word_vectorizer.transform(X_test)

    # Character-level features
    X_train_char = char_vectorizer.fit_transform(X_train)
    X_test_char = char_vectorizer.transform(X_test)

    # Combine both feature sets
    X_train_combined = hstack(
        [X_train_word, X_train_char]
    )

    X_test_combined = hstack(
        [X_test_word, X_test_char]
    )

    return (
        X_train_combined,
        X_test_combined
    )