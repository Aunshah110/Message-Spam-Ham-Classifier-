import re
import string
from pathlib import Path

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ============================================================
# Configuration
# ============================================================

DATA_PATH = Path("Datasets/SMSSpamCollection")

LABEL_MAP = {
    "ham": 0,
    "spam": 1
}


# ============================================================
# NLP Components
# ============================================================

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# ============================================================
# Load Dataset
# ============================================================

def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the SMS Spam Collection dataset.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "message"]
    )

    return df


# ============================================================
# Dataset Analysis
# ============================================================

def analyze_dataset(dataframe: pd.DataFrame) -> None:
    """
    Display basic information about the dataset.
    """

    print("=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)

    print(f"Rows       : {len(dataframe):,}")
    print(f"Columns    : {dataframe.shape[1]}")
    print(f"Missing    : {dataframe.isnull().sum().sum():,}")
    print(f"Duplicates : {dataframe.duplicated().sum():,}")

    print("\nColumn Information:")
    print(dataframe.dtypes)

    print("\nLabel Distribution:")
    print(dataframe["label"].value_counts())

    print("\nLabel Percentage:")
    print(
        (dataframe["label"].value_counts(normalize=True) * 100)
        .round(2)
    )

    print("\nMessage Length:")
    print(dataframe["message"].str.len().describe())


# ============================================================
# Text Preprocessing
# ============================================================

def preprocess_text(text: str) -> str:
    """
    Clean and preprocess a single SMS message.
    """

    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove HTML tags
    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Keep alphabetic words
    tokens = re.findall(
        r"\b[a-z]+\b",
        text
    )

    # Remove stopwords
    tokens = [
        word
        for word in tokens
        if word not in STOP_WORDS
    ]

    # Lemmatize words
    tokens = [
        LEMMATIZER.lemmatize(word)
        for word in tokens
    ]

    # Return cleaned text
    return " ".join(tokens)


# ============================================================
# Dataset Preprocessing
# ============================================================

def preprocess_dataset(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Clean and prepare the complete dataset.
    """

    df = dataframe.copy()

    # Remove duplicate messages
    df = df.drop_duplicates(
        subset=["label", "message"]
    ).reset_index(drop=True)

    # Create cleaned message column
    df["clean_message"] = (
        df["message"]
        .apply(preprocess_text)
    )

    # Convert labels to numerical values
    df["target"] = df["label"].map(LABEL_MAP)

    # Validate labels
    if df["target"].isnull().any():
        raise ValueError(
            "Dataset contains an unknown label."
        )

    # Remove messages that became empty
    df = df[
        df["clean_message"].str.strip().ne("")
    ].reset_index(drop=True)

    return df


# ============================================================
# Dataset Validation
# ============================================================

def validate_dataset(
    dataframe: pd.DataFrame
) -> None:
    """
    Validate the processed dataset.
    """

    print("=" * 50)
    print("FINAL DATASET VALIDATION")
    print("=" * 50)

    print(f"Rows: {len(dataframe):,}")
    print(f"Columns: {dataframe.shape[1]}")

    print("\nMissing values:")
    print(dataframe.isnull().sum())

    print("\nDuplicate rows:")
    print(dataframe.duplicated().sum())

    print("\nTarget distribution:")
    print(dataframe["target"].value_counts())

    print("\nEmpty processed messages:")
    print(
        dataframe["clean_message"]
        .str.strip()
        .eq("")
        .sum()
    )


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":

    df = load_dataset()

    analyze_dataset(df)

    df = preprocess_dataset(df)

    validate_dataset(df)

    print("\nSample processed data:")
    print(
        df[
            ["label", "message", "clean_message", "target"]
        ].head(10)
    )