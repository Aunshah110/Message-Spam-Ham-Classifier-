# SMS Spam Classifier

An NLP-based machine learning system that classifies SMS messages as 'SPAM' or 'NOT SPAM'.

## Dataset

The project uses the 'SMS Spam Collection*' dataset containing 5,572 SMS messages.

* Ham: 4,825
* Spam: 747
* Duplicate records removed before training
* 80% training / 20% testing
* Stratified train-test split

## Data Processing

The messages are processed using:

* Lowercasing
* URL and HTML removal
* Punctuation handling
* Stop-word removal
* Lemmatization
* Duplicate removal

The original messages are preserved, while cleaned text is used where required.

## Feature Extraction

The system combines two types of **TF-IDF features**:

### Word TF-IDF

Captures important words and word combinations using unigram and bigram features.

### Character TF-IDF

Captures character patterns, spelling variations, and common spam-like text patterns.

Combining both improves the model's ability to recognize different forms of spam messages.

## Machine Learning Model

The final classifier uses **Linear Support Vector Machine (Linear SVM)**.

The model was selected after comparing it with traditional approaches such as Multinomial Naive Bayes and Logistic Regression.

## Model Results

| Metric    |       Score |
| --------- | ----------: |
| Accuracy  |  **99.52%** |
| Precision | **100.00%** |
| Recall    |  **96.15%** |
| F1 Score  |  **98.04%** |

### Confusion Matrix

```text
[[902   0]
 [  5 125]]
```

The model correctly classified **902 ham** and **125 spam** messages, with only **5 spam messages incorrectly classified as ham**.

## Saved Model Files

The trained model and TF-IDF vectorizers are saved in the `models/` directory:

```text
models/
├── spam_classifier.pkl
├── word_tfidf_vectorizer.pkl
└── char_tfidf_vectorizer.pkl
```

## How to Try the System

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Run the FastAPI application from the project root:

```bash
uvicorn app.app:app --reload
```

3. Open the URL shown in the terminal, usually:

```text
http://127.0.0.1:8000
```

4. Enter an SMS message and click **Check Message** to see whether it is classified as **SPAM** or **NOT SPAM**.

> *Note: The reported accuracy is based on the held-out test set. Manually testing individual messages is useful for checking system behavior but is not an official accuracy measurement.
