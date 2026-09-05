from pathlib import Path

import joblib
from scipy.sparse import hstack
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "spam_classifier.pkl"
WORD_VECTORIZER_PATH = MODEL_DIR / "word_tfidf_vectorizer.pkl"
CHAR_VECTORIZER_PATH = MODEL_DIR / "char_tfidf_vectorizer.pkl"


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="SMS Spam Classifier API",
    description="API for detecting spam and normal SMS messages.",
    version="1.0.0"
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = BASE_DIR / "app" / "static"

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)

# --------------------------------------------------
# Request schema
# --------------------------------------------------

class MessageRequest(BaseModel):
    message: str


# --------------------------------------------------
# Load model artifacts
# --------------------------------------------------

def load_artifacts():

    required_files = {
        "model": MODEL_PATH,
        "word vectorizer": WORD_VECTORIZER_PATH,
        "character vectorizer": CHAR_VECTORIZER_PATH,
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
# Load artifacts when API starts
# --------------------------------------------------

try:

    (
        model,
        word_vectorizer,
        char_vectorizer
    ) = load_artifacts()

    MODEL_LOADED = True

except FileNotFoundError:

    model = None
    word_vectorizer = None
    char_vectorizer = None

    MODEL_LOADED = False


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/")
def home():
    return FileResponse(
        STATIC_DIR / "index.html"
    )

# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(request: MessageRequest):

    if not MODEL_LOADED:

        raise HTTPException(
            status_code=500,
            detail="Model artifacts are not available."
        )

    message = request.message.strip()

    if not message:

        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    # ----------------------------------------------
    # Word TF-IDF
    # ----------------------------------------------

    word_features = word_vectorizer.transform(
        [message]
    )

    # ----------------------------------------------
    # Character TF-IDF
    # ----------------------------------------------

    char_features = char_vectorizer.transform(
        [message]
    )

    # ----------------------------------------------
    # Combine features
    # ----------------------------------------------

    combined_features = hstack(
        [
            word_features,
            char_features
        ]
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

    label = (
        "SPAM"
        if prediction == 1
        else "NOT SPAM"
    )

    return {
        "message": message,
        "prediction": label,
        "decision_score": round(
            float(decision_score),
            4
        )
    }