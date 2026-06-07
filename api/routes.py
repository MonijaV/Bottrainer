from fastapi import APIRouter, Header, HTTPException, Depends
from pydantic import BaseModel
import os
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()


# ── Request / Response Models ─────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str

    model_config = {"json_schema_extra": {
        "example": {"text": "Book a flight to Delhi tomorrow"}
    }}


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model: str


class IntentsResponse(BaseModel):
    intents: list[str]
    total: int


# ── Auth ──────────────────────────────────────────────────────────────────────

def verify_api_key(x_api_key: str = Header(...)):
    """
    Simple API key authentication.
    Key must be passed as X-Api-Key header.

    Why auth matters:
    Without it anyone can spam your Groq API quota.
    Shows security awareness to interviewers.
    """
    expected = os.getenv("API_KEY", "bottrainer_secret_key_123")
    if x_api_key != expected:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Pass your key as X-Api-Key header."
        )
    return x_api_key


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Check if API is running.
    No auth required — used for deployment monitoring.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        model="llama-3.1-8b-instant via Groq"
    )


@router.get("/intents", response_model=IntentsResponse, tags=["Data"])
async def get_intents(api_key: str = Depends(verify_api_key)):
    """
    Returns all available intents the system can classify.
    Useful for frontend dropdowns and documentation.
    """
    from api.main import pipeline

    intents = pipeline.valid_intents
    return IntentsResponse(
        intents=intents,
        total=len(intents)
    )


@router.post("/predict", tags=["Prediction"])
async def predict(
    request: PredictRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Main NLU prediction endpoint.

    Takes user text, returns:
    - intent: classified intent
    - entities: extracted key information
    - similarity_score: FAISS confidence (0-1)
    - latency breakdown: retrieval, llm, total time in ms
    - retrieved_examples: examples used for classification
    """
    from api.main import pipeline

    if not request.text or not request.text.strip():
        raise HTTPException(
            status_code=422,
            detail="Text field cannot be empty"
        )

    response = await pipeline.predict(request.text)
    return response


@router.post("/evaluate", tags=["Evaluation"])
async def run_evaluation(api_key: str = Depends(verify_api_key)):
    """
    Triggers evaluation run on test dataset.
    Returns accuracy, F1 scores, and per-intent metrics.
    Note: Takes 2-5 minutes to complete on full eval set.
    """
    from api.main import pipeline
    from data.data_loader import DataLoader

    loader = DataLoader()
    eval_samples = loader.load_eval_set()

    true_labels = []
    predicted_labels = []
    errors = []

    for sample in eval_samples[:20]:  # Quick eval — first 20 samples
        response = await pipeline.predict(sample.text)
        true_labels.append(sample.intent)
        predicted_labels.append(response.intent)

        if response.intent != sample.intent:
            errors.append({
                "text": sample.text,
                "expected": sample.intent,
                "got": response.intent,
                "similarity": response.similarity_score
            })

    correct = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p)
    accuracy = correct / len(true_labels)

    return {
        "samples_evaluated": len(true_labels),
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "errors": errors[:5]  # Show first 5 errors
    }


@router.get("/metrics", tags=["Monitoring"])
async def get_metrics(api_key: str = Depends(verify_api_key)):
    """
    Returns performance metrics from SQLite logs.
    Shows average latency, intent distribution, total queries.
    """
    from api.main import pipeline

    stats = pipeline.logger.get_stats()
    return stats