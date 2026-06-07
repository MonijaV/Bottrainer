from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from contextlib import asynccontextmanager

from core.pipeline import NLUPipeline
from api.middleware import log_requests   # <-- ADD THIS


# Global pipeline instance
pipeline: NLUPipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    print("Starting BotTrainer API...")
    pipeline = NLUPipeline()
    print("API ready to serve requests")
    yield
    print("Shutting down BotTrainer API...")


app = FastAPI(
    title="BotTrainer NLU API",
    description=(
        "LLM-based Natural Language Understanding pipeline. "
        "Classifies user intent and extracts entities using "
        "FAISS semantic retrieval and Groq LLM inference."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ADD THESE LINES HERE
app.add_middleware(
    BaseHTTPMiddleware,
    dispatch=log_requests
)

# Register routes
from api.routes import router
app.include_router(router)