"""
SHL Assessment Advisor — FastAPI Application
--------------------------------------------
Endpoints:
    GET  /health  →  {"status": "ok"}
    POST /chat    →  ChatResponse
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()

from api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== SHL Assessment Advisor starting ===")

    if not os.getenv("GROQ_API_KEY"):
        logger.error("GROQ_API_KEY is not set. Add it to your .env file.")
        sys.exit(1)

    chroma_dir = Path(__file__).parent.parent / "data" / "chroma_db"
    if not chroma_dir.exists():
        logger.error(
            "ChromaDB not found at %s. Run: python rag/embedder.py", chroma_dir
        )
        sys.exit(1)

    # Pre-warm retriever at startup to avoid cold start on first request
    try:
        from rag.retriever import get_retriever
        retriever = get_retriever()
        logger.info("Retriever ready. Vector store has %d assessments.",
                    retriever._store.count())
    except Exception as e:
        logger.error("Failed to initialise retriever: %s", e)
        sys.exit(1)

    logger.info("=== Startup complete. Ready to serve ===")
    yield
    logger.info("=== Shutting down ===")


app = FastAPI(
    title="SHL Assessment Advisor",
    description="Conversational agent for SHL assessment recommendations.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,
        log_level="info",
    )
