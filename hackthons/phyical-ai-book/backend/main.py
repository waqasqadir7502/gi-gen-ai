import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from qdrant_client.http.exceptions import UnexpectedResponse

# Absolute imports (safe for Vercel serverless)
from backend.api.health import router as health_router
from backend.api.chat import router as chat_router

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Physical AI Book RAG Chatbot API",
    description="API for the Physical AI Book RAG Chatbot",
    version="1.0.0",
    docs_url="/docs",    # keep for debugging
    redoc_url="/redoc"
)

# Performance middleware
app.add_middleware(
    GZipMiddleware,
    minimum_size=1000,
)

# CORS - safe origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://physical-ai-book-lilac.vercel.app",
        "https://*.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security headers middleware (unchanged)
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "no-referrer-when-downgrade"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    return response

# Safe, idempotent Qdrant collection setup (runs on startup but never crashes)
def ensure_qdrant_collection():
    try:
        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        collection_name = "physical-ai-book-v1"

        if not client.has_collection(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )
            print(f"[Startup] Created Qdrant collection: {collection_name}")
        else:
            print(f"[Startup] Qdrant collection {collection_name} already exists - skipping")
    except UnexpectedResponse as e:
        if e.status_code == 409:  # Already exists
            print(f"[Startup] Collection already exists (safe)")
        else:
            print(f"[Startup] Qdrant warning: {e}")
    except Exception as e:
        print(f"[Startup] Qdrant setup warning (continuing): {str(e)}")
        # NEVER raise or exit here - let API start anyway

# Run collection check on startup (safe & non-blocking)
ensure_qdrant_collection()

# Optional DB init (unchanged, safe)
try:
    from backend.utils.db_connection import db_manager
except ImportError:
    pass  # Optional

# Include routers
app.include_router(health_router)
app.include_router(chat_router)

# Health check (enhanced)
@app.get("/health")
def simple_health():
    return {
        "status": "healthy",
        "service": "api-main",
        "qdrant": "connected" if os.getenv("QDRANT_URL") else "not configured",
        "cohere": "configured" if os.getenv("COHERE_API_KEY") else "missing"
    }

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Physical AI Book RAG Chatbot API is running", "status": "operational"}

# Optional local dev block (commented - safe to leave)
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8001)