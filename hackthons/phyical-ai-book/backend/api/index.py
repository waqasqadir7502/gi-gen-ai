"""
Vercel serverless function entry point for the Physical AI Book RAG Chatbot API
This file acts as the main entry point for Vercel's Python runtime
"""
import os
import sys
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from dotenv import load_dotenv
from mangum import Mangum

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

# CORS - safe origins for Vercel deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://physical-ai-book-lilac.vercel.app",
        "https://*.vercel.app",
        "https://physical-ai-book-git-main-waqasqadir7502-gmailcom.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "no-referrer-when-downgrade"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    return response

# Import and include routers with error handling
try:
    # Try relative imports first
    from .health import router as health_router
    from .chat import router as chat_router
except (ImportError, ValueError):
    # Fallback to absolute imports
    import sys
    from pathlib import Path
    # Add the backend directory to the path
    backend_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(backend_dir))

    try:
        from health import router as health_router
        from chat import router as chat_router
    except ImportError as e:
        print(f"Error importing routers: {e}")
        # Define minimal fallback routers in case of import errors
        from fastapi import APIRouter
        health_router = APIRouter()
        chat_router = APIRouter()

        @health_router.get("/")
        async def fallback_health():
            return {"status": "degraded", "error": "Health router failed to load"}

        @chat_router.post("/chat")
        async def fallback_chat():
            return {"answer": "Chat service temporarily unavailable", "sources": [], "metadata": {"error": "Chat router failed to load"}}

app.include_router(health_router, prefix="", tags=["health"])
app.include_router(chat_router, prefix="/api", tags=["chat"])  # Add /api prefix at the app level

# Health check endpoint
@app.get("/health")
def health_check():
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

# Create the ASGI handler for Vercel
handler = Mangum(app, lifespan="off")

# For direct execution (development)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)