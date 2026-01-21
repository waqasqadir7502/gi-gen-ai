"""
Vercel-compatible entry point for the Physical AI Book RAG Chatbot API
This file acts as the handler for Vercel's Python runtime using the ASGI adapter
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
        "https://*.vercel.app"
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

# Import routers with error handling
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

    from health import router as health_router
    from chat import router as chat_router

# Include routers
app.include_router(health_router)
app.include_router(chat_router)

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

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)