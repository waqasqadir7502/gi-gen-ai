"""
Vercel serverless function entry point for the Physical AI Book RAG Chatbot API
This file serves as the main handler for Vercel's Python runtime
"""
import os
import sys
from pathlib import Path

# Add the backend directory to the Python path to allow imports
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from dotenv import load_dotenv
from mangum import Mangum

# Load environment variables
load_dotenv()

# Create a new FastAPI app instance
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

# Import routers with comprehensive error handling
try:
    # Add backend to path and import using absolute imports
    from backend.api.chat import router as chat_router
    from backend.api.health import router as health_router

    # Include the routers
    app.include_router(chat_router, prefix="/api")
    app.include_router(health_router, prefix="/health")

    print("API routers imported and included successfully")

except ImportError as e:
    print(f"Error importing API routers: {e}")
    from fastapi import APIRouter

    # Create simple fallback routers
    chat_router = APIRouter()
    health_router = APIRouter()

    @chat_router.post("/chat")
    async def fallback_chat():
        return {
            "answer": "Service temporarily unavailable due to import errors. Please contact the administrator.",
            "sources": [],
            "metadata": {"error": str(e), "status": "degraded"}
        }

    @health_router.get("/")
    async def fallback_health():
        return {
            "status": "degraded",
            "error": f"Import error: {str(e)}",
            "service": "api-main"
        }

    # Include the fallback routers
    app.include_router(chat_router, prefix="/api")
    app.include_router(health_router, prefix="/health")

    print("Fallback routers created and included")

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Physical AI Book RAG Chatbot API is running", "status": "operational"}

# Create the ASGI handler for Vercel - this is the main entry point for the serverless function
handler = Mangum(app, lifespan="off")

print("Vercel API handler initialized successfully")

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)