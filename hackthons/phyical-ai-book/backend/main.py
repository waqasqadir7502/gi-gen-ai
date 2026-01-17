import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from dotenv import load_dotenv

# Use absolute imports to avoid path confusion in serverless
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

# Add security and performance middlewares
app.add_middleware(
    GZipMiddleware,
    minimum_size=1000,
)

# CORS - safe, specific origins (add more as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://physical-ai-book-lilac.vercel.app",
        "https://*.vercel.app"          # wildcard for preview deploys
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your custom security headers (unchanged)
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

# Initialize database connection if configured
try:
    from backend.utils.db_connection import db_manager
    # Note: Database tables will be created when first accessed or manually initialized
except ImportError:
    # Database module is optional, so we don't fail if it's not available
    pass

# Include routers (unchanged)
app.include_router(health_router)
app.include_router(chat_router)

# Health check endpoint (simple & useful)
@app.get("/health")
def simple_health():
    return {"status": "healthy", "service": "api-main"}

# Root endpoint for basic verification
@app.get("/")
def read_root():
    return {"message": "Physical AI Book RAG Chatbot API is running", "status": "operational"}

# Optional: add this back only for local development if you want
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8001)