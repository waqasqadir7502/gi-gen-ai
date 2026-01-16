import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from dotenv import load_dotenv

# Import the routers using relative imports
from api.health import router as health_router
from api.chat import router as chat_router

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Physical AI Book RAG Chatbot API",
    description="API for the Physical AI Book RAG Chatbot",
    version="1.0.0"
)

# Add security and performance middlewares
app.add_middleware(
    GZipMiddleware,
    minimum_size=1000,
)

# Add CORS middleware with specific origins (avoid wildcard in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add TrustedHost middleware for production
# app.add_middleware(
#     TrustedHostMiddleware,
#     allowed_hosts=["localhost", "127.0.0.1", "0.0.0.0", "*.vercel.app", "physical-ai-book-lilac.vercel.app", "*.example.com"],
# )

# Add security headers manually since FastAPI doesn't have a built-in security middleware
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "no-referrer-when-downgrade"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    return response

# Initialize database connection if configured
try:
    from utils.db_connection import db_manager
    # Note: Database tables will be created when first accessed or manually initialized
except ImportError:
    # Database module is optional, so we don't fail if it's not available
    pass

# Include routers
app.include_router(health_router)
app.include_router(chat_router)

@app.get("/")
def read_root():
    return {"message": "Physical AI Book RAG Chatbot API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)