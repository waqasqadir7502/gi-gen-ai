import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration class
class Config:
    # Cohere configuration
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")

    # Qdrant configuration
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

    # Database configuration
    DATABASE_URL = os.getenv("DATABASE_URL")

    # Backend API key for authentication
    BACKEND_API_KEY = os.getenv("BACKEND_API_KEY")

    # Qdrant collection name
    COLLECTION_NAME = "physical-ai-book-v1"

    # Vector size for embeddings
    VECTOR_SIZE = 1024

    # Distance metric for similarity search
    DISTANCE_METRIC = "Cosine"

    # Number of results to return from similarity search
    TOP_K = 8

    # Chunk size parameters
    CHUNK_TARGET_SIZE = 768  # tokens
    CHUNK_OVERLAP = 150  # tokens

    # Performance targets
    CHAT_WINDOW_ANIMATION_MS = 300
    MAX_RESPONSE_TIME_S = 5
    MAX_VECTOR_SEARCH_TIME_S = 2

# Validate required environment variables
def validate_config():
    required_vars = ["COHERE_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "BACKEND_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Validate configuration on import
validate_config()

config = Config()