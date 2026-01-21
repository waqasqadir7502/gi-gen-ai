from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os

# Handle relative imports for direct execution
try:
    from ..config import config
    from ..services.retrieval_service import retrieval_service
    from ..services.generation_service import generation_service
    from ..middleware.auth import verify_api_key
    from ..middleware.rate_limiter import check_rate_limit
    from ..utils.logger import log_info, log_error
except (ImportError, ValueError):
    # Fallback for direct execution
    import sys
    from pathlib import Path
    # Add the backend directory to the path
    backend_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(backend_dir))

    from config import config
    from services.retrieval_service import retrieval_service
    from services.generation_service import generation_service
    from middleware.auth import verify_api_key
    from middleware.rate_limiter import check_rate_limit
    from utils.logger import log_info, log_error

router = APIRouter(tags=["chat"])

# Define request/response models (unchanged)
class ChatRequest(BaseModel):
    question: str
    context: Optional[str] = None
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    metadata: Dict[str, Any]

@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key), Depends(check_rate_limit)])
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint that processes user questions through the RAG pipeline
    """
    try:
        log_info("Processing chat request", extra={
            "question_length": len(request.question),
            "has_context": request.context is not None,
            "session_id": request.session_id
        })

        # Step 1: Retrieve relevant chunks
        retrieved_chunks = retrieval_service.retrieve_relevant_chunks(
            query=request.question,
            top_k=config.TOP_K
        )

        if not retrieved_chunks:
            log_info("No relevant chunks found", extra={"question": request.question[:100]})
            return ChatResponse(
                answer="I couldn't find relevant information in the documentation.",
                sources=[],
                metadata={"retrieval_success": False, "chunks_found": 0}
            )

        # Step 2: Generate response
        generation_result = generation_service.generate_response(
            query=request.question,
            context_chunks=retrieved_chunks,
            selected_context=request.context
        )

        # Step 3: Format response
        response = ChatResponse(
            answer=generation_result["answer"],
            sources=generation_result["sources"],
            metadata={
                **generation_result["metadata"],
                "retrieval_success": True,
                "chunks_found": len(retrieved_chunks),
                "chunks_used": min(len(retrieved_chunks), 3),
                "processing_timestamp": __import__('datetime').datetime.now().isoformat()
            }
        )

        log_info("Chat response generated successfully", extra={
            "answer_length": len(generation_result["answer"]),
            "sources_count": len(generation_result["sources"])
        })

        return response

    except HTTPException:
        raise  # Let auth/rate-limit errors pass through

    except Exception as e:
        log_error(f"Error in chat endpoint: {str(e)}", extra={
            "question": request.question[:100] if request.question else "empty",
            "error_type": type(e).__name__,
            "traceback": "".join(__import__('traceback').format_exception(type(e), e, e.__traceback__)) if hasattr(e, '__traceback__') else "no traceback"
        })

        return ChatResponse(
            answer="Sorry, I encountered an internal error. Please try again later.",
            sources=[],
            metadata={"retrieval_success": False, "processing_error": True}
        )

# Health endpoint (simplified)
@router.get("/health")
async def chat_health():
    try:
        return {
            "status": "healthy",
            "service": "chat-api",
            "dependencies": {
                "cohere": "configured" if config.COHERE_API_KEY else "missing",
                "retrieval_service": "ready",
                "generation_service": "ready"
            }
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e)
        }

@router.post("/validate-question")
async def validate_question(question: str = None):
    if not question or len(question.strip()) == 0:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if len(question) > 1000:
        raise HTTPException(status_code=400, detail="Question is too long")

    return {"valid": True, "length": len(question)}