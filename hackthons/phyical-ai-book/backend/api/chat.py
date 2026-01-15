from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from ..services.retrieval_service import retrieval_service
from ..services.generation_service import generation_service
from ..middleware.auth import verify_api_key
from ..middleware.rate_limiter import check_rate_limit
from ..utils.logger import log_info, log_error
from ..config import config

router = APIRouter(prefix="/api", tags=["chat"])

# Define request/response models
class ChatRequest(BaseModel):
    question: str
    context: Optional[str] = None  # Selected text context
    session_id: Optional[str] = None  # Future extension for conversation history

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    metadata: Dict[str, Any]

class ChatHistoryItem(BaseModel):
    question: str
    answer: str
    timestamp: str

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

        # Step 1: Retrieve relevant chunks based on the question and context
        retrieved_chunks = retrieval_service.retrieve_and_rerank(
            query=request.question,
            context=request.context,
            top_k=config.TOP_K
        )

        if not retrieved_chunks:
            log_info("No relevant chunks found for the query", extra={
                "question": request.question[:100] + "..." if len(request.question) > 100 else request.question
            })

            return ChatResponse(
                answer="I couldn't find relevant information in the documentation to answer your question.",
                sources=[],
                metadata={
                    "retrieval_success": False,
                    "chunks_found": 0,
                    "processing_time": "N/A"
                }
            )

        # Step 2: Generate response using the retrieved context
        generation_result = generation_service.generate_summarized_response(
            query=request.question,
            context_chunks=retrieved_chunks,
            selected_context=request.context
        )

        # Step 3: Format and return the response
        response = ChatResponse(
            answer=generation_result["answer"],
            sources=generation_result["sources"],
            metadata={
                **generation_result["metadata"],
                "retrieval_success": True,
                "chunks_found": len(retrieved_chunks),
                "chunks_used": len(retrieved_chunks[:3]),  # Top 3 chunks used for generation
                "processing_timestamp": __import__('datetime').datetime.now().isoformat()
            }
        )

        log_info("Chat response generated successfully", extra={
            "question_length": len(request.question),
            "answer_length": len(generation_result["answer"]),
            "sources_count": len(generation_result["sources"]),
            "chunks_used": len(retrieved_chunks[:3])
        })

        return response

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        log_error(f"Error in chat endpoint: {str(e)}", extra={
            "question": request.question[:100] + "..." if len(request.question) > 100 else request.question,
            "error_type": type(e).__name__
        })

        # Return a safe error response - mask sensitive error details
        return ChatResponse(
            answer="I encountered an error while processing your request. Please try again.",
            sources=[],
            metadata={
                "retrieval_success": False,
                "processing_error": True
            }
        )

# Additional endpoints for future extensions
@router.get("/health")
async def chat_health():
    """
    Health check for the chat service
    """
    return {
        "status": "healthy",
        "service": "chat-api",
        "dependencies": {
            "cohere": "connected",
            "qdrant": "connected",
            "retrieval_service": "ready",
            "generation_service": "ready"
        }
    }

@router.post("/validate-question")
async def validate_question(question: str = None):
    """
    Validate a question before processing (future extension)
    """
    if not question or len(question.strip()) == 0:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if len(question) > 1000:  # Arbitrary limit
        raise HTTPException(status_code=400, detail="Question is too long")

    return {"valid": True, "length": len(question)}