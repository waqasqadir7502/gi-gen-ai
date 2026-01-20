from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from ..services.retrieval_service import retrieval_service
from ..services.generation_service import generation_service
from ..middleware.auth import verify_api_key
from ..middleware.rate_limiter import check_rate_limit
from ..utils.logger import log_info, log_error
from ..config import config
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

router = APIRouter(prefix="/api", tags=["chat"])

# Safe, global Qdrant client (initialized lazily, no crash on import/startup)
_qdrant_client = None

def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        try:
            _qdrant_client = QdrantClient(
                url=config.QDRANT_URL,
                api_key=config.QDRANT_API_KEY,
            )
            collection_name = "physical-ai-book-v1"
            
            if not _qdrant_client.has_collection(collection_name):
                _qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config={"size": 1024, "distance": "Cosine"}
                )
                log_info(f"Created Qdrant collection: {collection_name}")
            else:
                log_info(f"Qdrant collection {collection_name} already exists - using existing")
                
        except UnexpectedResponse as e:
            if e.status_code == 409:
                log_info("Collection already exists (safe)")
            else:
                log_error(f"Qdrant connection warning: {e}")
        except Exception as e:
            log_error(f"Qdrant init failed (continuing): {str(e)}")
    
    if _qdrant_client is None:
        raise HTTPException(500, "Qdrant client not initialized")
    
    return _qdrant_client

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

        # Get safe Qdrant client
        qdrant_client = get_qdrant_client()

        # Step 1: Retrieve relevant chunks
        retrieved_chunks = retrieval_service.retrieve_and_rerank(
            query=request.question,
            context=request.context,
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
        generation_result = generation_service.generate_summarized_response(
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
                "chunks_used": len(retrieved_chunks[:3]),
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
            "question": request.question[:100],
            "error_type": type(e).__name__,
            "traceback": "".join(__import__('traceback').format_exception(type(e), e, e.__traceback__))
        })

        return ChatResponse(
            answer="Sorry, I encountered an internal error. Please try again later.",
            sources=[],
            metadata={"retrieval_success": False, "processing_error": True}
        )

# Health endpoint (enhanced with Qdrant check)
@router.get("/health")
async def chat_health():
    try:
        client = get_qdrant_client()  # Safe init
        return {
            "status": "healthy",
            "service": "chat-api",
            "dependencies": {
                "cohere": "configured" if config.COHERE_API_KEY else "missing",
                "qdrant": "connected",
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