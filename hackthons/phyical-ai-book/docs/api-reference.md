# API Reference for Physical AI Book RAG Chatbot

This document provides reference information for the RAG Chatbot API endpoints.

## Base URL
The API is served at `/api` on the backend server.

## Authentication
All API endpoints require authentication using an API key in the `X-API-Key` header.

Example:
```
X-API-Key: your_backend_api_key
```

## Endpoints

### POST /api/chat
Process a user question and return a response based on the indexed documentation.

#### Request Body
```json
{
  "question": "string",
  "context": "string | null",
  "session_id": "string | null"
}
```

**Fields:**
- `question` (required): The user's question
- `context` (optional): Additional context (e.g., selected text) to include in the query
- `session_id` (optional): Session identifier for future conversation history features

#### Response
```json
{
  "answer": "string",
  "sources": ["string"],
  "metadata": {
    "model_used": "string",
    "context_chunks_used": "number",
    "generation_timestamp": "string",
    "retrieval_success": "boolean",
    "chunks_found": "number",
    "chunks_used": "number",
    "processing_timestamp": "string"
  }
}
```

**Response Fields:**
- `answer`: The AI-generated response to the question
- `sources`: Array of source documents referenced in the answer
- `metadata`: Additional information about the generation process

#### Example Request
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_backend_api_key" \
  -d '{
    "question": "What is Physical AI?",
    "context": "Selected text context if available"
  }'
```

#### Example Response
```json
{
  "answer": "Physical AI refers to artificial intelligence systems that interact with the physical world...",
  "sources": ["docs/chapter-template.md", "README.md"],
  "metadata": {
    "model_used": "command-r",
    "context_chunks_used": 8,
    "generation_timestamp": "2026-01-13T15:30:00.123456",
    "retrieval_success": true,
    "chunks_found": 8,
    "chunks_used": 3,
    "processing_timestamp": "2026-01-13T15:30:00.123456"
  }
}
```

### GET /health
Check the health status of the API service.

#### Response
```json
{
  "status": "healthy",
  "details": {
    "cohere_connection": "ok",
    "qdrant_connection": "ok",
    "collection_name": "physical-ai-book-v1",
    "vector_size": 1024
  }
}
```

### GET /api/health
Detailed health check for the chat API service.

#### Response
```json
{
  "status": "healthy",
  "service": "chat-api",
  "dependencies": {
    "cohere": "connected",
    "qdrant": "connected",
    "retrieval_service": "ready",
    "generation_service": "ready"
  }
}
```

## Error Responses

### Rate Limit Exceeded (HTTP 429)
```json
{
  "error": "Rate limit exceeded",
  "message": "Too many requests. Limit is 10 per minute.",
  "retry_after_seconds": 60
}
```

### Unauthorized (HTTP 401)
```json
{
  "detail": "Invalid API Key"
}
```

### Other Errors (HTTP 400, 500, etc.)
```json
{
  "detail": "Error message"
}
```

## Rate Limits
- 10 requests per minute per IP address
- Requests exceeding the limit will return a 429 status code
- Retry after the time specified in the `Retry-After` header

## Security Headers
All responses include security headers:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security: max-age=31536000; includeSubDomains`
- `Referrer-Policy: no-referrer-when-downgrade`
- `Permissions-Policy: geolocation=(), microphone=(), camera=()`