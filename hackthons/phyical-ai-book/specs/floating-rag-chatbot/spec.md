# Floating RAG Chatbot for Physical AI Book - Specification

## Project Overview

**Project**: Floating RAG Chatbot – Physical AI Book
**Book URL**: https://physical-ai-book-lilac.vercel.app/

This project integrates a Retrieval-Augmented Generation (RAG) chatbot into the Physical AI digital book website, allowing users to ask questions about the book content and receive accurate, contextual responses based on the book's documentation.

## Scope Definition

### In Scope
- Integration of a floating chatbot widget into the existing Docusaurus site
- Implementation of RAG pipeline using local markdown files from /docs folder as primary source
- Support for selected text as additional context in queries
- Backend API for processing queries and retrieving relevant content
- Vector storage and retrieval using Qdrant Cloud
- Frontend component with expandable chat interface
- Session management for future extensions
- Privacy-compliant operation with no logging of user queries
- Responsive design supporting mobile devices

### Out of Scope
- Real-time content fetching per query (v1 restriction)
- Live content synchronization without re-indexing
- Third-party chat service integration
- Client-side exposure of API keys
- Advanced analytics or user behavior tracking
- Multi-language support (v1)
- Voice interaction capabilities (v1)

## Functional Requirements

### FR-001: Content Indexing
The system shall index content from:
- Primary: Local markdown files from /docs folder
- Fallback: HTML pages via sitemap.xml (only when markdown is unavailable)

### FR-002: Query Processing
The system shall accept user queries and:
- Process the question with optional context from selected text
- Retrieve relevant content from vector database
- Generate contextual responses based on retrieved content
- Return sources for transparency

### FR-003: Chat Interface
The system shall provide:
- Floating FAB (bottom-right) chat widget
- Expandable window with conversation history
- Auto-population of selected text as context
- Responsive design for mobile devices (max ~90% width on small screens)

### FR-004: API Endpoints
The system shall expose:
- POST /api/chat endpoint for processing queries
- Proper error handling and response formatting

## Non-Functional Requirements

### NFR-001: Performance
- Chat window open/close: <300ms
- First answer (including network): target <5s
- Vector search and retrieval: <2s

### NFR-002: Security
- Never expose Cohere/Qdrant keys to frontend
- No logging of question/context/answer bodies
- HTTPS only connections
- CORS restricted to project domain(s)

### NFR-003: Availability
- Support serverless deployment model
- Handle concurrent users appropriately
- Graceful degradation when services are unavailable

### NFR-004: Scalability
- Support for growing content base
- Efficient vector storage and retrieval
- Minimal resource consumption

## Technical Architecture

### Backend Stack
- Node.js/Express API server
- Cohere for embeddings and generation
- Qdrant Cloud for vector storage
- Neon Serverless Postgres for metadata (optional for MVP)

### Frontend Integration
- React component for chat widget
- Global mount via src/theme/Root.js (Docusaurus swizzle)
- CSS-in-JS for dynamic styling
- Cross-browser compatibility

### Data Flow
1. Content indexing: markdown → embeddings → Qdrant vector storage
2. Query processing: user input → embeddings → similarity search → context retrieval → answer generation
3. Response delivery: generated answer + sources to frontend

## API Contract

### Main Endpoint: POST /api/chat
**Request Body:**
```json
{
  "question": "string",
  "context": "string | null",          // selected text when available
  "session_id": "string | null"        // future extension
}
```

**Response:**
```json
{
  "answer": "string",
  "sources": ["string"] | null,        // relative paths or URLs
  "metadata": {} | null
}
```

### Authentication
- Custom header: X-API-Key
- Value: Strong random secret (32+ chars)

## Infrastructure Configuration

### Cohere Services
- **COHERE_API_KEY**: dq9WhKNIrHOflZRUcNeAsLDuAylIOJ2IKSHMeu1j
- **Embedding Model**: embed-english-v3.0 (1024 dim)
- **Generation Model**: command-r (fallback: command-r-plus)

### Qdrant Cloud Configuration
- **QDRANT_URL**: https://83733787-c66e-4911-bb04-8eca65234e04.europe-west3-0.gcp.cloud.qdrant.io
- **QDRANT_API_KEY**: yJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.RbXgYj61AHM0q_2oFP_0-3OzqavIeO9Rgm7K2NI3m2c
- **Collection Name**: physical-ai-book-v1
- **Vector Size**: 1024
- **Distance Metric**: Cosine

### Database (Optional for MVP)
- **DATABASE_URL**: postgresql://neondb_owner:npg_0lHqDcJpG8vj@ep-blue-darkness-a4j1pd7e-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require

## Frontend Component Specifications

### Location
- **Component Path**: src/components/ChatWidget/
- **Global Mount**: src/theme/Root.js (via swizzle --wrap)

### Component Structure
- **ChatWidget**: Main component with toggle functionality
- **ChatWindow**: Expandable container with conversation history
- **InputArea**: Text input with optional context display
- **MessageList**: Scrollable area for conversation history

### Appearance & Behavior
- **Position**: Floating FAB bottom-right corner (20px from edges)
- **Size**: Default icon size 60x60px, expandable to 400x500px window
- **Z-index**: 1000–1099 range (above other UI elements)
- **Mobile Support**: Responsive design (max ~90% width on small screens)
- **Animation**: Smooth open/close transitions (<300ms)
- **Selected Text**: Auto-populate input field when text is selected
- **Focus Management**: Proper keyboard navigation and accessibility

### Interaction Patterns
- **Click FAB**: Expand chat window
- **Esc key**: Minimize chat window
- **Enter key**: Submit question (Shift+Enter for new line)
- **Text selection**: Capture selected text and offer to include as context
- **Scrolling**: Auto-scroll to latest message in conversation

### Styling Requirements
- **Theme**: Match Docusaurus site theme and colors
- **Typography**: Consistent with site font family and sizes
- **Accessibility**: WCAG 2.1 AA compliance (contrast ratios, focus indicators)
- **Icons**: Use consistent icon set (preferably Material Icons or similar)

### State Management
- **Open/Closed**: Track widget state
- **Loading**: Show loading indicators during API calls
- **Error States**: Handle and display API errors gracefully
- **Conversation History**: Maintain current session in component state

## Security & Privacy Requirements

### SR-001: Key Management
- Store all infrastructure credentials in .env file
- Never commit credentials to code/repository
- Never expose backend API keys to frontend
- Implement secure environment variable handling
- Use secrets management best practices

### SR-002: Data Privacy
- Do not log question/context/answer content
- Implement zero-knowledge architecture
- Ensure HTTPS-only communications
- Restrict CORS to project domain(s)
- No persistent storage of user queries or selections
- Clear temporary data after session ends

### SR-003: Access Control
- Implement API key authentication
- Rate limiting to prevent abuse (max 10 requests/minute per IP)
- Input validation and sanitization
- Authentication header validation (X-API-Key)
- IP-based request limiting

### SR-004: Client-Side Security
- Validate all API responses before rendering
- Prevent XSS through proper output encoding
- Sanitize any user-generated content in responses
- Implement Content Security Policy (CSP) headers

### SR-005: Data Transmission Security
- All API communications via HTTPS only
- Encrypt sensitive data in transit
- Validate SSL certificates for external services
- Implement proper TLS configuration

### SR-006: Compliance Requirements
- Adhere to privacy regulations (GDPR, CCPA where applicable)
- Provide transparency about data handling
- Implement data deletion capabilities if needed
- Maintain audit logs of system access (not content)

## Content Sourcing & Freshness

### CS-001: Source Hierarchy
1. Primary: Local markdown files from /docs folder (highest quality)
2. Fallback: HTML pages via sitemap.xml (when markdown unavailable)
3. Other methods: Out of scope for v1

### CS-002: Update Mechanism
- MVP: Content is controlled snapshot taken at indexing time
- Update target: Manual run or CI/CD triggered re-indexing
- Acceptable delay: Days to weeks (manual or semi-automated re-index acceptable)
- Real-time fetching: Explicitly out of scope for v1

## Acceptance Criteria

### AC-001: Basic Functionality
- [ ] Chat widget appears in bottom-right corner
- [ ] Widget expands to show chat interface
- [ ] User can submit questions and receive relevant answers
- [ ] Sources are provided for generated answers

### AC-002: Context Handling
- [ ] Selected text is captured and sent as context
- [ ] Context improves answer relevance
- [ ] Answers account for both question and context

### AC-003: Performance
- [ ] Chat window opens/closes in <300ms
- [ ] First answer delivered in <5s
- [ ] System handles concurrent users appropriately

### AC-004: Security
- [ ] API keys are not exposed in frontend code
- [ ] No user query data is logged
- [ ] All connections use HTTPS
- [ ] CORS is properly restricted

### AC-005: Responsiveness
- [ ] Chat widget works on mobile devices
- [ ] Interface adapts to different screen sizes
- [ ] Touch interactions work properly on mobile

## Risk Assessment

### High-Risk Areas
- API key exposure in client-side code
- Performance degradation with large content sets
- Accuracy of retrieved context affecting answer quality
- Privacy compliance with user data handling

### Mitigation Strategies
- Strict separation of frontend and backend concerns
- Proper environment variable handling
- Comprehensive testing with realistic content volumes
- Regular security reviews and audits

## Success Metrics

- User engagement with chat feature
- Answer accuracy and relevance
- Response time performance
- Error rate and system reliability
- User satisfaction with answers