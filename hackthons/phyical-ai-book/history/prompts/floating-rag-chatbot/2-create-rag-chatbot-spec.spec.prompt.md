---
id: 2
title: create-rag-chatbot-spec
stage: spec
date: 2026-01-13
surface: agent
model: claude-sonnet-4-5-20250929
feature: floating-rag-chatbot
branch: main
user: user
command: sp.specify
labels: [specification, rag, chatbot, docusaurus]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - specs/floating-rag-chatbot/spec.md
tests:
 - none
---

## Prompt

Project: Floating RAG Chatbot – Physical AI Book
Book URL: https://physical-ai-book-lilac.vercel.app/

Content Sourcing & Freshness Rules:
Primary source (strongly preferred): local markdown files from /docs folder
Fallback source (only when necessary): HTML pages via sitemap.xml
Real-time fetching per query: explicitly NOT allowed in v1
Update mechanism target: manual run or CI/CD triggered re-indexing

Infrastructure Credentials (MUST be stored in .env – never in code/repo):

Cohere:
• COHERE_API_KEY = dq9WhKNIrHOflZRUcNeAsLDuAylIOJ2IKSHMeu1j
• Embedding model: embed-english-v3.0 (1024 dim)
• Generation model: command-r (fallback: command-r-plus)

Qdrant Cloud Free Tier:
• QDRANT_URL          = https://83733787-c66e-4911-bb04-8eca65234e04.europe-west3-0.gcp.cloud.qdrant.io
• QDRANT_API_KEY      = yJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.RbXgYj61AHM0q_2oFP_0-3OzqavIeO9Rgm7K2NI3m2c
• Collection name     : physical-ai-book-v1
• Vector size         : 1024
• Distance            : Cosine

Neon Serverless Postgres (metadata – optional for MVP):
• DATABASE_URL = postgresql://neondb_owner:npg_0lHqDcJpG8vj@ep-blue-darkness-a4j1pd7e-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require

Backend → Frontend Authentication:
• Custom header: X-API-Key
• Value: generate your own strong random secret (32+ chars)

API Contract (main endpoint):
POST /api/chat
Request body:
{
  "question": string,
  "context": string | null,          // selected text when available
  "session_id": string | null        // future extension
}
Response:
{
  "answer": string,
  "sources": [string] | null,        // relative paths or URLs
  "metadata": object | null
}

Frontend Integration Rules:
• Location: src/components/ChatWidget/
• Global mount: src/theme/Root.js (via swizzle --wrap)
• Appearance: floating FAB bottom-right + expandable window
• Selected text: auto-populate input field on selection
• Mobile: responsive (max ~90% width on small screens)
• Z-index: 1000–1099 range

Security & Privacy:
• NEVER expose Cohere / Qdrant keys to frontend
• No logging of question / context / answer bodies
• HTTPS only
• CORS restricted to project domain(s)

Performance Targets:
• Chat window open/close: <300ms
• First answer (incl. network): target <5s

This specification is the single source of truth for all implementation decisions.

## Response snapshot

Created comprehensive specification document for the Floating RAG Chatbot integration project. The spec includes:

- Project overview and scope definition (in/out of scope)
- Functional requirements (content indexing, query processing, chat interface, API endpoints)
- Non-functional requirements (performance, security, availability, scalability)
- Technical architecture (backend stack, frontend integration, data flow)
- Detailed API contract for POST /api/chat endpoint
- Infrastructure configuration details for Cohere, Qdrant, and Neon Postgres
- Frontend component specifications with structure, appearance, behavior, and state management
- Security and privacy requirements with key management, data privacy, access control, and compliance
- Content sourcing rules with hierarchy and update mechanisms
- Acceptance criteria with checklists for functionality, context handling, performance, security, and responsiveness
- Risk assessment and success metrics

The specification aligns with all provided requirements and serves as the single source of truth for implementation.

## Outcome

- ✅ Impact: Created comprehensive specification document for RAG chatbot integration
- 🧪 Tests: none
- 📁 Files: Created specs/floating-rag-chatbot/spec.md with complete requirements
- 🔁 Next prompts: Proceed with planning and implementation phases
- 🧠 Reflection: Specification covers all required aspects with clear acceptance criteria

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): PASS - all requirements documented
- Prompt variant (if applicable): none
- Next experiment (smallest change to try): Begin with project planning phase