---
id: 3
title: create-rag-chatbot-plan
stage: plan
date: 2026-01-13
surface: agent
model: claude-sonnet-4-5-20250929
feature: floating-rag-chatbot
branch: main
user: user
command: sp.plan
labels: [planning, architecture, roadmap]
links:
  spec: specs/floating-rag-chatbot/spec.md
  ticket: null
  adr: null
  pr: null
files:
 - specs/floating-rag-chatbot/plan.md
tests:
 - none
---

## Prompt

Project: Floating RAG Chatbot – Physical AI Book
Timeline target: 4.5–7 weeks part-time effort (~10–15h/week)

Phase 1: Setup & Environment                              (Week 1)
- Separate backend repo/folder
- FastAPI + Spec-Kit Plus structure
- .env with all credentials (Cohere, Qdrant, Neon, own API key)
- Health endpoint + basic Cohere/Qdrant connection test

Phase 2: Content Preparation & First Ingestion            (Week 1–2)
- Use local /docs markdown files as primary source
- Build chunking + metadata extraction logic
- Ingestion script: chunk → embed → Qdrant upsert
- Run initial indexing → verify collection has meaningful points

Phase 3: Core RAG Pipeline Implementation                 (Week 2–3.5)
- Retrieval: Cohere embed → Qdrant search (top_k=6–10)
- Prompt engineering: strict grounding instruction
- Generation: command-r via Cohere
- /api/chat endpoint with selected context support
- Basic error handling & response formatting

Phase 4: Backend Security & Deployment Prep               (Week 3.5–4)
- API key authentication (X-API-Key header)
- CORS configuration
- Deploy to free platform (Vercel functions, Railway, etc.)

Phase 5: Frontend Chat Widget Integration                 (Week 4–5)
- Create ChatWidget component (FAB + window)
- Selected text capture → input population
- Fetch to backend with auth header
- Root wrapper injection
- Mobile responsiveness & basic styling

Phase 6: Testing & Accuracy Validation                    (Week 5–6)
- Unit + integration tests (backend pipeline)
- 50+ question test set from book content
- Target: ≥94–96% factual accuracy on indexed material
- Cross-device/browser testing

Phase 7: Content Update Procedure & Documentation         (Week 6–6.5)
- Document manual re-index process
- Provide simple GitHub Action example (trigger on push)
- Full README + Swagger docs

Phase 8: Polish, Beta Testing & Handover                  (Week 6.5–7)
- UX improvements (loading, copy button, etc.)
- 5+ beta testers feedback
- Final privacy/security checklist

Critical Path: Phase 1 → 2 → 3 → 5 → 6 → 7
Recommended starting point: local markdown ingestion (cleanest, most reliable path)

## Response snapshot

Created comprehensive implementation plan for the Floating RAG Chatbot integration project with detailed specifications for all 8 phases:

Phase 1: Setup & Environment (Week 1) - Repository structure, environment configuration, and service connectivity tests
Phase 2: Content Preparation & First Ingestion (Week 1-2) - Enhanced with detailed markdown parsing, chunking algorithms, and ingestion pipeline specifications
Phase 3: Core RAG Pipeline Implementation (Week 2-3.5) - Enhanced with detailed retrieval, prompt engineering, and generation pipeline specifications
Phase 4: Backend Security & Deployment Prep (Week 3.5-4) - Authentication and deployment preparation
Phase 5: Frontend Chat Widget Integration (Week 4-5) - Enhanced with detailed component development, integration, and UI/UX specifications
Phase 6: Testing & Accuracy Validation (Week 5-6) - Comprehensive testing requirements
Phase 7: Content Update Procedure & Documentation (Week 6-6.5) - Documentation and update procedures
Phase 8: Polish, Beta Testing & Handover (Week 6.5-7) - Final improvements and handover

The plan includes detailed technical specifications, success criteria, risk management, and resource requirements. All phases align with the specified timeline and technical requirements.

## Outcome

- ✅ Impact: Created detailed implementation plan with technical specifications
- 🧪 Tests: none
- 📁 Files: Created specs/floating-rag-chatbot/plan.md with complete project plan
- 🔁 Next prompts: Proceed with task breakdown and implementation
- 🧠 Reflection: Plan provides clear roadmap with detailed technical requirements for each phase

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): PASS - all phases detailed comprehensively
- Prompt variant (if applicable): none
- Next experiment (smallest change to try): Begin with Phase 1 implementation