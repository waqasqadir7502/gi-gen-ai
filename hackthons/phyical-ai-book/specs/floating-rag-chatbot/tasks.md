---
description: "Task list for Floating RAG Chatbot integration project"
---

# Tasks: Floating RAG Chatbot for Physical AI Book

**Input**: Design documents from `/specs/floating-rag-chatbot/`
**Prerequisites**: plan.md (required), spec.md (required for user stories)

**Tests**: Tests are included based on the accuracy validation requirements in the specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Backend**: `backend/`, `api/`, `services/`, `models/`, `utils/`
- **Frontend**: `src/components/`, `src/theme/`, `static/`
- **Configuration**: `.env`, `package.json`, `requirements.txt`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Create backend project structure in backend/ directory
- [x] T002 [P] Initialize FastAPI application in backend/main.py
- [x] T003 [P] Create .env file with all required credentials
- [x] T004 Install and configure required dependencies in requirements.txt
- [x] T005 Create basic configuration files for deployment

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T006 Setup environment variable loading in backend/config.py
- [x] T007 [P] Configure Cohere API client in backend/clients/cohere_client.py
- [x] T008 [P] Configure Qdrant client in backend/clients/qdrant_client.py
- [x] T009 Create health endpoint in backend/api/health.py
- [x] T010 Setup basic logging configuration in backend/utils/logger.py
- [x] T011 Configure CORS middleware for frontend integration
- [x] T012 Setup API key authentication middleware in backend/middleware/auth.py

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Content Indexing (Priority: P1) 🎯 MVP

**Goal**: Implement content indexing pipeline using local markdown files as primary source

**Independent Test**: Verify that markdown files from /docs folder are successfully parsed, chunked, and stored in the Qdrant vector database

### Implementation for User Story 1

- [x] T013 [P] Create markdown parser utility in backend/utils/markdown_parser.py
- [x] T014 [P] Implement content extraction logic from /docs folder
- [x] T015 Create chunking algorithm in backend/utils/chunking.py with 512-1024 token target
- [x] T016 [P] Implement metadata extraction utility in backend/utils/metadata_extractor.py
- [x] T017 Create embedding pipeline using Cohere embed-english-v3.0 in backend/pipelines/embedding_pipeline.py
- [x] T018 Implement Qdrant upsert logic in backend/pipelines/indexing_pipeline.py
- [x] T019 Create ingestion script in backend/scripts/ingest_docs.py
- [x] T020 [P] Add duplicate detection and handling to ingestion pipeline
- [x] T021 [P] Implement batch processing for efficient indexing
- [x] T022 Test indexing pipeline with sample content
- [x] T023 Verify collection physical-ai-book-v1 has properly indexed vectors

**Checkpoint**: At this point, content indexing should be fully functional and testable independently

---

## Phase 4: User Story 2 - RAG Pipeline & API (Priority: P2)

**Goal**: Implement retrieval and generation pipeline with the main chat API endpoint

**Independent Test**: Verify that user questions are processed through the RAG pipeline to return accurate, grounded responses with proper citations

### Implementation for User Story 2

- [x] T024 [P] Create retrieval service in backend/services/retrieval_service.py
- [x] T025 [P] Implement Cohere embedding function for queries
- [x] T026 Create Qdrant search with top_k=6-10 in retrieval service
- [x] T027 [P] Implement prompt engineering with strict grounding instructions in backend/prompts/chat_prompt.py
- [x] T028 Create generation service using Cohere command-r in backend/services/generation_service.py
- [x] T029 [P] Implement fallback to command-r-plus in generation service
- [x] T030 Create POST /api/chat endpoint in backend/api/chat.py
- [x] T031 [P] Add selected text context integration to the chat endpoint
- [x] T032 [P] Implement response formatting with source attribution
- [x] T033 Add error handling and validation to chat endpoint
- [x] T034 Test end-to-end RAG pipeline with sample queries

**Checkpoint**: At this point, RAG pipeline should be fully functional and testable independently

---

## Phase 5: User Story 3 - Frontend Chat Widget (Priority: P3)

**Goal**: Create floating chat widget component that integrates with the backend API

**Independent Test**: Verify that the chat widget appears in the bottom-right corner, expands to show interface, and successfully communicates with the backend API

### Implementation for User Story 3

- [x] T035 [P] Create ChatWidget React component in src/components/ChatWidget/ChatWidget.jsx
- [x] T036 [P] Implement FAB (Floating Action Button) with proper styling
- [x] T037 Create expandable chat window component in src/components/ChatWidget/ChatWindow.jsx
- [x] T038 [P] Implement MessageList component with conversation history display
- [x] T039 Create InputArea component with multi-line support in src/components/ChatWidget/InputArea.jsx
- [x] T040 [P] Add typing indicators and status icons to chat components
- [x] T041 Implement API call to /api/chat endpoint with proper authentication
- [x] T042 [P] Add loading states and error handling to UI
- [x] T043 Implement smooth CSS transitions for open/close (<300ms)
- [x] T044 [P] Add accessibility features (ARIA labels, keyboard navigation)
- [x] T045 Create responsive design for mobile devices (max 90% width)
- [x] T046 [P] Add copy-to-clipboard functionality for responses
- [x] T047 Test chat widget functionality with backend API

**Checkpoint**: At this point, frontend chat widget should be fully functional and testable independently

---

## Phase 6: User Story 4 - Selected Text Integration (Priority: P4)

**Goal**: Implement selected text capture functionality that auto-populates the chat input with context

**Independent Test**: Verify that when text is selected on the page, it can be captured and sent as context to the chat API

### Implementation for User Story 4

- [x] T048 [P] Implement text selection detection using window.getSelection() in src/components/ChatWidget/textSelection.js
- [x] T049 Create context preview and editing capability in InputArea component
- [x] T050 [P] Add keyboard shortcut for context inclusion
- [x] T051 Implement auto-population of input field with selected text
- [x] T052 [P] Handle edge cases for selection (multiple elements, special content)
- [x] T053 Add visual indicator for context inclusion
- [x] T054 Test selected text functionality with various content types

**Checkpoint**: At this point, selected text integration should be fully functional and testable independently

---

## Phase 7: User Story 5 - Security & Performance (Priority: P5)

**Goal**: Implement security measures and ensure performance targets are met

**Independent Test**: Verify that API keys are not exposed, no user data is logged, and performance targets (<300ms open/close, <5s responses) are met

### Implementation for User Story 5

- [x] T055 [P] Implement rate limiting (10 requests/minute per IP) in backend/middleware/rate_limiter.py
- [x] T056 [P] Add request/response logging (excluding sensitive content) in logger utility
- [x] T057 Create security headers configuration in backend/main.py
- [x] T058 [P] Implement input validation and sanitization for API endpoints
- [x] T059 Add error masking to prevent sensitive information leakage
- [x] T060 [P] Implement content moderation checks in generation service
- [x] T061 Add retry logic for failed API requests in frontend
- [x] T062 [P] Add request timeout handling (default 30s) in frontend
- [x] T063 Test performance targets (open/close <300ms, response <5s)
- [x] T064 [P] Validate CORS configuration restricts to project domain

**Checkpoint**: At this point, security and performance measures should be fully functional and testable independently

---

## Phase 8: Testing & Validation

**Goal**: Conduct comprehensive testing to validate accuracy and performance requirements

### Tests for Accuracy Validation

- [x] T065 Create 50+ question test set from book content in test_data/questions.json
- [x] T066 [P] Implement accuracy validation script in backend/tests/accuracy_test.py
- [x] T067 Run factual accuracy evaluation on indexed material
- [x] T068 [P] Measure hallucination rate against test dataset
- [x] T069 Validate that accuracy achieves ≥94-96% target
- [x] T070 [P] Test cross-platform compatibility (Chrome, Firefox, Safari, Edge)

### Tests for Performance

- [x] T071 [P] Test response time performance against <5s target
- [x] T072 Validate chat window open/close performance (<300ms)
- [x] T073 [P] Test concurrent user scenarios
- [x] T074 Run vector search and retrieval performance test (<2s)
- [x] T075 [P] Validate resource usage under load

**Checkpoint**: At this point, all testing and validation should be complete

---

## Phase 9: Root Wrapper Integration

**Goal**: Integrate the chat widget globally into the Docusaurus site

- [x] T076 Create Root wrapper component in src/theme/Root.js
- [x] T077 [P] Inject ChatWidget globally via Docusaurus swizzle
- [x] T078 Ensure proper z-index layering (1000-1099) in CSS
- [x] T079 [P] Test integration with existing site components and layouts
- [x] T080 Validate compatibility with Docusaurus plugins and themes
- [x] T081 [P] Implement lazy loading to minimize initial bundle impact
- [x] T082 Test complete integration with Docusaurus site

---

## Phase 10: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [x] T083 [P] Add loading indicators and progress feedback to UI
- [x] T084 Improve error messaging and handling throughout application
- [x] T085 [P] Add conversation history persistence capability
- [x] T086 Create documentation for manual re-index process
- [x] T087 [P] Create GitHub Actions workflow example for automated re-indexing
- [x] T088 Update README with installation and setup instructions
- [x] T089 [P] Create API documentation with examples
- [x] T090 Document security and privacy measures implemented
- [x] T091 [P] Add deployment instructions for Vercel/Railway
- [x] T092 Run final security checklist validation
- [x] T093 [P] Complete privacy compliance verification

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3 → P4 → P5)
- **Testing & Validation (Phase 8)**: Depends on User Stories 1-5 being complete
- **Root Wrapper Integration (Phase 9)**: Depends on User Story 3 being complete
- **Polish (Phase 10)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Depends on User Story 1 completion (needs indexed content)
- **User Story 3 (P3)**: Depends on User Story 2 completion (needs working API)
- **User Story 4 (P4)**: Depends on User Story 3 completion (needs working UI)
- **User Story 5 (P5)**: Can start after Foundational (Phase 2) - May enhance other stories

### Within Each User Story

- Basic functionality before advanced features
- Core components before integration
- Individual components before combined functionality
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, User Stories 1, 2, and 5 can start in parallel
- User Stories 3 and 4 must wait for their dependencies but can be developed in parallel with their prerequisites
- Different components within stories marked [P] can run in parallel

---

## Parallel Example: User Story 2 (RAG Pipeline)

```bash
# Launch all components for RAG pipeline together:
Task: "Create retrieval service in backend/services/retrieval_service.py"
Task: "Implement Cohere embedding function for queries"
Task: "Create generation service using Cohere command-r in backend/services/generation_service.py"
Task: "Create POST /api/chat endpoint in backend/api/chat.py"
```

---

## Implementation Strategy

### MVP First (User Stories 1-2 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 (Content Indexing)
4. Complete Phase 4: User Story 2 (RAG Pipeline & API)
5. **STOP and VALIDATE**: Test basic chat functionality independently
6. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test content indexing independently → Deploy/Demo
3. Add User Story 2 → Test RAG pipeline independently → Deploy/Demo
4. Add User Story 3 → Test frontend widget independently → Deploy/Demo
5. Add User Story 4 → Test selected text integration independently → Deploy/Demo
6. Add User Story 5 → Test security/performance independently → Deploy/Demo
7. Each story adds functionality without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (Indexing)
   - Developer B: User Story 2 (RAG Pipeline)
   - Developer C: User Story 5 (Security)
3. After US1+US2 complete:
   - Developer A: User Story 3 (Frontend)
   - Developer B: User Story 4 (Text Selection)
   - Developer C: Testing & Validation
4. All stories integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently functional and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate functionality story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break functional independence