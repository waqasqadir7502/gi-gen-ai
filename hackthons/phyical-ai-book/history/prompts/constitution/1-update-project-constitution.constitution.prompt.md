---
id: 1
title: update-project-constitution
stage: constitution
date: 2026-01-13
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: main
user: user
command: sp.constitution
labels: [constitution, project-setup, governance]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
 - none
---

## Prompt

Project: Floating RAG Chatbot Integration for Physical AI Digital Book
Book live URL: https://physical-ai-book-lilac.vercel.app/
Current reference context: January 2026

Core Principles:
• Maximum answer accuracy and faithfulness to source material (strict no-hallucination policy)
• Reasonable content freshness with minimal manual effort and zero paid infrastructure
• Clean, non-destructive, maintainable integration into existing Docusaurus project
• Strong preference for source material quality over automation complexity
• Accept realistic trade-off: near-real-time freshness is not feasible in free-tier + part-time scenario
• Privacy-first: never log user questions, selected text, or responses
• Security-first: no LLM keys ever exposed client-side

Key Standards & Explicit Decisions:
Data Freshness Expectation:
  • MVP phase: content is a controlled snapshot taken at ingestion time
  • Acceptable update delay: days to weeks (manual or semi-automated re-index acceptable)
  • Real-time / per-query live fetching is explicitly out of scope

Content Sourcing Priority (strict order – do not deviate):
  1. Local markdown files from /docs folder (cleanest, highest quality)
  2. Rendered HTML via sitemap.xml + content extraction (only as fallback)
  3. Any other method → out of scope for v1

Integration Philosophy:
  • Use Docusaurus recommended patterns (Root wrapper preferred)
  • Floating bottom-right chatbot widget
  • Support for selected text as strong context (nice-to-have but high priority)
  • Zero external third-party chat services/widgets

All decisions must align with: accuracy > simplicity > freshness > automation

## Response snapshot

Updated the project constitution file (.specify/memory/constitution.md) to reflect the new Floating RAG Chatbot Integration project instead of the previous Physical AI and Humanoid Robotics book project. The new constitution includes six core principles aligned with the user's requirements:
1. Maximum Answer Accuracy and Faithfulness
2. Reasonable Content Freshness with Minimal Effort
3. Clean, Non-Destructive Integration
4. Source Material Quality Over Automation Complexity
5. Privacy-First Design
6. Security-First Architecture

The constitution also includes sections on Content Sourcing Standards, Integration Requirements, Data Freshness Policy, and Governance, all aligned with the project requirements. The version was updated from 1.0.0 to 1.1.0 with a sync impact report included.

## Outcome

- ✅ Impact: Updated project constitution to align with RAG chatbot integration requirements
- 🧪 Tests: none
- 📁 Files: Updated .specify/memory/constitution.md with new principles and governance
- 🔁 Next prompts: Proceed with spec, plan, and implementation phases
- 🧠 Reflection: The constitution now properly guides the project toward the stated goals of accuracy, simplicity, and security

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): PASS - all requirements implemented
- Prompt variant (if applicable): none
- Next experiment (smallest change to try): Begin with project specification phase