<!-- SYNC IMPACT REPORT
Version change: 1.0.0 -> 1.1.0
Modified principles: None (completely new constitution for different project)
Added sections: All sections (new constitution for RAG Chatbot Integration)
Removed sections: Old Physical AI and Humanoid Robotics principles
Templates requiring updates:
- .specify/templates/plan-template.md ✅ updated
- .specify/templates/spec-template.md ✅ updated
- .specify/templates/tasks-template.md ✅ updated
- .specify/templates/commands/*.md ⚠ pending (no specific command files to update)
Templates requiring updates:
- README.md ⚠ pending (if exists)
Follow-up TODOs: None
-->

# Floating RAG Chatbot Integration for Physical AI Digital Book Constitution

## Core Principles

### I. Maximum Answer Accuracy and Faithfulness
The chatbot must provide answers that are strictly faithful to source material with a zero-hallucination policy. All responses must be grounded in documented content from the book, with clear attribution to specific sources when possible.

### II. Reasonable Content Freshness with Minimal Effort
Content updates should require minimal manual effort and must work within zero paid infrastructure constraints. The system should prioritize simplicity and maintainability over complex automated freshness mechanisms.

### III. Clean, Non-Destructive Integration
All integration work must follow clean, maintainable practices that integrate seamlessly into the existing Docusaurus project without disrupting current functionality. Changes should be minimal and reversible.

### IV. Source Material Quality Over Automation Complexity
Prioritize the quality and reliability of source materials over complex automation. High-quality content from reliable sources is preferred over sophisticated but risky automation processes.

### V. Privacy-First Design
Never log user questions, selected text, or responses. The system must be designed to protect user privacy by default, with no personally identifiable information stored or transmitted.

### VI. Security-First Architecture
No LLM keys should ever be exposed client-side. All sensitive information must be properly secured, and the architecture must prevent client-side exposure of API keys or other sensitive data.

## Content Sourcing Standards
Content sourcing must follow this strict priority order: (1) Local markdown files from /docs folder as the primary source, (2) Rendered HTML via sitemap.xml + content extraction as fallback, (3) All other methods are out of scope for v1. This ensures the highest quality and most reliable content sources.

## Integration Requirements
The chatbot must integrate using Docusaurus recommended patterns (Root wrapper preferred), appear as a floating bottom-right widget, support selected text as strong context, and avoid any external third-party chat services or widgets. The integration should be seamless and unobtrusive.

## Data Freshness Policy
For MVP phase, content is a controlled snapshot taken at ingestion time. Acceptable update delay is measured in days to weeks, with manual or semi-automated re-indexing as acceptable. Real-time or per-query live fetching is explicitly out of scope.

## Governance
This constitution governs all development for the Floating RAG Chatbot Integration project. All contributions must comply with these principles. Amendments require documentation of changes and approval from project maintainers. The system must be reviewed for compliance with accuracy, privacy, and security standards.

**Version**: 1.1.0 | **Ratified**: 2026-01-13 | **Last Amended**: 2026-01-13