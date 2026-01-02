---
id: 1
title: Create Constitution
stage: constitution
date: 2026-01-02
surface: agent
model: Qwen
feature: none
branch: main
user: user
command: /sp.constitution
labels: [constitution, yaml, todo-app]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - D:\Profession\Studies\Coding\GI AI\Hackathon\to-do-hackathon\constitution.yaml
tests:
 - none
---

## Prompt

Generate the complete content of the constitution file for Phase I of the hackathon project titled "The Evolution of Todo". The constitution must be in YAML format and suitable for direct saving as `constitution.yaml` in the root of the GitHub repository.

### Project Context
- Project Name: The Evolution of Todo
- Phase: Phase I - Todo In-Memory Python Console App
- Date: January 02, 2026
- Overall Theme: This project simulates the real-world evolution of software, starting with a simple script and progressing toward a Kubernetes-managed, event-driven, AI-powered distributed system.
- Focus for Phase I: Build a command-line todo application that stores tasks entirely in memory (no persistence; data is lost when the program exits).
- Development Approach: Strictly follow the Agentic Dev Stack workflow:
  Write spec → Generate plan → Break into tasks → Implement via Qwen Code (or Claude Code).
  No manual coding or editing of generated code is allowed.
- Judging Criteria: The process (prompts, spec iterations, planning quality), adherence to spec-driven development, and overall use of Spec-Kit Plus and Claude Code will be reviewed.

### Core Requirements
Implement exactly these 5 Basic Level features:
1. Add a task with title and description
2. Delete a task by ID
3. Update task details (title and/or description) by ID
4. View/List all tasks with status indicators
5. Mark a task as complete or incomplete by ID

### Additional Functional Details
- Interactive console application (menu-driven or command-based loop)
- Each task must have:
  - Unique incremental integer ID (starting from 1)
  - Title (required, non-empty string)
  - Description (optional string)
  - Completed status (boolean, default false)
- When listing tasks:
  - Show ID, title, status indicator (e.g., [ ] for incomplete, [x] for complete), and description preview
  - Display "No tasks" message if empty
- Robust error handling for invalid inputs (non-existent IDs, empty titles, invalid commands)

### Non-Functional Requirements
- Follow clean code principles: PEP 8 compliance, meaningful names, modularity, docstrings, type hints
- Proper Python project structure (package layout with src/, __init__.py where needed, clear entry point)
- No external dependencies beyond the Python standard library
- Use Python 3.13+ features where appropriate

### Technology Stack
- Package management: UV
- Language: Python 3.13+
- Spec management: Spec-Kit Plus
- Code generation: Qwen Code (primary) and Claude Code (for planning/specs)

### Repository Deliverables
The GitHub repository must contain:
- `constitution.yaml` (this file) at the root
- `specs_history/` folder containing all specification files and iterations (e.g., spec_v1.yaml, spec_v2.yaml, etc.)
- `src/` folder containing all AI-generated Python source code (no manual modifications allowed)
- `README.md` with:
  - Project overview
  - Setup instructions (including UV commands)
  - How to run the application
  - Demonstration of all features

### Working Application Requirements
The console application must fully demonstrate:
- Adding tasks with title and description
- Listing all tasks with clear status indicators
- Updating task details by ID
- Deleting tasks by ID
- Marking tasks as complete/incomplete by ID

### Constitution Structure Guidelines
Use YAML format with clear, logical sections such as:
- project_metadata (name, phase, description, date)
- overview
- objectives
- features (detailed breakdown of the 5 features with acceptance criteria)
- user_interface (menu options and interaction flow)
- data_model (task structure and in-memory storage)
- technical_constraints
- tech_stack
- project_structure (expected files and folders)
- deliverables
- evaluation_criteria (emphasizing process, iterations, and spec quality)

Make the constitution precise, detailed, and unambiguous to guide subsequent spec creation and ensure AI-generated code perfectly matches the requirements. Use proper YAML syntax and indentation.

## Response snapshot

Created a complete YAML constitution file for the "The Evolution of Todo" project with all required sections including project metadata, overview, objectives, detailed features with acceptance criteria, user interface specifications, data model, technical constraints, tech stack, project structure, deliverables, and evaluation criteria.

## Outcome

- ✅ Impact: Created the foundational constitution.yaml file that will guide the entire Phase I development process
- 🧪 Tests: none
- 📁 Files: Created constitution.yaml in the project root
- 🔁 Next prompts: Will proceed with generating specifications based on this constitution
- 🧠 Reflection: The constitution provides clear, unambiguous requirements for AI code generation with specific acceptance criteria for each feature

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): not applicable
- Prompt variant (if applicable): not applicable
- Next experiment (smallest change to try): Generate the initial specification based on this constitution