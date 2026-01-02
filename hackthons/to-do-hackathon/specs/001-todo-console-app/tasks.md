# Implementation Tasks: Todo In-Memory Python Console App

**Feature**: Todo In-Memory Python Console App
**Branch**: `001-todo-console-app`
**Created**: 2026-01-02
**Status**: Draft
**Input**: Feature specification from `/specs/001-todo-console-app/spec.md`

## Dependencies

- User Story 2 (View Tasks) depends on User Story 1 (Add Tasks) for testing purposes
- User Stories 3-5 (Update, Delete, Mark Complete/Incomplete) depend on User Story 1 (Add Tasks) for testing purposes

## Parallel Execution Examples

- [P] Tasks in the foundational phase can be executed in parallel with each other
- [P] Tasks in different user story phases can be executed in parallel if they modify different files
- [P] UI implementation can be done in parallel with service implementation

## Implementation Strategy

- MVP: Implement User Story 1 (Add Tasks) with minimal UI to demonstrate core functionality
- Incremental delivery: Add one user story at a time, ensuring each is independently testable
- Follow clean architecture principles with separation of concerns

---

## Phase 1: Setup

- [X] T001 Create project structure per implementation plan: src/, src/models/, src/services/, src/cli/
- [X] T002 Create src/__init__.py files for proper Python package structure
- [X] T003 Set up basic project configuration files (pyproject.toml, .gitignore)

## Phase 2: Foundational

- [X] T004 [P] Create Task dataclass in src/models/task.py with id, title, description, completed fields
- [X] T005 [P] Create TodoList class in src/services/todo_manager.py with add_task method
- [X] T006 [P] Create basic UI functions in src/cli/ui.py for displaying menus
- [X] T007 [P] Create main.py entry point with basic menu structure

## Phase 3: User Story 1 - Add New Tasks (Priority: P1)

**Goal**: As a user, I want to add new tasks to my todo list with a title and optional description so that I can keep track of what I need to do.

**Independent Test**: Can be fully tested by adding tasks with various titles and descriptions, and verifying they appear in the task list with unique IDs.

**Acceptance Scenarios**:
1. Given I am using the todo application, When I choose to add a task with a valid title and optional description, Then the task is added to the list with a unique incremental ID and status of incomplete.
2. Given I am trying to add a task, When I enter an empty title, Then I receive an error message and the task is not added.

- [X] T008 [US1] Implement add_task method in TodoList class with validation for non-empty title
- [X] T009 [US1] Implement input validation for empty titles in UI layer
- [X] T010 [US1] Create add_task UI function to collect title and description from user
- [X] T011 [US1] Integrate add_task functionality with main menu loop
- [X] T012 [US1] Test adding tasks with various titles and descriptions

## Phase 4: User Story 2 - View All Tasks (Priority: P1)

**Goal**: As a user, I want to view all my tasks with clear status indicators so that I can see what I need to do and what I've completed.

**Independent Test**: Can be fully tested by adding tasks and then viewing the complete list with proper formatting, status indicators, and descriptions.

**Acceptance Scenarios**:
1. Given I have added several tasks, When I choose to view all tasks, Then I see a formatted list showing ID, title, status indicator ([ ] or [x]), and description preview.
2. Given I have no tasks in the list, When I choose to view all tasks, Then I see a "No tasks" message.

- [X] T013 [US2] Implement list_tasks method in TodoList class to return all tasks
- [X] T014 [US2] Create view_tasks UI function to display tasks with proper formatting
- [X] T015 [US2] Implement status indicators ([ ] or [x]) in task display
- [X] T016 [US2] Handle empty task list case with "No tasks" message
- [X] T017 [US2] Integrate view_tasks functionality with main menu loop
- [X] T018 [US2] Test viewing tasks with various statuses and descriptions

## Phase 5: User Story 3 - Update Task Details (Priority: P2)

**Goal**: As a user, I want to update the title and/or description of existing tasks so that I can modify my tasks as needed.

**Independent Test**: Can be fully tested by updating existing tasks and verifying the changes are reflected when viewing the task list.

**Acceptance Scenarios**:
1. Given I have existing tasks, When I choose to update a task with a valid ID and new details, Then the task details are updated in the list.
2. Given I attempt to update a task, When I enter an invalid task ID, Then I receive an error message and no changes are made.

- [X] T019 [US3] Implement update_task method in TodoList class with validation
- [X] T020 [US3] Create update_task UI function to collect task ID and new details
- [X] T021 [US3] Implement validation for invalid task IDs in UI layer
- [X] T022 [US3] Integrate update_task functionality with main menu loop
- [X] T023 [US3] Test updating task details with valid and invalid IDs

## Phase 6: User Story 4 - Delete Tasks (Priority: P2)

**Goal**: As a user, I want to delete tasks that I no longer need so that I can keep my todo list clean and focused.

**Independent Test**: Can be fully tested by deleting tasks and verifying they no longer appear in the task list.

**Acceptance Scenarios**:
1. Given I have existing tasks, When I choose to delete a task with a valid ID, Then the task is removed from the list.
2. Given I attempt to delete a task, When I enter an invalid task ID, Then I receive an error message and no tasks are removed.

- [X] T024 [US4] Implement delete_task method in TodoList class with validation
- [X] T025 [US4] Create delete_task UI function to collect task ID from user
- [X] T026 [US4] Implement validation for invalid task IDs in UI layer
- [X] T027 [US4] Preserve unique IDs after deletion (don't reuse IDs)
- [X] T028 [US4] Integrate delete_task functionality with main menu loop
- [X] T029 [US4] Test deleting tasks with valid and invalid IDs

## Phase 7: User Story 5 - Mark Tasks Complete/Incomplete (Priority: P2)

**Goal**: As a user, I want to mark tasks as complete or incomplete so that I can track my progress and know what still needs to be done.

**Independent Test**: Can be fully tested by marking tasks as complete/incomplete and verifying the status indicators update correctly.

**Acceptance Scenarios**:
1. Given I have existing tasks, When I mark a task as complete, Then the status indicator changes to [x] and the task is marked as completed.
2. Given I have completed tasks, When I mark a task as incomplete, Then the status indicator changes to [ ] and the task is marked as not completed.

- [X] T030 [US5] Implement mark_task_complete method in TodoList class
- [X] T031 [US5] Implement mark_task_incomplete method in TodoList class
- [X] T032 [US5] Create mark_complete UI function to collect task ID from user
- [X] T033 [US5] Create mark_incomplete UI function to collect task ID from user
- [X] T034 [US5] Integrate mark complete/incomplete functionality with main menu loop
- [X] T035 [US5] Test marking tasks as complete and incomplete

## Phase 8: Polish & Cross-Cutting Concerns

- [X] T036 Implement robust error handling for all user inputs (invalid IDs, empty titles, invalid menu choices)
- [X] T037 Ensure clean console output with proper formatting and status indicators
- [X] T038 Add confirmation messages for operations (task added, updated, deleted)
- [X] T039 Test edge cases: invalid menu choices, empty titles when updating, very long descriptions
- [X] T040 Create README.md with UV setup instructions and usage examples
- [X] T041 Verify all code follows PEP 8 standards with type hints and docstrings
- [X] T042 Run complete end-to-end test: adding → listing → updating → marking → deleting → listing again
- [X] T043 Verify data is lost on restart (in-memory storage requirement)