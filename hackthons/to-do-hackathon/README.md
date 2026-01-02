# Todo In-Memory Python Console App

A command-line todo application that stores tasks entirely in memory with a menu-driven interface. The application supports adding, listing, updating, deleting, and marking tasks as complete/incomplete with unique incremental IDs.

## Features

- Add tasks with title and optional description
- View all tasks with clear status indicators
- Update task details (title and/or description)
- Delete tasks
- Mark tasks as complete or incomplete
- Robust error handling for invalid inputs
- Clean console output with proper formatting

## Prerequisites

- Python 3.13 or higher

## Setup

1. Clone the repository
2. Ensure you have Python 3.13+ installed
3. No additional setup required - the application uses only Python standard library

## Usage

1. Navigate to the project directory
2. Run the application:

```bash
python src/main.py
```

3. Use the menu-driven interface to interact with your todo list:
   - 1. Add Task: Create a new task with a title and optional description
   - 2. List Tasks: View all tasks with their status indicators
   - 3. Update Task: Modify the title or description of an existing task
   - 4. Delete Task: Remove a task by its ID
   - 5. Mark Complete: Mark a task as complete
   - 6. Mark Incomplete: Mark a completed task as incomplete
   - 7. Exit: Quit the application

## Example Usage

```
Welcome to the Todo App!
1. Add Task
2. List Tasks
3. Update Task
4. Delete Task
5. Mark Complete
6. Mark Incomplete
7. Exit
Choose an option: 1
Enter task title: Buy groceries
Enter task description (optional): Get milk, bread, and eggs
Task added with ID 1.
```

## Architecture

The application follows a clean architecture with separation of concerns:

- `src/models/task.py`: Defines the Task dataclass
- `src/services/todo_manager.py`: Contains the business logic for managing tasks
- `src/cli/ui.py`: Handles user interface and input/output operations
- `src/main.py`: Entry point with the main application loop

## Notes

- All data is stored in memory only and will be lost when the program exits
- Task IDs are unique incremental integers that are preserved after deletions (not reused)
- The application follows PEP 8 standards with type hints and docstrings