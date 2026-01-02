# Data Model for Todo In-Memory Python Console App

## Task Entity

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class Task:
    id: int
    title: str
    description: Optional[str] = None
    completed: bool = False
```

### Fields
- **id**: Unique incremental integer identifier (starting from 1)
- **title**: Required string, non-empty, represents the task name
- **description**: Optional string, provides additional details about the task
- **completed**: Boolean flag indicating whether the task is completed (default: False)

### Validation Rules
- Title must be a non-empty string
- ID must be a positive integer
- ID must be unique across all tasks

### State Transitions
- A task can transition from incomplete (completed=False) to complete (completed=True)
- A task can transition from complete (completed=True) back to incomplete (completed=False)

## Todo List Collection

```python
from typing import List
from models import Task

class TodoList:
    def __init__(self):
        self.tasks: List[Task] = []
        self.next_id: int = 1
    
    def add_task(self, title: str, description: Optional[str] = None) -> Task:
        # Creates a new task with the next available ID
        pass
    
    def get_task(self, task_id: int) -> Optional[Task]:
        # Retrieves a task by its ID
        pass
    
    def update_task(self, task_id: int, title: Optional[str] = None, description: Optional[str] = None) -> bool:
        # Updates a task's title and/or description
        pass
    
    def delete_task(self, task_id: int) -> bool:
        # Removes a task by its ID
        pass
    
    def mark_task_complete(self, task_id: int) -> bool:
        # Marks a task as complete
        pass
    
    def mark_task_incomplete(self, task_id: int) -> bool:
        # Marks a task as incomplete
        pass
```

### Collection Behavior
- Maintains a list of Task objects
- Tracks the next available ID to ensure uniqueness
- Preserves IDs after deletion (does not reuse deleted IDs)
- Provides methods for all required CRUD operations