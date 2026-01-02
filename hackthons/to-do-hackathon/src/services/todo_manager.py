from typing import List, Optional

# Import Task using sys.path manipulation to work from project root
import sys
import os
from pathlib import Path

# Add the project root to the Python path so imports work correctly
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.task import Task


class TodoList:
    """
    Manages a collection of Task objects in memory.
    """
    
    def __init__(self):
        """
        Initialize an empty todo list with a starting ID counter.
        """
        self.tasks: List[Task] = []
        self.next_id: int = 1

    def add_task(self, title: str, description: Optional[str] = None) -> Task:
        """
        Creates a new task with the next available ID and adds it to the list.
        
        Args:
            title: Required string, non-empty, represents the task name
            description: Optional string, provides additional details about the task
            
        Returns:
            Task: The newly created task object
            
        Raises:
            ValueError: If title is empty
        """
        if not title or not title.strip():
            raise ValueError("Title cannot be empty")
        
        task = Task(
            id=self.next_id,
            title=title.strip(),
            description=description,
            completed=False
        )
        
        self.tasks.append(task)
        self.next_id += 1  # Increment for next task
        
        return task

    def get_task(self, task_id: int) -> Optional[Task]:
        """
        Retrieves a task by its ID.
        
        Args:
            task_id: The unique identifier of the task
            
        Returns:
            Task object if found, None otherwise
        """
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def update_task(self, task_id: int, title: Optional[str] = None, description: Optional[str] = None) -> bool:
        """
        Updates a task's title and/or description.
        
        Args:
            task_id: The unique identifier of the task to update
            title: New title (optional)
            description: New description (optional)
            
        Returns:
            bool: True if task was updated, False if task not found
        """
        task = self.get_task(task_id)
        if task is None:
            return False
        
        # Only update fields that are provided
        if title is not None:
            if not title.strip():
                raise ValueError("Title cannot be empty")
            task.title = title.strip()
        
        if description is not None:
            task.description = description
        
        return True

    def delete_task(self, task_id: int) -> bool:
        """
        Removes a task by its ID.
        
        Args:
            task_id: The unique identifier of the task to remove
            
        Returns:
            bool: True if task was removed, False if task not found
        """
        task = self.get_task(task_id)
        if task is None:
            return False
        
        # Remove the task from the list
        self.tasks.remove(task)
        return True

    def mark_task_complete(self, task_id: int) -> bool:
        """
        Marks a task as complete.
        
        Args:
            task_id: The unique identifier of the task to mark complete
            
        Returns:
            bool: True if task was marked complete, False if task not found
        """
        task = self.get_task(task_id)
        if task is None:
            return False
        
        task.completed = True
        return True

    def mark_task_incomplete(self, task_id: int) -> bool:
        """
        Marks a task as incomplete.
        
        Args:
            task_id: The unique identifier of the task to mark incomplete
            
        Returns:
            bool: True if task was marked incomplete, False if task not found
        """
        task = self.get_task(task_id)
        if task is None:
            return False
        
        task.completed = False
        return True

    def list_tasks(self) -> List[Task]:
        """
        Returns all tasks in the list.
        
        Returns:
            List of all Task objects
        """
        return self.tasks.copy()  # Return a copy to prevent external modification