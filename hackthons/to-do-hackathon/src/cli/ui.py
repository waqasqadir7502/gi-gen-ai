from typing import Optional

# Import using sys.path manipulation to work from project root
import sys
import os
from pathlib import Path

# Add the project root to the Python path so imports work correctly
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.services.todo_manager import TodoList
from src.models.task import Task


def display_menu() -> None:
    """
    Display the main menu options to the user.
    """
    print("\n" + "="*40)
    print("Welcome to the Todo App!")
    print("="*40)
    print("1. Add Task")
    print("2. List Tasks")
    print("3. Update Task")
    print("4. Delete Task")
    print("5. Mark Complete")
    print("6. Mark Incomplete")
    print("7. Exit")
    print("="*40)


def get_user_choice() -> str:
    """
    Get and validate the user's menu choice.
    
    Returns:
        str: The user's choice as a string
    """
    while True:
        try:
            choice = input("Choose an option (1-7): ").strip()
            if choice in ["1", "2", "3", "4", "5", "6", "7"]:
                return choice
            else:
                print("Invalid choice. Please enter a number between 1 and 7.")
        except KeyboardInterrupt:
            print("\nExiting...")
            return "7"  # Return exit option


def get_task_input(todo_list: TodoList) -> Optional[Task]:
    """
    Helper function to get a valid task ID from user input.
    
    Args:
        todo_list: The TodoList instance to check against
        
    Returns:
        Task object if found, None if user cancels or invalid input
    """
    while True:
        try:
            task_id_input = input("Enter task ID (or 'cancel' to go back): ").strip()
            if task_id_input.lower() == 'cancel':
                return None
            
            task_id = int(task_id_input)
            task = todo_list.get_task(task_id)
            if task:
                return task
            else:
                print(f"No task found with ID {task_id}. Please try again.")
        except ValueError:
            print("Please enter a valid task ID (number) or 'cancel'.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return None


def display_tasks(todo_list: TodoList) -> None:
    """
    Display all tasks in a formatted way.
    
    Args:
        todo_list: The TodoList instance containing tasks to display
    """
    tasks = todo_list.list_tasks()
    
    if not tasks:
        print("\nNo tasks found.")
        return
    
    print("\nYour Tasks:")
    print("-" * 60)
    for task in tasks:
        status = "[x]" if task.completed else "[ ]"
        description = task.description if task.description else ""
        # Limit description length for display
        desc_preview = (description[:30] + "...") if len(description) > 30 else description
        print(f"{task.id:2d} {status} {task.title} - {desc_preview}")
    print("-" * 60)


def add_task_ui(todo_list: TodoList) -> None:
    """
    UI function to collect task details from user and add to the list.
    
    Args:
        todo_list: The TodoList instance to add the task to
    """
    try:
        title = input("Enter task title: ").strip()
        if not title:
            print("Title cannot be empty.")
            return
        
        description = input("Enter task description (optional): ").strip()
        if not description:
            description = None
        
        task = todo_list.add_task(title, description)
        print(f"Task added with ID {task.id}.")
    except ValueError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nOperation cancelled.")


def update_task_ui(todo_list: TodoList) -> None:
    """
    UI function to collect task update details from user.
    
    Args:
        todo_list: The TodoList instance containing the task to update
    """
    task = get_task_input(todo_list)
    if not task:
        return
    
    print(f"Current task: {task.title}")
    if task.description:
        print(f"Current description: {task.description}")
    
    new_title = input(f"Enter new title (or press Enter to keep '{task.title}'): ").strip()
    if not new_title:
        new_title = None  # Keep the original title
    
    new_description = input(f"Enter new description (or press Enter to keep current): ").strip()
    if not new_description:  # If user enters empty string, keep original
        new_description = task.description
    elif new_description.lower() == task.description:  # If user enters same as current
        new_description = task.description
    elif new_description.lower() == 'none' or new_description.lower() == 'null':
        new_description = None  # Allow setting to None
    
    try:
        updated = todo_list.update_task(task.id, new_title, new_description)
        if updated:
            print("Task updated successfully.")
        else:
            print("Failed to update task.")
    except ValueError as e:
        print(f"Error: {e}")


def delete_task_ui(todo_list: TodoList) -> None:
    """
    UI function to collect task ID from user and delete the task.
    
    Args:
        todo_list: The TodoList instance containing the task to delete
    """
    task = get_task_input(todo_list)
    if not task:
        return
    
    confirm = input(f"Are you sure you want to delete task '{task.title}'? (y/N): ").strip().lower()
    if confirm in ['y', 'yes']:
        deleted = todo_list.delete_task(task.id)
        if deleted:
            print("Task deleted successfully.")
        else:
            print("Failed to delete task.")
    else:
        print("Deletion cancelled.")


def mark_complete_ui(todo_list: TodoList) -> None:
    """
    UI function to mark a task as complete.
    
    Args:
        todo_list: The TodoList instance containing the task to mark
    """
    task = get_task_input(todo_list)
    if not task:
        return
    
    marked = todo_list.mark_task_complete(task.id)
    if marked:
        print(f"Task '{task.title}' marked as complete.")
    else:
        print("Failed to mark task as complete.")


def mark_incomplete_ui(todo_list: TodoList) -> None:
    """
    UI function to mark a task as incomplete.
    
    Args:
        todo_list: The TodoList instance containing the task to mark
    """
    task = get_task_input(todo_list)
    if not task:
        return
    
    marked = todo_list.mark_task_incomplete(task.id)
    if marked:
        print(f"Task '{task.title}' marked as incomplete.")
    else:
        print("Failed to mark task as incomplete.")