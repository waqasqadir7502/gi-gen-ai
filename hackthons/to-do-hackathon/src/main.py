import sys
import os
from pathlib import Path

# Add the project root to the Python path so imports work correctly
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.services.todo_manager import TodoList
from src.cli.ui import (
    display_menu, get_user_choice, display_tasks,
    add_task_ui, update_task_ui, delete_task_ui,
    mark_complete_ui, mark_incomplete_ui
)


def main():
    """
    Main entry point for the todo application.
    """
    print("Starting Todo App...")
    
    # Initialize the todo list
    todo_list = TodoList()
    
    while True:
        display_menu()
        choice = get_user_choice()
        
        if choice == "1":
            add_task_ui(todo_list)
        elif choice == "2":
            display_tasks(todo_list)
        elif choice == "3":
            update_task_ui(todo_list)
        elif choice == "4":
            delete_task_ui(todo_list)
        elif choice == "5":
            mark_complete_ui(todo_list)
        elif choice == "6":
            mark_incomplete_ui(todo_list)
        elif choice == "7":
            print("Thank you for using the Todo App. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")
        
        # Pause to let user see the result before showing menu again
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()