from dataclasses import dataclass
from typing import Optional


@dataclass
class Task:
    """
    Represents a single todo item with ID, title, description, and completion status.
    """
    id: int
    title: str
    description: Optional[str] = None
    completed: bool = False