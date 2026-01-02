# Todo App API Contracts

## Core Operations

### Add Task
- **Input**: title (string, required), description (string, optional)
- **Output**: task object with id, title, description, completed status
- **Errors**: Invalid input (empty title)

### List Tasks
- **Input**: None
- **Output**: Array of task objects
- **Errors**: None

### Update Task
- **Input**: task_id (int), title (string, optional), description (string, optional)
- **Output**: boolean indicating success
- **Errors**: Task not found

### Delete Task
- **Input**: task_id (int)
- **Output**: boolean indicating success
- **Errors**: Task not found

### Mark Complete
- **Input**: task_id (int)
- **Output**: boolean indicating success
- **Errors**: Task not found

### Mark Incomplete
- **Input**: task_id (int)
- **Output**: boolean indicating success
- **Errors**: Task not found