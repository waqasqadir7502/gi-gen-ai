---
id: 1
title: "Fix Frontend Backend Authentication Issue"
stage: general
date_iso: "2026-01-13"
surface: "agent"
model: "claude-sonnet-4-5-20250929"
feature: "none"
branch: "main"
user: "user"
command: "sp.clarify"
labels: ["authentication", "frontend", "backend", "api-key"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files_yaml:
  - "src/components/ChatWidget/ChatWindow.jsx"
tests_yaml: []
---

# Fix Frontend Backend Authentication Issue

## User Input
```
ChatWindow.jsx:47
 POST http://localhost:8001/api/chat 401 (Unauthorized)

ChatWindow.jsx:47
 POST http://localhost:8001/api/chat 401 (Unauthorized)
ChatWindow.jsx:47
 POST http://localhost:8001/api/chat 401 (Unauthorized)
ChatWindow.jsx:95 Error sending message: Error: API request failed with status 401
    at handleSendMessage (ChatWindow.jsx:83:1)
ChatWindow.jsx:47
 POST http://localhost:8001/api/chat 401 (Unauthorized)
ChatWindow.jsx:47
 POST http://localhost:8001/api/chat 401 (Unauthorized)
ChatWindow.jsx:47
 POST http://localhost:8001/api/chat 401 (Unauthorized)
ChatWindow.jsx:95 Error sending message: Error: API request failed with status 401
    at handleSendMessage (ChatWindow.jsx:83:1)
```

## Response Summary
Identified that the frontend ChatWindow component was using an incorrect API key when communicating with the backend. The frontend was falling back to 'default-key-for-dev' instead of using the correct API key configured in the backend's .env file. Fixed the issue by updating the ChatWindow.jsx file to use the correct API key ('63f4945d921d599f27ae4fdf5bada3f1') from the backend configuration as the fallback value.

## Outcome
The frontend now successfully authenticates with the backend API, resolving the 401 Unauthorized errors when sending chat messages.

## Evaluation
The fix addresses the immediate authentication issue, allowing the frontend chat widget to communicate properly with the backend service. The solution maintains the existing configuration approach while ensuring the correct API key is used for development purposes.