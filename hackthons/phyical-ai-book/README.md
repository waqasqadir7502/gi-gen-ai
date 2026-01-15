# Physical AI and Humanoid Robotics Book with RAG Chatbot

This repository contains the educational content for a comprehensive book on Physical AI and Humanoid Robotics, structured as 4 modules with 4 chapters each (16 total chapters). The content is designed for beginners and intermediate learners, with hands-on exercises and practical examples. The site includes an integrated RAG (Retrieval-Augmented Generation) chatbot for enhanced user experience.

## Project Structure

The book is organized into 4 modules, each containing 4 chapters:

### Module 1: Foundations of Physical AI
- Chapter 1.1: Introduction to Physical AI Concepts
- Chapter 1.2: History and Evolution of Physical AI
- Chapter 1.3: Basic Mathematics for Physical AI
- Chapter 1.4: Simulation Environment Setup

### Module 2: Humanoid Robotics Fundamentals
- Chapter 2.1: Kinematics and Movement Systems
- Chapter 2.2: Sensors and Perception Systems
- Chapter 2.3: Actuators and Control Systems
- Chapter 2.4: Basic Locomotion Patterns

### Module 3: Control and Intelligence
- Chapter 3.1: Balance and Stability Control
- Chapter 3.2: Path Planning and Navigation
- Chapter 3.3: Manipulation and Grasping
- Chapter 3.4: Learning-Based Control

### Module 4: Applications and Integration
- Chapter 4.1: Human-Robot Interaction
- Chapter 4.2: Multi-Sensor Fusion
- Chapter 4.3: Real-World Deployment Considerations
- Chapter 4.4: Capstone Project - Complete Robot System

## Getting Started

### Prerequisites

- Node.js version 18 or higher
- Python 3.8 or higher (for backend services and simulation exercises)
- Basic programming knowledge

### Installation

1. Clone this repository:
```bash
git clone [repository-url]
cd physical-ai-book
```

2. Install Docusaurus dependencies:
```bash
npm install
```

3. Install backend dependencies:
```bash
cd backend
pip install -r requirements.txt
cd ..
```

4. Create environment file for backend:
```bash
cp .env.example .env
# Edit .env with your API keys
```

5. Start the development server:
```bash
# Terminal 1: Start the Docusaurus development server
npm start

# Terminal 2: Start the backend API server
cd backend
uvicorn main:app --reload --port 8000
```

This will start a local development server and open the documentation in your browser at `http://localhost:3000`. The chatbot will connect to http://localhost:8000/api/chat.

## Chatbot Integration

The site includes a RAG (Retrieval-Augmented Generation) chatbot that allows users to ask questions about the book content:

### Backend Services:
- Uses Cohere for embeddings and generation
- Uses Qdrant Cloud for vector storage
- Protected with API key authentication

### Frontend Widget:
- Floating chat widget in bottom-right corner
- Supports selected text context
- Responsive design for mobile devices

### Content Indexing:
- Automatically indexes markdown files from `/docs` folder
- Updates vector database with new content
- Preserves document structure and metadata

### Environment Variables (set in `.env` file):
```
COHERE_API_KEY=your_cohere_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
BACKEND_API_KEY=generate_your_own_secure_key
```

## Re-indexing Content

When documentation is updated, re-run the indexing process:

```bash
cd backend
python scripts/ingest_docs.py --docs-path ../docs
```

## Simulation Environment

The hands-on exercises in this book use PyBullet, a physics simulator for robotics. To set up the simulation environment:

1. Install Python dependencies:
```bash
pip install pybullet pybullet-data numpy matplotlib
```

2. Follow the setup instructions in Chapter 1.4 for detailed configuration.

## Contributing

This project follows the content standards outlined in `docs/content-standards.md`. All contributions should adhere to the established format and quality guidelines.

## License

This educational content is made available under [specify license]. See the LICENSE file for more details.

## About

This project was created as part of the AI Book Hackathon to provide accessible educational content on Physical AI and Humanoid Robotics. The content follows a hands-on learning approach with practical examples and exercises. The integrated RAG chatbot enhances the learning experience by allowing users to ask questions about the content.