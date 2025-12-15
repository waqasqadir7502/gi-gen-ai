# Quickstart: Physical AI and Humanoid Robotics Book

## Setup Instructions

### 1. Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Install dependencies
npm install

# Install Docusaurus globally (if not already installed)
npm install -g @docusaurus/core@latest
```

### 2. Project Initialization
```bash
# Initialize Docusaurus project
npx create-docusaurus@latest website classic

# Navigate to the project directory
cd website
```

### 3. Content Structure Setup
```bash
# Create the required directory structure
mkdir -p docs/intro
mkdir -p docs/examples/chapter-1 docs/examples/chapter-2 docs/examples/chapter-3 docs/examples/chapter-4
mkdir -p docs/exercises/chapter-1 docs/exercises/chapter-2 docs/exercises/chapter-3 docs/exercises/chapter-4
mkdir -p docs/tutorials
mkdir -p src/components
mkdir -p static/img
```

### 4. Running the Development Server
```bash
# Start the development server
npm start

# The site will be available at http://localhost:3000
```

### 5. Adding New Content
```bash
# Add a new chapter
# Create a new markdown file in docs/intro/ (e.g., chapter-5.md)

# Add examples for the chapter
# Create example files in docs/examples/chapter-5/

# Add exercises for the chapter
# Create exercise files in docs/exercises/chapter-5/
```

### 6. Building for Production
```bash
# Build the static site
npm run build

# Serve the built site locally for testing
npm run serve
```

## Key Configuration Files

### docusaurus.config.js
- Main configuration file for site metadata, themes, and plugins
- Configure navigation sidebar, footer, and other site-wide settings

### sidebars.js
- Defines the navigation structure for the documentation
- Organizes content in a logical, hierarchical manner

## Content Creation Workflow

1. **Create Chapter Content**: Write the main content in Markdown format
2. **Add Examples**: Create practical examples demonstrating concepts
3. **Develop Exercises**: Create hands-on exercises for student practice
4. **Review & Test**: Validate content for clarity and Docusaurus compatibility
5. **Publish**: Add to navigation and deploy to production

## Quality Checks

Before marking content as complete:
- [ ] Content is accessible to AI beginners
- [ ] Chapter includes practical examples
- [ ] Hands-on exercises are provided
- [ ] Content builds progressively from basic to advanced
- [ ] Docusaurus build passes without errors
- [ ] Navigation links work correctly
- [ ] All images and assets load properly