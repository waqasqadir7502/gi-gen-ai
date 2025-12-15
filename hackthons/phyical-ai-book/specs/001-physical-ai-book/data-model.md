# Data Model: Physical AI Book

## Educational Content Entity
- **Name**: EducationalContent
- **Fields**:
  - id: string (unique identifier)
  - title: string (chapter/module title)
  - content: markdown (main content body)
  - description: string (brief description)
  - level: enum (beginner, intermediate, advanced)
  - prerequisites: array of strings (required prior knowledge)
  - learningObjectives: array of strings (what students will learn)
  - examples: array of Example entities
  - exercises: array of Exercise entities
  - createdAt: date
  - updatedAt: date
- **Relationships**: Belongs to LearningPath, contains multiple Examples and Exercises

## Learning Path Entity
- **Name**: LearningPath
- **Fields**:
  - id: string (unique identifier)
  - title: string (path title)
  - description: string (path description)
  - modules: array of EducationalContent entities (ordered sequence)
  - totalDuration: number (estimated completion time in hours)
  - prerequisites: array of strings (overall prerequisites)
- **Relationships**: Contains multiple EducationalContent entities

## Example Entity
- **Name**: Example
- **Fields**:
  - id: string (unique identifier)
  - title: string (example title)
  - description: string (what the example demonstrates)
  - code: string (code snippet)
  - explanation: markdown (detailed explanation)
  - difficulty: enum (easy, medium, hard)
  - associatedContentId: string (links to EducationalContent)
- **Relationships**: Belongs to EducationalContent

## Exercise Entity
- **Name**: Exercise
- **Fields**:
  - id: string (unique identifier)
  - title: string (exercise title)
  - description: markdown (what students need to do)
  - instructions: markdown (step-by-step instructions)
  - starterCode: string (optional starter code)
  - solution: string (reference solution)
  - difficulty: enum (easy, medium, hard)
  - hints: array of strings (optional hints)
  - associatedContentId: string (links to EducationalContent)
- **Relationships**: Belongs to EducationalContent

## Docusaurus Structure Entity
- **Name**: DocusaurusStructure
- **Fields**:
  - sidebarConfig: object (navigation structure)
  - themeConfig: object (styling and layout options)
  - pluginConfig: object (additional functionality)
  - markdownExtensions: array of strings (supported markdown features)
- **Validation rules**: Must follow Docusaurus schema requirements

## Validation Rules
- All content must be accessible to AI beginners and intermediate learners (FR-001)
- Each chapter must include practical examples, exercises, or mini-projects (FR-002)
- Content must be properly formatted for Docusaurus platform (FR-003)
- Content must build progressively from basic to advanced topics (FR-004)
- Content must emphasize practical, implementable solutions (FR-005)
- Each chapter must include hands-on exercises appropriate for skill level (FR-008)
- Consistent terminology and teaching style across all chapters (FR-009)