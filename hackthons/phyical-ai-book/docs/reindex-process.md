# Re-indexing Process for Physical AI Book RAG Chatbot

This document describes how to re-index the content when the documentation is updated.

## Manual Re-index Process

### Prerequisites
- Python 3.8+
- Required packages (install with `pip install -r backend/requirements.txt`)
- Valid Cohere API key
- Valid Qdrant Cloud credentials
- Updated documentation in the `/docs` folder

### Steps

1. **Set up environment variables**
   ```bash
   # Copy the example environment file
   cp .env.example .env

   # Update the .env file with your credentials:
   COHERE_API_KEY=your_cohere_api_key
   QDRANT_URL=your_qdrant_url
   QDRANT_API_KEY=your_qdrant_api_key
   BACKEND_API_KEY=generate_your_own_secure_key
   ```

2. **Navigate to the backend directory**
   ```bash
   cd backend
   ```

3. **Run the ingestion script**
   ```bash
   python scripts/ingest_docs.py --docs-path ../docs
   ```

4. **Verify the indexing**
   - Check the console output for successful indexing messages
   - Verify that the expected number of documents were processed
   - Test the chatbot to confirm new content is searchable

### Command Line Options

The ingestion script supports the following options:

- `--docs-path`: Path to the docs folder (default: `../docs`)
- `--batch-size`: Batch size for indexing (default: 10)
- `--skip-duplicates`: Skip duplicate content detection
- `--dry-run`: Show what would be indexed without actually indexing

Example:
```bash
python scripts/ingest_docs.py --docs-path ../my-docs --batch-size 20 --skip-duplicates
```

### Troubleshooting

- **Connection errors**: Verify your API keys and network connectivity
- **Memory issues**: Reduce the batch size
- **Timeout errors**: Check your internet connection and API provider status
- **Authentication errors**: Double-check your API keys

### Best Practices

- Always run a `--dry-run` first to see what will be processed
- Monitor the console output for any warnings or errors
- Test the chatbot after re-indexing to ensure content is searchable
- Keep backups of your vector database if needed