# Agentic RAG Local Indexer (ARLI)

A Django-based application for intelligent file indexing and searching using RAG (Retrieval Augmented Generation) and embedding models.

## Features

- **Intelligent File Search**: Natural language search powered by RAG models
  - Use natural language to describe the files you're looking for
  - Smart file matching using both embeddings and content analysis
  - Fallback to basic search when RAG model is unavailable
  
- **File Indexing**:
  - Automatic indexing of common user directories
  - File metadata extraction and storage
  - Embedding generation for improved similarity search
  - Support for multiple languages including Arabic
  
- **Search Capabilities**:
  - Natural language queries using smolagents and LLMs
  - Similarity-based search using embeddings
  - Basic keyword search as fallback
  - File type filtering and path-based search

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Agentic-RAG-Local-Indexer-ARLI-
   ```

2. **Set up environment:**
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   Create a `.env` file based on `.env.example` with your settings:
   ```env
   SECRET_KEY=your_secret_key
   DEBUG=True
   ALLOWED_HOSTS=localhost 127.0.0.1 [::1]
   
   # Database settings
   DB_NAME=agentic_rag
   DB_USER=postgres
   DB_PASSWORD=your_password
   DB_HOST=localhost
   DB_PORT=5432
   ```

4. **Initialize the database:**
   ```bash
   python manage.py migrate
   ```

5. **Run the development server:**
   ```bash
   python manage.py runserver
   ```

## Usage

### Web Interface

1. Access the web interface at `http://localhost:8000`
2. Use the search bar to enter natural language queries:
   - "Find all PDF documents created last week"
   - "Show me Python files in the src directory"
   - "Find images larger than 1MB"

### API Endpoints

- `GET /api/files/` - List all indexed files
- `GET /api/files/<id>/` - Get specific file details
- `POST /api/search/` - Search files with natural language query
  ```json
  {
    "query": "find python files with database code",
    "top_n": 5
  }
  ```

## Development

### Running Tests

```bash
# Run all tests
python manage.py test

# Run specific test cases
python manage.py test indexer.tests.SearchAgentTestCase
```

### Project Structure

```
src/
├── core/            # Django project settings
├── indexer/         # Main application
│   ├── models.py    # Database models
│   ├── views.py     # View controllers
│   ├── utils.py     # Utility functions
│   ├── search_agent.py  # RAG search implementation
│   └── templates/   # HTML templates
└── manage.py        # Django management script
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Django](https://www.djangoproject.com/)
- RAG implementation using [smolagents](https://github.com/huggingface/smolagents)
- Embeddings by [SentenceTransformers](https://www.sbert.net/)