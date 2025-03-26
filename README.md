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
  - Multiple embedding models for improved matching:
    - UAE-Large-V1 for general purpose embeddings
    - Arabert-v2 for Arabic text support
  - Intelligent language detection and model selection
  
- **Multilingual Support**:
  - Arabic language support using AraberV2 model
  - Automatic language detection for optimal model selection
  - High-quality embeddings for both English and Arabic content
  
- **Search Capabilities**:
  - Natural language queries using smolagents and LLMs
  - Similarity-based search using embeddings
  - Basic keyword search as fallback
  - File type filtering and path-based search

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aHishamm/Agentic-RAG-Local-Indexer-ARLI-.git
   cd Agentic-RAG-Local-Indexer-ARLI-
   ```

2. **Create a .env file:**
   Create a `.env` file in the root directory (or use the provided `.env.example`). The default values should work out of the box:
   ```env
   SECRET_KEY=your_secret_key
   DEBUG=True
   ALLOWED_HOSTS=localhost 127.0.0.1 [::1]
   ```

3. **Start the application:**
   ```bash
   # (-d) to run in detached mode 
   docker-compose up -d --build  

   ```
   This will:
   - Build the Docker containers
   - Set up the PostgreSQL database
   - Install all dependencies
   - Run migrations automatically
   - Start the development server

4. **Access the application:**
   Open your web browser and navigate to `http://localhost:8000`

That's it! The application is now running and ready to use.

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

## Configuration

The application uses the following models:

- **Default Embedding Model**: WhereIsAI/UAE-Large-V1
- **Arabic Embedding Model**: aubmindlab/bert-base-arabertv2
- **RAG Model**: Qwen/Qwen2.5-Coder-3B-Instruct

These can be configured in your environment or settings.py:

```env
DEFAULT_EMBEDDING=WhereIsAI/UAE-Large-V1
DEFAULT_EMBEDDING_ARABIC=aubmindlab/bert-base-arabertv2
DEFAULT_RAG_MODEL=Qwen/Qwen2.5-Coder-3B-Instruct
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