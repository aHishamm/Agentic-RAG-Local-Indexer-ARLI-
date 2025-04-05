# Agentic RAG Local Indexer (ARLI) - Usage Guide

## Overview

ARLI is a powerful file indexing and search system that combines traditional search capabilities with modern AI techniques:

- Natural language search using RAG (Retrieval Augmented Generation)
- Intelligent file matching using embeddings
- Support for both English and Arabic content
- Automatic file indexing of common user directories

## Basic Usage

### Web Interface

1. Access the search interface:
   ```
   http://localhost:8000
   ```

2. Use the search form to:
   - Enter natural language queries
   - Specify number of results to return
   - View detailed file information including size, type, and creation date

### Features

#### Natural Language Search
Instead of using exact keywords, you can search using natural descriptions:

- "Find Python files related to database operations"
- "Show me documents created last week about project planning"
- "Find images in my Downloads folder"

#### Multilingual Support
ARLI supports both English and Arabic content:
- Uses UAE-Large-V1 for general embeddings
- Employs ArabertV2 for Arabic text
- Automatically detects language and uses appropriate model

#### File Management
The system automatically:
- Indexes common directories (Documents, Downloads, Desktop)
- Extracts and stores file metadata
- Generates and updates embeddings for efficient search

## Advanced Usage

### API Endpoints

#### Search Files
```http
POST /api/search/
Content-Type: application/json

{
    "query": "find python files with database code",
    "top_n": 5
}
```

#### List Files
```http
GET /api/files/
```

#### Get File Details
```http
GET /api/files/<id>/
```

### Environment Configuration

Key settings in `.env`:

```env
# Model Selection
DEFAULT_EMBEDDING=WhereIsAI/UAE-Large-V1
DEFAULT_EMBEDDING_ARABIC=aubmindlab/bert-base-arabertv2
DEFAULT_RAG_MODEL=Qwen/Qwen2.5-Coder-3B-Instruct

# Database
DB_NAME=agentic_rag
DB_USER=postgres
DB_PASSWORD=postgrespassword
DB_HOST=db
DB_PORT=5432
```

### Customization

1. **Adding Custom Directories**
   Modify `get_common_directories()` in `indexer/utils.py`:
   ```python
   def get_common_directories():
       home = str(Path.home())
       return [
           os.path.join(home, 'Documents'),
           os.path.join(home, 'Downloads'),
           # Add your custom directories here
       ]
   ```

2. **Changing Embedding Models**
   Update settings in `core/settings.py`:
   ```python
   DEFAULT_EMBEDDING = 'your-preferred-model'
   DEFAULT_EMBEDDING_ARABIC = 'your-preferred-arabic-model'
   ```
### Examples 
1. **Indexing a custom directory** 
```bash
python manage.py shell 
```
```python
from indexer import utils 
# example path to a directory in the host OS 
path_to_directory = '/host/Users/ahishamm/Downloads' #keep /host/ in path 
utils.update_indexed_files(path_to_directory,scan_specific_only=True)
```

## Troubleshooting

### Common Issues

1. **Search Returns No Results**
   - Check if directories are properly indexed
   - Verify database connection settings
   - Ensure RAG model is properly loaded

2. **Slow Search Performance**
   - Consider reducing indexed directories
   - Check available system resources
   - Verify database indexes are properly created

3. **Model Loading Errors**
   - Ensure sufficient disk space for models
   - Check internet connection for first-time model downloads
   - Verify GPU/CPU compatibility

### Debugging

Enable Django debug mode in `.env`:
```env
DEBUG=True
```

Check logs for detailed error messages:
```bash
docker logs arli-py
```

## Contributing

1. Run tests before submitting changes:
   ```bash
   python manage.py test
   ```

2. Follow the existing code style and documentation patterns
3. Add unit tests for new features
4. Update documentation as needed

For more details, see the main [README.md](../README.md) in the project root.