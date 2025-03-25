# Agentic RAG Local Indexer
This is an agentic RAG pipeline that runs locally, allowing access to external APIs and tools for performing tasks, as well as indexing local files for faster search. 

## Project Structure

```
agentic-rag-indexer
├── src
│   ├── manage.py
│   ├── core
│   │   ├── __init__.py
│   │   ├── settings.py
│   │   ├── urls.py
│   │   ├── asgi.py
│   │   └── wsgi.py
│   └── indexer
│       ├── __init__.py
│       ├── admin.py
│       ├── apps.py
│       ├── models.py
│       ├── services.py
│       ├── urls.py
│       ├── utils.py
│       ├── unit_test.py
│       └── views.py
├── docker
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── init-db
├── HF_Models            # Directory for storing HuggingFace models
│   ├── datasets
│   ├── hub
│   ├── metrics
│   └── models
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

## Getting Started

To get started with the project, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aHishamm/Agentic-RAG-Local-Indexer-ARLI-.git
   cd agentic-rag-indexer
   ```

2. **Set up environment variables:**
   - Copy the .env.example to .env (if not already done)
   - Update the environment variables as needed

3. **Start the Docker containers in detached mode:**
   ```bash
   cd docker && docker compose up -d
   ```
   This will start both the web (arli-web) and database (arli-db) containers in the background.

4. **Access the Python container:**
   ```bash
   docker exec -it arli-web bash
   ```
   This command gives you an interactive shell inside the Python container.

5. **Run database migrations:**
   Once inside the container, run:
   ```bash
   python src/manage.py makemigrations
   python src/manage.py migrate
   ```

6. **Create a superuser (optional):**
   ```bash
   python src/manage.py createsuperuser
   ```

7. **Access the application:**
   - Main application: http://localhost:8000
   - Admin interface: http://localhost:8000/admin

## Development Commands

Here are some useful commands for development:

- **View container logs:**
  ```bash
  docker compose logs -f
  ```

- **Stop the containers:**
  ```bash
  docker compose down
  ```

- **Rebuild and start containers:**
  ```bash
  docker compose up -d --build
  ```

## Features

- Local file system indexing
- Filename embedding generation using HuggingFace models
- Similar file search functionality
- PostgreSQL database for storing file metadata
- Docker containerization for easy deployment

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the Apache 2.0 License. See the LICENSE file for more details.