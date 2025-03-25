import os
from typing import List, Dict, Any, Optional
import torch
from django.conf import settings
from smolagents import CodeAgent, tool
from smolagents.models import LocalModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from .models import Indexer

@tool
def query_database(sql: str) -> str:
    """
    Execute an SQL query against the PostgreSQL database to find files.
    
    Args:
        sql: SQL query to execute
    """
    try:
        from django.db import connection
        with connection.cursor() as cursor:
            cursor.execute(sql)
            results = cursor.fetchall()
            return str(results)
    except Exception as e:
        return f"Error executing query: {str(e)}"

class SearchAgentService:
    """Service that manages RAG-powered file search using smolagents"""
    
    def __init__(self):
        """Initialize the search agent with the configured RAG model"""
        self.model_name = settings.DEFAULT_RAG_MODEL
        self.model_path = settings.DEFAULT_RAG_MODEL_PATH
        self.device = self._get_device()
        self._initialize_model()
        self._initialize_agent()
    
    def _get_device(self) -> str:
        """Determine the best available device for inference"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _initialize_model(self) -> None:
        """Initialize the RAG model and tokenizer"""
        print(f"Loading model {self.model_name} on {self.device}...")
        
        # Set up cache directory
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(settings.BASE_DIR.parent, "HF_Models")
        
        # Initialize model with appropriate configurations
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.model_path,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.model_path,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map=self.device,
            trust_remote_code=True
        )
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.1
        )
    
    def _initialize_agent(self) -> None:
        """Initialize the CodeAgent for search operations"""
        # Create a smolagents compatible model
        model = LocalModel(
            pipeline=self.pipe,
            temperature=0.1,
            max_tokens=512
        )
        
        # Initialize the CodeAgent with our model and tools
        self.agent = CodeAgent(
            model=model,
            tools=[query_database]
        )
    
    def search_files(self, query: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Search for files using natural language query
        
        Args:
            query: Natural language query string
            top_n: Number of results to return
            
        Returns:
            List of file data dictionaries with match scores
        """
        query_prompt = f"""
        I need to search for files in a database that match the following description:
        "{query}"
        
        The database contains file metadata with:
        - file_name: The name of the file
        - file_path: Full path to the file
        - file_type: File extension
        - creation_date: When the file was created
        - size: Size in bytes
        - embedding: Vector embedding of the filename (ignore this for the query)
        
        Create a PostgreSQL query to find the most relevant files, considering:
        1. Exact or partial matches in file_name
        2. File types that might be related to the query
        3. Paths that might be relevant
        
        Limit the results to {top_n} items.
        
        Return only the POSTGRESQL QUERY:
        ```sql
        <your query here>
        ```
        """
        
        # Get the SQL query from the agent
        response = self.agent.run(query_prompt)
        sql_query = self._extract_sql_from_response(response)
        
        if not sql_query:
            # Fallback to basic search
            return self._fallback_search(query, top_n)
        
        # Execute the query and process results
        try:
            # Use Django's ORM raw query
            results = list(Indexer.objects.raw(sql_query))
            
            # Format the results
            formatted_results = []
            for item in results[:top_n]:
                formatted_results.append({
                    'id': item.id,
                    'file_name': item.file_name,
                    'file_path': item.file_path,
                    'file_type': item.file_type,
                    'creation_date': item.creation_date,
                    'size': self._format_size(item.size),
                })
            
            return formatted_results
        
        except Exception as e:
            print(f"Error executing SQL query: {e}")
            print(f"Query was: {sql_query}")
            return self._fallback_search(query, top_n)
    
    def _extract_sql_from_response(self, response: str) -> Optional[str]:
        """Extract SQL query from agent response"""
        if "```sql" in response and "```" in response.split("```sql")[1]:
            return response.split("```sql")[1].split("```")[0].strip()
        
        if "```" in response and "```" in response.split("```")[1]:
            return response.split("```")[1].split("```")[0].strip()
        
        return None
    
    def _fallback_search(self, query: str, top_n: int) -> List[Dict[str, Any]]:
        """Fallback to basic search when agent fails"""
        # Simple search by filename containing the query string
        results = Indexer.objects.filter(
            file_name__icontains=query
        ).order_by('-creation_date')[:top_n]
        
        return [
            {
                'id': item.id,
                'file_name': item.file_name,
                'file_path': item.file_path,
                'file_type': item.file_type,
                'creation_date': item.creation_date,
                'size': self._format_size(item.size),
            } for item in results
        ]
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size from bytes to human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0 or unit == 'TB':
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0