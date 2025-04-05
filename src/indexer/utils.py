import os
from datetime import datetime
from typing import Dict, Any, Optional, List, TYPE_CHECKING
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
from django.conf import settings
from tqdm import tqdm

if TYPE_CHECKING:
    from .models import Indexer

# Import the SearchAgentService
from .search_agent import SearchAgentService

# Singleton instances
_search_agent_instance = None
_search_agent_initialized = False
_arabic_model = None
_arabic_tokenizer = None

def setup_model_directory() -> str:
    """
    Create and return the path to the HF_Models directory for storing downloaded models.
    Returns:
        str: Absolute path to the HF_Models directory
    """
    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / 'HF_Models'
    models_dir.mkdir(exist_ok=True)
    return str(models_dir)

def init_embedding_model(model_name: Optional[str] = None) -> SentenceTransformer:
    """
    Initialize the sentence transformer model with the appropriate device.
    Downloads the model to HF_Models directory if not already present and loads it to the most efficient available device.
    
    Args:
        model_name: Optional name of the model to load from HuggingFace hub. 
                   If None, uses DEFAULT_EMBEDDING from settings
        
    Returns:
        Initialized SentenceTransformer model
    """
    models_dir = setup_model_directory()
    os.environ['TRANSFORMERS_CACHE'] = models_dir
    os.environ['HF_HOME'] = models_dir
    
    # Use settings-defined model if none specified
    if model_name is None:
        model_name = settings.DEFAULT_EMBEDDING
        model_path = settings.DEFAULT_EMBEDDING_MODEL_PATH
    else:
        model_path = os.path.join(models_dir, model_name)
    
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print(f"Loading model {model_name} on {device} device...")
    print(f"Models will be stored in: {model_path}")
    
    model = SentenceTransformer(model_name, device=device, cache_folder=model_path)
    return model

def init_arabic_model():
    """Initialize the Arabic BERT model using transformers directly"""
    global _arabic_model, _arabic_tokenizer
    
    if _arabic_model is not None and _arabic_tokenizer is not None:
        return
        
    models_dir = setup_model_directory()
    model_name = settings.DEFAULT_EMBEDDING_ARABIC
    model_path = settings.DEFAULT_EMBEDDING_MODEL_PATH_ARABIC
    
    # Determine device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
        
    print(f"Loading Arabic model {model_name} on {device} device...")
    
    # Initialize tokenizer and model
    _arabic_tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_path)
    _arabic_model = AutoModel.from_pretrained(model_name, cache_dir=model_path).to(device)
    _arabic_model.eval()  # Set to evaluation mode

def generate_arabic_embedding(text: str) -> np.ndarray:
    """Generate embeddings using the Arabic BERT model"""
    global _arabic_model, _arabic_tokenizer
    
    # Initialize model if not already done
    if _arabic_model is None or _arabic_tokenizer is None:
        init_arabic_model()
    
    # Tokenize and prepare input
    inputs = _arabic_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(_arabic_model.device) for k, v in inputs.items()}
    
    # Generate embeddings
    with torch.no_grad():
        outputs = _arabic_model(**inputs)
        # Use mean pooling of last hidden states as embedding
        attention_mask = inputs['attention_mask']
        embeddings = mean_pooling(outputs.last_hidden_state, attention_mask)
        
    return embeddings[0].cpu().numpy().astype(np.float32)

def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Calculate mean pooling of token embeddings using attention mask"""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Initialize the default model when the module is loaded
model = init_embedding_model()

def get_file_metadata(file_path: str) -> Dict[str, Any]:
    """
    Get metadata for a file including size, creation date, and type.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dict containing file metadata
    """
    try:
        stats = os.stat(file_path)
        _, extension = os.path.splitext(file_path)
        
        return {
            'file_name': os.path.basename(file_path),
            'file_path': os.path.abspath(file_path),
            'file_type': extension[1:] if extension else 'unknown',
            'creation_date': datetime.fromtimestamp(stats.st_ctime),
            'size': stats.st_size
        }
    except OSError as e:
        raise ValueError(f"Error getting file metadata: {e}")

def generate_filename_embedding(filename: str, use_arabic_model: bool = False) -> np.ndarray:
    """
    Generate embedding for a filename using either sentence-transformers or Arabic BERT.
    
    Args:
        filename: Name of the file to generate embedding for
        use_arabic_model: Whether to use the Arabic-specific model
    
    Returns:
        numpy array containing the embedding
    """
    if use_arabic_model:
        return generate_arabic_embedding(filename)
    else:
        embedding = model.encode(filename, convert_to_tensor=False)
        return embedding.astype(np.float32)

def search_similar_files(query: str, indexed_files: List['Indexer'], top_k: int = 5) -> List[Dict[str, Any]]:
    """Search for files with similar names using embeddings."""
    query_embedding = generate_filename_embedding(query)
    
    similarities = []
    for file in indexed_files:
        file_embedding = file.get_embedding()
        if file_embedding is not None:
            similarity = np.dot(query_embedding, file_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(file_embedding)
            )
            similarities.append((file, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [
        {
            'file_name': file.file_name,
            'file_path': file.file_path,
            'similarity': float(sim)
        }
        for file, sim in similarities[:top_k]
    ]

def scan_directory(directory: str) -> List[Dict[str, Any]]:
    """Recursively scan a directory and collect file metadata."""
    files_metadata = []
    try:
        # First pass to count total files
        total_files = sum([len(files) for _, _, files in os.walk(directory)])
        
        # Second pass with progress bar
        with tqdm(total=total_files, desc=f"Scanning {os.path.basename(directory)}") as pbar:
            for root, _, files in os.walk(directory):
                for file in files:
                    try:
                        file_path = os.path.join(root, file)
                        metadata = get_file_metadata(file_path)
                        embedding = generate_filename_embedding(file)
                        metadata['embedding'] = embedding
                        files_metadata.append(metadata)
                    except Exception as e:
                        print(f"Error processing {file}: {e}")
                        continue
                    finally:
                        pbar.update(1)
    except Exception as e:
        print(f"Error scanning directory {directory}: {e}")
    
    return files_metadata

def get_common_directories() -> List[str]:
    """directories based on exposed home OS"""
    base_path = '/host/Users'
    # Get all user directories
    user_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    common_dirs = []
    # For each user, add their common directories
    for user in user_dirs:
        user_path = os.path.join(base_path, user)
        common_dirs.extend([
            os.path.join(user_path, 'Documents'),
            os.path.join(user_path, 'Downloads'), 
            os.path.join(user_path, 'Desktop'),
        ])
    return [d for d in common_dirs if os.path.exists(d)]

def update_indexed_files(specific_directory: Optional[str] = None, scan_specific_only: bool = False, batch_size: int = 35):
    """Update the database with the latest file system state in batches.
    
    Args:
        specific_directory: Optional path to a specific directory to scan
        scan_specific_only: If True, only scan the specific_directory. If False, scan common directories as well.
        batch_size: Number of files to process in each database batch
    """
    from .models import Indexer
    from django.db import transaction
    current_batch = []
    def process_batch(batch):
        """Helper function to process a batch of files"""
        with transaction.atomic():
            for file_data in batch:
                embedding = file_data.pop('embedding')
                try:
                    indexed_file, created = Indexer.objects.get_or_create(
                        file_path=file_data['file_path'],
                        defaults=file_data
                    )
                    if not created:
                        for key, value in file_data.items():
                            setattr(indexed_file, key, value)
                    
                    indexed_file.set_embedding(embedding)
                    indexed_file.save()
                    
                except Exception as e:
                    print(f"Error updating database for {file_data['file_path']}: {e}")
                    continue
    if specific_directory:
        if os.path.exists(specific_directory):
            for file_data in scan_directory(specific_directory):
                current_batch.append(file_data)
                if len(current_batch) >= batch_size:
                    process_batch(current_batch)
                    current_batch = []
        else:
            print(f"Warning: Specified directory {specific_directory} does not exist")    
    if not scan_specific_only:
        directories = get_common_directories()
        for directory in directories:
            for file_data in scan_directory(directory):
                current_batch.append(file_data)
                if len(current_batch) >= batch_size:
                    process_batch(current_batch)
                    current_batch = []    
    if current_batch:
        process_batch(current_batch)

def get_search_agent() -> Optional[SearchAgentService]:
    """
    Get or create a singleton instance of the SearchAgentService.
    This avoids loading the RAG model multiple times.
    
    Returns:
        Initialized SearchAgentService instance or None if initialization failed
    """
    global _search_agent_instance
    global _search_agent_initialized
    
    if not _search_agent_initialized:
        try:
            _search_agent_instance = SearchAgentService()
            _search_agent_initialized = True
        except Exception as e:
            print(f"Error initializing SearchAgentService: {e}")
            _search_agent_initialized = True  # Mark as initialized even if it failed
            _search_agent_instance = None     # But set instance to None to indicate failure
    
    return _search_agent_instance

def search_files_with_rag(query: str, top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Search for files using the RAG model and natural language query.
    This is a convenience wrapper around the SearchAgentService.
    
    Args:
        query: Natural language query string
        top_n: Number of results to return
        
    Returns:
        List of file data dictionaries with match information
    """
    search_agent = get_search_agent()
    
    if search_agent is None:
        print("SearchAgentService not available. Falling back to basic search.")
        from .models import Indexer
        # Use a simple search as fallback
        results = Indexer.objects.filter(file_name__icontains=query).order_by('-creation_date')[:top_n]
        return [{
            'id': item.id,
            'file_name': item.file_name,
            'file_path': item.file_path,
            'file_type': item.file_type,
            'creation_date': item.creation_date,
            'size': _format_size(item.size),
            'note': 'Basic search only (RAG model unavailable)'
        } for item in results]
    
    results = search_agent.search_files(query, top_n)
    
    if not results:
        print(f"No results found for query: '{query}'")
    
    return results

def _format_size(size_bytes: int) -> str:
    """Format file size from bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0 or unit == 'TB':
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0