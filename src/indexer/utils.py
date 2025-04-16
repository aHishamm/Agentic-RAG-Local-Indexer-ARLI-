import os
from datetime import datetime
from typing import Dict, Any, Optional, List, TYPE_CHECKING
import numpy as np
import torch
### added for new parallel code segment 
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tqdm import tqdm
from tqdm.auto import tqdm
### end of added parallel code segment 
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
from django.conf import settings

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
    Get metadata for a file including size, creation date, type, and whether it's a textual file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dict containing file metadata
    """
    try:
        stats = os.stat(file_path)
        _, extension = os.path.splitext(file_path)
        file_type = extension[1:] if extension else 'unknown'
        
        return {
            'file_name': os.path.basename(file_path),
            'file_path': os.path.abspath(file_path),
            'file_type': file_type,
            'creation_date': datetime.fromtimestamp(stats.st_ctime),
            'size': stats.st_size,
            'is_textual': is_textual_file(file_type)
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
def update_indexed_files(
    specific_directory: Optional[str] = None, 
    scan_specific_only: bool = False, 
    batch_size: int = 100
):
    """Update the database with the latest file system state using parallel processing."""
    from .models import Indexer
    from django.db import transaction
    from django.db.models import F
    
    def process_batch(batch):
        """Process a batch of files using bulk operations"""
        with transaction.atomic():
            # Separate new and existing files
            existing_paths = set(Indexer.objects.filter(
                file_path__in=[f['file_path'] for f in batch]
            ).values_list('file_path', flat=True))
            
            new_files = []
            update_files = []
            
            for file_data in batch:
                file_path = file_data['file_path']
                embedding = file_data.pop('embedding')
                
                if file_path not in existing_paths:
                    new_file = Indexer(**file_data)
                    new_file.set_embedding(embedding)
                    new_files.append(new_file)
                else:
                    update_files.append({
                        'file_path': file_path,
                        'data': file_data,
                        'embedding': embedding
                    })
            
            # Bulk create new files
            if new_files:
                Indexer.objects.bulk_create(new_files, batch_size=100)
            
            # Bulk update existing files
            if update_files:
                for update_batch in update_files:
                    Indexer.objects.filter(file_path=update_batch['file_path']).update(
                        **update_batch['data']
                    )
                    # Handle embedding update separately since it needs the model method
                    indexed_file = Indexer.objects.get(file_path=update_batch['file_path'])
                    indexed_file.set_embedding(update_batch['embedding'])
                    indexed_file.save(update_fields=['embedding'])

    def scan_directory(directory):
        """Scan directory using parallel processing"""
        cpu_count = mp.cpu_count()        
        progress_bar = None
        def init_progress_bar(total):
            nonlocal progress_bar
            progress_bar = tqdm(
                total=total,
                desc=f"Scanning {os.path.basename(directory)}",
                unit="files",
                position=0,
                leave=True
            )
            return progress_bar
        
        def update_progress(n):
            if progress_bar:
                progress_bar.update(n)        
        files = []
        for root, _, filenames in os.walk(directory):
            files.extend([os.path.join(root, f) for f in filenames])
        if not files:
            return []
        progress_bar = init_progress_bar(len(files))
        chunk_size = max(len(files) // (cpu_count * 2), 1)
        file_chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
        results = []
        with ThreadPoolExecutor(max_workers=cpu_count) as executor:
            scan_with_progress = partial(scan_files_chunk, update_progress=update_progress)
            for chunk_result in executor.map(scan_with_progress, file_chunks):
                if chunk_result:
                    results.extend(chunk_result)
        progress_bar.close()
        return results
    def scan_files_chunk(files, update_progress=None):
        """Process a chunk of files with progress updates"""
        results = []
        for f in files:
            if os.path.isfile(f):
                try:
                    metadata = get_file_metadata(f)
                    embedding = generate_filename_embedding(os.path.basename(f))
                    metadata['embedding'] = embedding
                    results.append(metadata)     
                    if update_progress:
                        update_progress(1)
                except Exception as e:
                    print(f"Error processing {f}: {e}")
                    if update_progress:
                        update_progress(1)
        
        return results
    current_batch = []
    if specific_directory and os.path.exists(specific_directory):
        files_data = scan_directory(specific_directory)
        for file_data in files_data:
            current_batch.append(file_data)
            if len(current_batch) >= batch_size:
                process_batch(current_batch)
                current_batch = []         
    if not scan_specific_only:
        directories = get_common_directories()
        for directory in directories:
            if os.path.exists(directory):
                files_data = scan_directory(directory)
                for file_data in files_data:
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

def is_textual_file(file_type: str) -> bool:
    """
    Determine if a file is textual based on its extension.
    
    Args:
        file_type: The file extension without the dot
        
    Returns:
        bool: True if the file is considered textual, False otherwise
    """
    textual_extensions = {
        # Document formats
        'txt', 'md', 'rtf', 'doc', 'docx', 'odt', 'pdf',
        # Code files
        'py', 'js', 'java', 'c', 'cpp', 'h', 'hpp', 'cs', 'php', 'rb', 'go',
        'rs', 'swift', 'kt', 'scala', 'r', 'sql', 'sh', 'bash', 'ps1',
        # Web files
        'html', 'htm', 'css', 'xml', 'json', 'yaml', 'yml',
        # Config files
        'ini', 'conf', 'cfg', 'properties', 'env',
        # Other text formats
        'csv', 'tsv', 'log', 'tex'
    }
    return file_type.lower() in textual_extensions

# Initialize the default model when the module is loaded
model = init_embedding_model()