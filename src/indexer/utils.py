import os
from datetime import datetime
from typing import Dict, Any, Optional, List, TYPE_CHECKING
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path
from django.conf import settings

if TYPE_CHECKING:
    from .models import Indexer

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

# Initialize the model when the module is loaded
model = init_embedding_model()
#arabic_model = init_embedding_model(settings.DEFAULT_EMBEDDING_ARABIC)

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
    Generate embedding for a filename using sentence-transformers.
    
    Args:
        filename: Name of the file to generate embedding for
        use_arabic_model: Whether to use the Arabic-specific model
    
    Returns:
        numpy array containing the embedding
    """
    embedding_model = arabic_model if use_arabic_model else model
    embedding = embedding_model.encode(filename, convert_to_tensor=False)
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
    except Exception as e:
        print(f"Error scanning directory {directory}: {e}")
    
    return files_metadata

def get_common_directories() -> List[str]:
    """Get list of common directories to scan based on OS."""
    home = str(Path.home())
    common_dirs = [
        os.path.join(home, 'Documents'),
        os.path.join(home, 'Downloads'),
        os.path.join(home, 'Desktop'),
    ]
    return [d for d in common_dirs if os.path.exists(d)]

def update_indexed_files():
    """Update the database with the latest file system state."""
    from .models import Indexer
    
    directories = get_common_directories()
    all_files = []
    for directory in directories:
        files = scan_directory(directory)
        all_files.extend(files)
    
    for file_data in all_files:
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