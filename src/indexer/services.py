from .models import Indexer
from .utils import get_file_metadata, process_file_content, is_supported_file_type
from typing import Optional, Dict, Any

def create_indexed_file(file_path: str) -> Optional[Indexer]:
    """
    Create a new indexed file entry in the database.
    
    Args:
        file_path: Path to the file to index
    Returns:
        Indexer instance or None if file can't be indexed
    """
    if not is_supported_file_type(file_path):
        return None
        
    try:
        metadata = get_file_metadata(file_path)
        document = process_file_content(file_path)
        
        if document:
            indexed_file = Indexer.objects.create(**metadata)
            return indexed_file
        return None
        
    except Exception as e:
        print(f"Error creating indexed file: {e}")
        return None

def update_indexed_file(file_id: int, **metadata: Dict[str, Any]) -> Optional[Indexer]:
    """
    Update an existing indexed file entry in the database.
    
    Args:
        file_id: ID of the indexed file to update
        metadata: Dictionary containing updated metadata for the file
    Returns:
        Updated Indexer instance or None if not found
    """
    try:
        indexed_file = Indexer.objects.get(id=file_id)
        for key, value in metadata.items():
            if hasattr(indexed_file, key):
                setattr(indexed_file, key, value)
        indexed_file.save()
        return indexed_file
    except Indexer.DoesNotExist:
        return None

def delete_indexed_file(file_id: int) -> bool:
    """
    Delete an indexed file entry from the database.
    
    Args:
        file_id: ID of the indexed file to delete
    Returns:
        True if file was deleted, False otherwise
    """
    try:
        Indexer.objects.filter(id=file_id).delete()
        return True
    except Exception:
        return False

def get_indexed_file(file_id: int) -> Optional[Indexer]:
    """
    Retrieve an indexed file entry from the database.
    
    Args:
        file_id: ID of the indexed file to retrieve
    Returns:
        Indexer instance or None if not found
    """
    try:
        return Indexer.objects.get(id=file_id)
    except Indexer.DoesNotExist:
        return None