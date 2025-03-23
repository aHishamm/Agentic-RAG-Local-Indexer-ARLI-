from .models import Indexer

def create_indexed_file(file_name, file_path, file_type, creation_date, size):
    """
    Create a new indexed file entry in the database.
    
    :param file_name: Name of the file
    :param file_path: Path to the file
    :param file_type: Type of the file
    :param creation_date: Creation date of the file
    :param size: Size of the file in bytes
    :return: Indexer instance
    """
    indexed_file = Indexer.objects.create(
        file_name=file_name,
        file_path=file_path,
        file_type=file_type,
        creation_date=creation_date,
        size=size
    )
    return indexed_file

def update_indexed_file(file_id, **metadata):
    """
    Update an existing indexed file entry in the database.
    
    :param file_id: ID of the indexed file to update
    :param metadata: Dictionary containing updated metadata for the file
    :return: Updated Indexer instance
    """
    indexed_file = Indexer.objects.get(id=file_id)
    for key, value in metadata.items():
        if hasattr(indexed_file, key):
            setattr(indexed_file, key, value)
    indexed_file.save()
    return indexed_file

def delete_indexed_file(file_id):
    """
    Delete an indexed file entry from the database.
    
    :param file_id: ID of the indexed file to delete
    """
    Indexer.objects.filter(id=file_id).delete()

def get_indexed_file(file_id):
    """
    Retrieve an indexed file entry from the database.
    
    :param file_id: ID of the indexed file to retrieve
    :return: Indexer instance or None if not found
    """
    try:
        return Indexer.objects.get(id=file_id)
    except Indexer.DoesNotExist:
        return None