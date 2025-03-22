from .models import IndexedFile

def create_indexed_file(file_path, metadata):
    """
    Create a new indexed file entry in the database.
    
    :param file_path: Path to the file being indexed.
    :param metadata: Dictionary containing metadata for the file.
    :return: IndexedFile instance.
    """
    indexed_file = IndexedFile.objects.create(file_path=file_path, **metadata)
    return indexed_file

def update_indexed_file(indexed_file_id, metadata):
    """
    Update an existing indexed file entry in the database.
    
    :param indexed_file_id: ID of the indexed file to update.
    :param metadata: Dictionary containing updated metadata for the file.
    :return: Updated IndexedFile instance.
    """
    indexed_file = IndexedFile.objects.get(id=indexed_file_id)
    for key, value in metadata.items():
        setattr(indexed_file, key, value)
    indexed_file.save()
    return indexed_file

def delete_indexed_file(indexed_file_id):
    """
    Delete an indexed file entry from the database.
    
    :param indexed_file_id: ID of the indexed file to delete.
    """
    IndexedFile.objects.filter(id=indexed_file_id).delete()

def get_indexed_file(indexed_file_id):
    """
    Retrieve an indexed file entry from the database.
    
    :param indexed_file_id: ID of the indexed file to retrieve.
    :return: IndexedFile instance or None if not found.
    """
    try:
        return IndexedFile.objects.get(id=indexed_file_id)
    except IndexedFile.DoesNotExist:
        return None