def read_file(file_path):
    """Read the contents of a file and return them."""
    with open(file_path, 'r') as file:
        return file.read()

def write_file(file_path, content):
    """Write content to a file."""
    with open(file_path, 'w') as file:
        file.write(content)

def delete_file(file_path):
    """Delete a file."""
    import os
    if os.path.exists(file_path):
        os.remove(file_path)

def list_files_in_directory(directory_path):
    """List all files in a directory."""
    import os
    return [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

def copy_file(source_path, destination_path):
    """Copy a file from source to destination."""
    import shutil
    shutil.copy(source_path, destination_path)

def move_file(source_path, destination_path):
    """Move a file from source to destination."""
    import shutil
    shutil.move(source_path, destination_path)