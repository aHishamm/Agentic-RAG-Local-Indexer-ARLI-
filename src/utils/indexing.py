def index_files(file_paths):
    """
    Indexes the given list of file paths and returns a dictionary with file names as keys
    and their content as values.

    Args:
        file_paths (list): A list of file paths to be indexed.

    Returns:
        dict: A dictionary containing file names and their content.
    """
    indexed_files = {}
    for path in file_paths:
        with open(path, 'r') as file:
            indexed_files[path] = file.read()
    return indexed_files


def update_index(index, new_file):
    """
    Updates the index with a new file.

    Args:
        index (dict): The current index of files.
        new_file (str): The path of the new file to be added.

    Returns:
        dict: The updated index.
    """
    with open(new_file, 'r') as file:
        index[new_file] = file.read()
    return index


def remove_from_index(index, file_path):
    """
    Removes a file from the index.

    Args:
        index (dict): The current index of files.
        file_path (str): The path of the file to be removed.

    Returns:
        dict: The updated index.
    """
    if file_path in index:
        del index[file_path]
    return index