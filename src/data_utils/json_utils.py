import json
import os

def read_json_file(file_path, verbose=False):
    """Loads JSON data from a file.
    
    Args:
        file_path (str): Path to the JSON file.
        verbose (bool): If True, prints loading messages.
    
    Returns:
        dict or list: The parsed JSON data.
    """
    if verbose:
        print(f"Loading data from file: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        if verbose:
            print(f"Data loaded successfully, {len(data)} items found.")

        return data

    except (json.JSONDecodeError, FileNotFoundError, IOError) as e:
        if verbose:
            print(f"Error loading JSON file: {e}")
        return None  # Return None if there's an issue

def save_json(data, file_path, indent=4, verbose=False):
    """
    Saves a dictionary as a JSON file with optional indentation.

    Args:
        data (dict): The dictionary to be saved.
        file_path (str): The path where the JSON file will be saved.
        indent (int, optional): Indentation level for formatting. Default is 4.
        verbose (bool, optional): If True, prints a success message. Default is False.

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    try:
        # Ensure the directory exists
#         os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=indent, ensure_ascii=False)
        
        if verbose:
            print(f"✅ JSON data successfully saved to {file_path}")
        return True

    except (IOError, TypeError, ValueError) as e:
        print(f"❌ Error saving JSON file: {e}")
        return False
