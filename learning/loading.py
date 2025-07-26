# /home/maxim/Programming/dicewar_learning/utils/json_reader.py
import json

def read_json(file_path):
    """
    Read and parse a JSON file.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        dict: Parsed JSON data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        return None

# Example usage
if __name__ == "__main__":
    data = read_json("path/to/your/file.json")
    if data:
        print(f"Successfully loaded JSON with {len(data)} elements")
