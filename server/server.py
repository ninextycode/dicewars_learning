from flask import Flask, request, jsonify, send_from_directory
import os
import json
from datetime import datetime
import glob
import re

app = Flask(__name__)

# Directory to save game history files
SAVE_DIR = "history"
# Directory where your game files are located
PAGE_DIR = "page"  # Path to the page directory containing DICEWARS.html

# Create directory if it doesn't exist
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

@app.route('/api/save-history', methods=['POST'])
def save_history():
    try:
        # Get JSON data from request
        game_history = request.json
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Count existing history files using glob
        existing_files = glob.glob(os.path.join(SAVE_DIR, "history_*.json"))
        index = len(existing_files)
        filename = f"history_{index:05}_{timestamp}.json"
        filepath = os.path.join(SAVE_DIR, filename)
        
        # Save the JSON data to a file with lists of integers on a single line
        with open(filepath, 'w') as f:
            # Convert to JSON string with compact arrays
            # First create normal indented JSON
            json_str = json.dumps(game_history, indent=4, separators=(',', ': '))
            
            # Use regex to compress arrays of integers to a single line
            def compact_array(match):
                numbers = re.findall(r'\d+', match.group(0))
                return '[' + ', '.join(numbers) + ']'
                
            # Apply to arrays of integers in the JSON
            json_str = re.sub(r'\[\s*\n\s+\d+(?:,\s*\n\s+\d+)*\s*\n\s*\]', compact_array, json_str)
            # Write to file
            f.write(json_str)
        
        return jsonify({"success": True, "filename": filename})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# Serve the main HTML file
@app.route('/')
def index():
    return send_from_directory(PAGE_DIR, 'DICEWARS.html')

# Serve static files from the PAGE_DIR directory
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(PAGE_DIR, path)

# Serve files from the DICEWARS_files directory
@app.route('/DICEWARS_files/<path:path>')
def serve_game_files(path):
    return send_from_directory(os.path.join(PAGE_DIR, 'DICEWARS_files'), path)

if __name__ == '__main__':
    print(f"Starting server. Game files will be served from the '{PAGE_DIR}' directory")
    print(f"Game history will be saved to '{SAVE_DIR}' directory")
    print("Access the game at http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True)