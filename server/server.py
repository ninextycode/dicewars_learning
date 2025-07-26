from flask import Flask, request, jsonify, send_from_directory
import os
import json
from datetime import datetime
import glob

app = Flask(__name__)

# Directory to save game history files
SAVE_DIR = "history"
# Directory where your game files are located
PAGE_DIR = "page"  # Path to the page directory containing DICEWARS.html

# Create directory if it doesn't exist
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


class CompactJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that formats bottom-level objects on a single line
    and higher-level objects with indentation.
    """
    def __init__(self, *args, **kwargs):
        # Extract indent, and if it's missing or None, default to 4
        indent = kwargs.pop('indent', None)
        if indent is None:
            indent = 4
        kwargs['indent'] = indent

        super().__init__(*args, **kwargs)

    def iterencode(self, o, _one_shot=False):
        def _iterencode(obj, level):
            # bottom-level dict: all values are primitives
            if isinstance(obj, dict) and level > 0 and all(
                not isinstance(v, (dict, list)) for v in obj.values()
            ):
                yield json.dumps(obj,
                                 separators=(',', ': '),
                                 ensure_ascii=self.ensure_ascii)
                return

            # bottom-level list: all items are primitives
            if isinstance(obj, list) and level > 0 and all(
                not isinstance(v, (dict, list)) for v in obj
            ):
                yield json.dumps(obj,
                                 separators=(',', ':'),
                                 ensure_ascii=self.ensure_ascii)
                return

            # pretty-print dict
            if isinstance(obj, dict):
                yield '{\n'
                indent_str = ' ' * (self.indent * (level + 1))
                for i, (k, v) in enumerate(obj.items()):
                    if i:
                        yield ',\n'
                    yield indent_str + json.dumps(k) + ': '
                    yield from _iterencode(v, level + 1)
                yield '\n' + ' ' * (self.indent * level) + '}'
                return

            # pretty-print list
            if isinstance(obj, list):
                yield '[\n'
                indent_str = ' ' * (self.indent * (level + 1))
                for i, v in enumerate(obj):
                    if i:
                        yield ',\n'
                    yield indent_str
                    yield from _iterencode(v, level + 1)
                yield '\n' + ' ' * (self.indent * level) + ']'
                return

            # primitive fallback
            yield json.dumps(obj, ensure_ascii=self.ensure_ascii)

        return _iterencode(o, 0)


def json_to_string(data):
    """
    Convert JSON data to a formatted string with bottom-level objects on a single line.
    
    Args:
        data: The JSON data to format
        
    Returns:
        A formatted JSON string
    """
    return json.dumps(data, cls=CompactJSONEncoder)


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
        
        # Save the JSON data to a file
        with open(filepath, 'w') as f:
            json_str = json_to_string(game_history)
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