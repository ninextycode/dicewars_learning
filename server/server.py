# Standard library imports
import os
import glob
import traceback
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
import torch
from server.utils import add_winners_info, extract_input, extract_action
from server.json_utils import json_to_string
from learning.model.actionvalue_v2 import ActionValueModelV2
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# Initialize Flask application
app = Flask(__name__)

# Directory to save game history files
SAVE_DIR = "history"
# Directory where your game files are located
PAGE_DIR = "page"  # Path to the page directory containing DICEWARS.html

# Global variable for the model (initialized as None)
global_model = None


@app.route("/api/save_history/<history_tag>", methods=["POST"])
def save_history(history_tag):
    try:
        # Get JSON data from request
        game_history = request.json
        add_winners_info(game_history)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Count existing history files using glob
        history_dir = os.path.join(SAVE_DIR, history_tag)
        
        # Create directory if it doesn"t exist
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)
            logging.info(f"Created history directory: {history_dir}")

        existing_files = glob.glob(os.path.join(history_dir, "history_*.json"))
        index = len(existing_files)
        filename = f"history_{index:05}_{timestamp}.json"
        filepath = os.path.join(history_dir, filename)

        # Save the JSON data to a file
        with open(filepath, "w") as f:
            json_str = json_to_string(game_history)
            f.write(json_str)
        
        logging.info(f"Saved history to: {filepath}")
        return jsonify({"success": True, "filepath": filepath})
    except Exception as e:
        error_traceback = traceback.format_exc()
        logging.error(f"Exception in save_history({history_tag}): {str(e)}\n{error_traceback}")
        return jsonify({"success": False, "error": str(e)}), 500


def get_model():
    """
    Get the initialized model instance.
    
    Returns:
        The initialized and loaded model in evaluation mode
    """
    global global_model
    
    # If the model hasn't been loaded yet, load it
    if global_model is None:
        global_model = ActionValueModelV2()
        checkpoint = torch.load(
            "learning/actionvalue_v2_no_freeze_checkpoints/actionvalue_v2_no_freeze_000213.pt",
            map_location=torch.device("cpu")
        )
        model_state = checkpoint["model_state"]
        global_model.load_state_dict(model_state)
        global_model.eval()
    
    return global_model


@app.route("/api/model_action", methods=["POST"])
def model_action():
    try:
        game_state = request.json
        model = get_model()  # Get the model instance
        with torch.no_grad():
            input_tensors = extract_input(game_state)
            model_output = model(*input_tensors)
            node_states, edges = input_tensors[:2]
        attach_edge = extract_action(model_output, node_states, edges)
        return jsonify({"success": True, "attach_edge": attach_edge})
    except Exception as e:
        error_traceback = traceback.format_exc()
        logging.error(f"Exception: {str(e)}\n{error_traceback}")
        return jsonify({"success": False, "error": str(e)}), 500


# Serve the main HTML file
@app.route("/")
def index():
    return send_from_directory(PAGE_DIR, "DICEWARS.html")

# Serve static files from the PAGE_DIR directory
@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(PAGE_DIR, path)

# Serve files from the DICEWARS_files directory
@app.route("/DICEWARS_files/<path:path>")
def serve_game_files(path):
    return send_from_directory(os.path.join(PAGE_DIR, "DICEWARS_files"), path)

if __name__ == "__main__":
    print(f"Starting server. Game files will be served from the '{PAGE_DIR}' directory")
    print(f"Game history will be saved to '{SAVE_DIR}/<history_tag>' directories")
    print(f"History API endpoint: /api/save_history/<history_tag>")
    print("Access the game at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)