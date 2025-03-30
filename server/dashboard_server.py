import glob
from flask import Flask, request, jsonify, send_file, stream_with_context
from flask_cors import CORS, cross_origin
from os import path
import os
import json
import zipfile
import gdown
import time
import numpy as np

from util import logger, remove_dir_contents
logger = logger.logger

app = Flask(__name__)
cors = CORS(app) # allow CORS for all domains on all routes.
app.config['CORS_HEADERS'] = 'Content-Type'

# Configuration
CWD =   os.getcwd()
CONFIG_FILE_NAME = "config.json"
DEFAULT_CREDENTIAL = "120120"

CONFIG_FILE_PATH = path.join(CWD, CONFIG_FILE_NAME)
DEFAULT_CONFIG = {
    "server_config": {},
    "node_config": []
}

class DashboardServer():
    def __init__(self):
        self.server = None 
        self.server_config = None

        # Initialize config at startup
        self.initialize_config()

    def start_server(self, server, server_config = {}):
        host = server_config.get("host", "0.0.0.0")
        port = server_config.get("port", "5000")

        self.server = server
        self.server_config = server

        # Download the model partitions if the model partitions dir is not available
        partitions_dir = self.server.model_partitions_dir
        has_model = len(glob.glob(f"{partitions_dir}/*.onnx")) > 0

        if not has_model:
            print("No model parts available. Downloading model from config.json...")
            self.download_and_extract_model_parts()

        app.run(debug=False, host=host, port=port)
    
    def download_and_extract_model_parts(self):
        # Open the updated config file
        with open(CONFIG_FILE_PATH, 'r') as f:
            config = json.loads(f.read())

        server_config = config.get("server_config", None)

        if server_config == None: 
            logger.log("Server config not found. Aborting download.")
            return

        gdrive_model_parts_id = server_config.get('model_gdrive_id');
        download_file_name = "model_parts.zip"

        if(gdrive_model_parts_id == None):
            raise Exception("Model parts url not configured")
        
        print(gdrive_model_parts_id)
        # Download the model parts zip
        output_file_name = gdown.download(id=gdrive_model_parts_id, output=download_file_name)

        # Clear the current directory
        remove_dir_contents.remove_dir_contents(self.server.model_partitions_dir) 

        with zipfile.ZipFile(output_file_name, 'r') as zip_ref:
            zip_ref.extractall(self.server.model_partitions_dir)
        
        # Verify the model parts exist
        files_list = os.listdir(self.server.model_partitions_dir)
        has_model = False
        for file in files_list:
            has_model = file.endswith('.onnx')
            if(has_model):
                break
        
        if not has_model:
            raise Exception("Models unavailable")

    # Ensure the config file exists
    def initialize_config(self):
        if not os.path.exists(CONFIG_FILE_PATH):
            with open(CONFIG_FILE_PATH, 'w') as f:
                json.dump(DEFAULT_CONFIG, f, indent=4)

    def authenticate(self, password):
        valid_password = os.environ.get("SERVER_PWD", DEFAULT_CREDENTIAL)
        return password == valid_password


dashboard_server = DashboardServer()

@app.route('/config', methods=['GET'])
@cross_origin()
def get_config():
    """
    Endpoint to retrieve the current JSON configuration
    """
    return send_file(CONFIG_FILE_PATH, mimetype='application/json')

@app.route('/download-progress', methods=['GET'])
@cross_origin()
def download_progress():
    """
    Endpoint to retrieve the current JSON configuration
    """
    return send_file(CONFIG_FILE_PATH, mimetype='application/json')

@app.route('/config', methods=['POST'])
@cross_origin()
def update_config():
    """
    Endpoint to upload and save a new JSON configuration
    """
    def generate():
        try:
            # Check if the request contains JSON data
            if not request.is_json:
                yield "Request must contain JSON data.\n"
                return  # Stop execution if the request is invalid
            
            # Get the JSON data from the request
            new_config = request.get_json()

            # Save the new configuration to the file
            try:
                with open(CONFIG_FILE_PATH, 'w') as f:
                    json.dump(new_config, f, indent=4)
            except Exception as file_error:
                logger.error(f"Failed to save configuration: {file_error}")
                yield f"Error saving configuration: {file_error}\n"
                return
            
            yield "Updated configuration.\n"
            time.sleep(1)
            yield "Downloading ModelParts...\n"

            # Attempt to download and extract model parts
            try:
                dashboard_server.download_and_extract_model_parts()
                yield "Successfully downloaded ModelParts.\n"
                time.sleep(1)
            except Exception as model_error:
                logger.error(f"Failed to download ModelParts: {model_error}")
                yield f"Error downloading ModelParts: {model_error}\n"
                return

            yield "Model deployment success. Please restart the network to see the configuration in effect.\n"

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            yield f"An unexpected error occurred: {e}\n"

    return stream_with_context(generate())

@app.route('/infer', methods=['POST'])
async def handle_raw_inference():
    try:
        image_bytes = request.data  # Read raw bytes

        # Convert JSON to Python dict
        data = json.loads(image_bytes)

        # Convert dict values to byte array
        image_bytes = bytes([data[str(i)] for i in range(len(data))])
        image_bytes = dashboard_server.server.preprocess_image(image_bytes)
        result = await dashboard_server.server.start_inference(image_bytes)  # Pass bytes to inference function
        
        result = np.squeeze(result)
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        result = softmax(result)
        # Get indices of top 5 probabilities (unordered but faster)
        top_5_unordered = np.argpartition(result, -5)[-5:]

        # Sort these indices based on actual values to get correct order
        top_5_indices = top_5_unordered[np.argsort(result[top_5_unordered])[::-1]][:5]

        # Get corresponding probabilities
        top_5_probs = result[top_5_indices]

        return jsonify({
            'class_indices': top_5_indices.tolist(),
            'probabilities': top_5_probs.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
@cross_origin()
def health_check():
    """
    Simple health check endpoint
    """
    is_ready = dashboard_server.server.network_ready_future.done()
    return jsonify({"status": "ok", "is_ready": is_ready, "is_online": True}), 200

@app.route('/auth', methods=['POST'])
@cross_origin()
def authenticate():
    """
    Simple login endpoint 
    """
    try:
        # Check if the request contains JSON data
        if not request.is_json:
            return jsonify({"status": "error", "message": "Credentials are Required"}) 
                
        # Get the JSON data from the request
        credentials = request.get_json()
        password = credentials.get('password')

        if dashboard_server.authenticate(password):
            return jsonify({"status": "authenticated", "message": "Login Success"})
        else:
            return jsonify({"status": "error", "message": "Invalid Credentials"}) 
    except:
            return jsonify({"status": "error", "message": "Unexpected Error Occurred"}) 


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)