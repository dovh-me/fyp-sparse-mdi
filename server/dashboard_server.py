from flask import Flask, request, jsonify, send_file, stream_with_context
from flask_cors import CORS, cross_origin
from os import path
import os
import json
import zipfile
import gdown
import time

from util import logger, remove_dir_contents
logger = logger.logger

app = Flask(__name__)
cors = CORS(app) # allow CORS for all domains on all routes.
app.config['CORS_HEADERS'] = 'Content-Type'

# Configuration
CWD =   os.getcwd()
CONFIG_FILE_NAME = "config.json"

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
    print('update config hit')

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

@app.route('/health', methods=['GET'])
@cross_origin()
def health_check():
    """
    Simple health check endpoint
    """
    return jsonify({"status": "ok"}), 200


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)