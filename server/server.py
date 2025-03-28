import sys
import os
import asyncio
import multiprocessing 
from concurrent import futures
import grpc
import traceback
import numpy as np
import json
from PIL import Image
import threading
import io
import glob

from modules.SparsityEngine import SparsityEngine

module_path = os.path.abspath('../')
sys.path.insert(0, module_path)

import generated.server_pb2 as server_pb2
import generated.server_pb2_grpc as server_pb2_grpc
import generated.node_pb2 as node_pb2
import generated.node_pb2_grpc as node_pb2_grpc
from modules.EncoderDecoder import EncoderDecoderManager
from modules.NetworkObservabilityTracker import NetworkObservabilityTracker
from util import status, logger
from server.dashboard_server import dashboard_server 

node_ports = 50052

logger = logger.Logger()
logger.set_logger_id('server')

class InferenceTask:
    def __init__(self, task_id: str, input_tensor):
        self.task_id = task_id
        self.input_tensor = input_tensor

class Server(server_pb2_grpc.ServerServicer):
    def __init__(self, node_config = [], server_config = {}):
        super().__init__()
        self.node_registry = {}  # Maps node IPs to their assigned model parts
        self.model_partitions_dir = "./model_parts"
        self.first_node_ip = None
        self.last_node_ip = None
        self.task_registry = {}
        self.task_track_counter = 0
        self.inference_queue = asyncio.Queue()
        self.network_ready_future = asyncio.Future()
        self.node_config = node_config 
        self.server_config = server_config
        self.network_observer = NetworkObservabilityTracker()
        self.sparsity_engine = SparsityEngine(network_observer=self.network_observer)
        self.encoderDecoder = EncoderDecoderManager(network_observer=self.network_observer, sparsity_engine=self.sparsity_engine)
        self.PROCESSES = multiprocessing.cpu_count() - 1

        if not os.path.exists(self.model_partitions_dir):
            os.makedirs(self.model_partitions_dir)

        # TODO Remove if possible
        self.assigned_node_config_index = 0 

    async def RegisterNode(self, request, context):
        """
        Register a new node and stream the assigned model part.
        """
        # This is to make sure the ports don't overlap when running with mininet
        # Not required in the final implementation
        # TODO: Remove
        global node_ports
        node_ports+=1
        node_port = node_ports

        peer_info = context.peer().split(":") # Example: context.peer() "ipv4:192.168.1.5:50432"
        new_node_ip = peer_info[1]  # Extract the IP address with port => 192.168.1.5:50432 
        new_node_ip += f":{node_port}"
        logger.log(f"Registering new node with IP: {new_node_ip}")

        if new_node_ip in self.node_registry:
            logger.log(f"IP is already registered: {new_node_ip}")
            yield server_pb2.RegisterResponse(
                status_code=status.IP_ALREADY_REGISTERED,
            )
            return

        # Allocate the next available model part
        model_parts = sorted([os.path.splitext(filename)[0] for filename in filter(lambda x: x.endswith('.onnx'), os.listdir(self.model_partitions_dir))])
        assigned_part = None

        for part in model_parts:
            if part not in self.node_registry.values():
                assigned_part = part
                break

        if not assigned_part:
            yield server_pb2.RegisterResponse(
                status_code=status.NO_MORE_MODEL_PARTS_TO_ASSIGN,
            )
            return

        # Register the new node
        self.node_registry[new_node_ip] = assigned_part
        logger.log(f"Node {new_node_ip} assigned model part: {assigned_part}")

        # Determine the initial next node
        if len(self.node_registry) == 1:
            self.first_node_ip = new_node_ip
            self.last_node_ip = new_node_ip

        # Read and stream the model part back to the new node
        model_part_path = os.path.join(self.model_partitions_dir, assigned_part + '.onnx')

        node_index = self.assigned_node_config_index 
        node_config = {}
        if node_index < len(self.node_config):
            node_config = self.node_config[node_index]
            self.assigned_node_config_index+=1

        try:
            with open(model_part_path, "rb") as f:
                while chunk := f.read(1024 * 1024):  # Stream 1MB chunks
                    yield server_pb2.RegisterResponse(
                        status_code=status.REGISTER_SUCCESS,
                        port=node_port,
                        prev_node=self.last_node_ip,
                        model_part_id=assigned_part,
                        chunk=chunk,
                        config=json.dumps(node_config)
                    )
        except FileNotFoundError:
            yield server_pb2.RegisterResponse(
                status_code=status.SERVER_ERROR,
                message="Error reading model part. Missing or corrupted resource"
            )
            return

    async def InformReady(self, request: server_pb2.ReadyRequest, context: grpc.RpcContext):
        peer_info = context.peer().split(":") # Example: context.peer() "ipv4:192.168.1.5:50432"
        ready_node_ip = peer_info[1]  # Extract the IP address with port => 192.168.1.5:50432 

        # Get the port from the request
        if(request.port == None):
            return  server_pb2.ReadyResponse(status_code=status.NODE_UPDATE_ERROR, message="Port is required")

        if(request.node_id == None): 
            return  server_pb2.ReadyResponse(status_code=status.NODE_UPDATE_ERROR, message="Node Id is required")
        
        node_id = request.node_id
        ready_node_ip += f":{request.port}"
        is_single_node = len(self.node_config) == 1
        logger.log(f"Node ready informed: {ready_node_ip}, is_single_node: {is_single_node}")

        # Early termination upon receiving the confirmation from the first node
        if(ready_node_ip == self.last_node_ip and is_single_node != True):
            logger.log(f"First node initialize informed: {ready_node_ip}")
            return server_pb2.ReadyResponse(status_code=status.NODE_READY_INFORM_SUCCESS) 

        current_last_node = self.last_node_ip
        self.last_node_ip = ready_node_ip
        logger.log(f"Last node ip updated: {self.last_node_ip}")

        # is_final_node = list(self.node_registry)[-1] == ready_node_ip and len or is_single_node 
        is_final_node = self.node_config[-1].get("node_id","") == node_id or is_single_node 
        
        if (is_single_node == False):
            logger.log(f"Connecting to current last to node: {current_last_node} -> {ready_node_ip}")
            async with grpc.aio.insecure_channel(current_last_node) as channel:
                stub = node_pb2_grpc.NodeServiceStub(channel)
                request = node_pb2.UpdateNextNodeRequest(next_node=ready_node_ip)
                response : node_pb2.UpdateNextNodeResponse = await stub.UpdateNextNode(request)

                if(response.status_code != status.NODE_UPDATE_SUCCESS): 
                    logger.error(f"Error updating the next node for ip:{current_last_node}\n{response.message}")
                    self.last_node_ip = current_last_node
                    logger.log(f"Reverted last_node_ip to {current_last_node}")
                    return server_pb2.ReadyResponse(status_code=status.NODE_UPDATE_ERROR, message=response.message) 
                    
        if(is_final_node):
            logger.log(f"Final Node Ready Received: Informing ({ready_node_ip}) to connect to the server")
            await self.inform_final_node_to_connect_to_coordinator(last_node_ip=ready_node_ip) 

        self.update_network_is_ready()
        # Update the last node
        return server_pb2.ReadyResponse(status_code=status.NODE_READY_INFORM_SUCCESS) 

    async def inform_final_node_to_connect_to_coordinator(self, last_node_ip: str):
        logger.log(f"Informing previous last node {self.last_node_ip} to connect to coordinator")
        async with grpc.aio.insecure_channel(last_node_ip) as channel:
            stub = node_pb2_grpc.NodeServiceStub(channel)
            request = node_pb2.UpdateNextNodeRequest()
            response : node_pb2.UpdateNextNodeResponse = await stub.UpdateNextNode(request)

            if(response.status_code != status.NODE_UPDATE_SUCCESS): 
                logger.log(f"Error updating the next node for ip:{last_node_ip}\n{response.message}")
                return server_pb2.ReadyResponse(status_code=status.NODE_UPDATE_ERROR, message=response.message) 

    # TODO : Remove 
    async def Hello(self, request, context):
        logger.log("Client says Hello")
        return server_pb2.Test() 

    async def Ping(self, request:server_pb2.ServerPingRequest, context: grpc.RpcContext) -> server_pb2.ServerPingResponse:
        return server_pb2.ServerPingResponse();

    def preprocess_image(self, input_bytes, target_size=(224, 224)):
        """
        Detailed image preprocessing with extensive debugging information
        
        Args:
        - input_bytes: Image bytes
        - target_size: Desired output image size (default 224x224)
        
        Returns:
        - Preprocessed image array
        """
        try:
            # Open and convert image to RGB
            img = Image.open(io.BytesIO(input_bytes)).convert('RGB')
            
            # Resize with aspect ratio preservation and center crop
            width, height = img.size
            
            # Calculate resize scaling
            scale = max(target_size[0] / width, target_size[1] / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize image
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Calculate crop coordinates for center crop
            left = (new_width - target_size[0]) // 2
            top = (new_height - target_size[1]) // 2
            right = left + target_size[0]
            bottom = top + target_size[1]
            
            # Perform center crop
            img = img.crop((left, top, right, bottom))
            
            # Convert to numpy array and print details
            img_array = np.array(img, dtype=np.float32)
            
            # Normalize to [0, 1]
            img_array = img_array / 255.0
            
            # Standardize using ImageNet mean and std
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_array = (img_array - mean) / std
            
            # Transpose to (C, H, W)
            img_array = img_array.transpose((2, 0, 1))
            
            # Add batch dimension
            img_array = img_array[np.newaxis, :, :, :]
            
            return np.ascontiguousarray(img_array)
        
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            raise 
         
    async def start_inference(self, input_tensor):
        task_id = self.task_track_counter

        try:
            self.task_track_counter += 1
            logger.log(f"Queuing new inference task with id: {task_id}") 

            if(self.task_registry.get(task_id, None) != None):
                logger.log(f"Task ids overlapped. {task_id}: is already in use.")
                return None

            # encoded_values, encoded_indices = self.encoderDecoder.encode(input_tensor)
            # compressed_tensor = self.compress(encoded_values, encoded_indices)
            # input_tensor = np.frombuffer(input_tensor, dtype=np.float32)
            input_shape = self.server_config.get('input_shape')
            # input_tensor = input_tensor.reshape(input_shape)
            logger.log('From Server shape:', input_tensor.shape, input_shape)
            tensor = self.encoderDecoder.encode('huffman', input_tensor)

            # Queue the task
            # inference_task = InferenceTask(task_id, compressed_tensor)
            inference_task = InferenceTask(task_id, tensor)

            # TODO : Should the put call be awaited?
            await self.inference_queue.put(inference_task)

            task = asyncio.Future()
            self.task_registry[task_id] = task 
            await asyncio.sleep(1)
            while(not task.done()):
                await asyncio.sleep(.5)

            print(f"result: {task.done()}")
            result = await task 
            return  result

        except Exception as e:
            logger.log(f"There was error processing inference task: {task_id}")
            logger.log(f"Err: {e}")
            traceback.logger.log_exc()

    async def StartInference(self, request: server_pb2.StartInferenceRequest, context):
        task_id = self.task_track_counter

        try:
            input_tensor = request.input_tensor
            if(input_tensor == None or input_tensor == []):
                message=f"Inference Error: Input tensor not provided. Task ID: {task_id}"
                return server_pb2.StartInferenceResponse(status_code=status.BAD_REQUEST, message=message)

            self.task_track_counter += 1
            logger.log(f"Inference request received! Queuing new inference task with id: {task_id}") 

            if(self.task_registry.get(task_id, None) != None):
                logger.log(f"Task ids overlapped. {task_id}: is already in use.")
                return server_pb2.StartInferenceResponse(status_code=status.SERVER_ERROR, message="An internal server error occurred. Please try again.")

            # encoded_values, encoded_indices = self.encoderDecoder.encode(input_tensor)
            # compressed_tensor = self.compress(encoded_values, encoded_indices)
            input_tensor = np.frombuffer(input_tensor, dtype=np.float32)
            input_shape = self.server_config.get('input_shape')
            input_tensor = input_tensor.reshape(input_shape)
            logger.log('From Server shape:', input_tensor.shape, input_shape)
            tensor = self.encoderDecoder.encode('huffman', input_tensor)

            # Queue the task
            # inference_task = InferenceTask(task_id, compressed_tensor)
            inference_task = InferenceTask(task_id, tensor)

            # TODO : Should the put call be awaited?
            await self.inference_queue.put(inference_task)

            task = asyncio.Future()
            self.task_registry[task_id] = task 
            result = await task 
            result = result.tobytes()

            return server_pb2.StartInferenceResponse(status_code=status.INFERENCE_SUCCESS, result=result)
        except:
            logger.log(f"There was error processing inference task: {task_id}")
            traceback.logger.log_exc()

    async def EndInference(self, request: server_pb2.EndInferenceRequest, context):
        task_id = request.task_id
        if(task_id == None):
            logger.log("Not processing end inference request since task_id is not defined")
            return server_pb2.EndInferenceResponse(status_code=status.BAD_REQUEST, message="Task id is not defined")

        task = self.task_registry.get(task_id, None)
        if(task == None): 
            logger.log(f"Received task id is not valid {task_id}")
            return server_pb2.EndInferenceResponse(status_code=status.SERVER_ERROR, message="Invalid task id.") 

        result = request.result

        if(result == None): 
            logger.log(f"Invalid inference end request without results received for  {task_id}")
            return server_pb2.EndInferenceResponse(status_code=status.SERVER_ERROR, message="Invalid task id.") 

        result = self.encoderDecoder.decode(result)

        task.set_result(result)
        logger.log(f"task_id: {task_id} future has been resolved {task.done()}")

        return server_pb2.EndInferenceResponse(status_code=status.INFERENCE_END_ACCEPTED) 

    async def test(self):
        await asyncio.sleep(5)
        return

    def update_network_is_ready(self):
       current_nodes_count = len(self.node_registry) 
       required_nodes_count = len(self.node_config)
       logger.log(f"is ready {current_nodes_count} == {required_nodes_count}")
       self.is_network_ready = current_nodes_count == required_nodes_count

       if(self.is_network_ready):
           self.network_ready_future.set_result(True)

    async def wait_for_network_ready(self):
        return await self.network_ready_future

    async def process_inference_queue(self):
        """Continuously processes inference tasks from the queue."""
        while True:
            inference_task = await self.inference_queue.get()  # Wait until a task is available
            logger.log(f"Processing inference task : {inference_task.task_id}")
            await self.send_for_inference(inference_task)
            self.inference_queue.task_done()  # Mark the task as done

    async def GetInferenceMetrics(self, request, context):
        logger.log(f"NodeServer: Inference request received")
        egress_bytes= 0
        for node_ip in self.node_registry:
            metrics_task = await self.get_inference_metrics_from_node(node_ip=node_ip)
            egress_bytes += metrics_task.egress_bytes
            logger.log(f"{node_ip}: {egress_bytes}B")
        
        return server_pb2.InferenceMetricsResponse(egress_bytes=egress_bytes)
    
    async def get_inference_metrics_from_node(self, node_ip) -> node_pb2.NodeInferenceMetricsResponse:
        async with grpc.aio.insecure_channel(node_ip) as channel:
            stub = node_pb2_grpc.NodeServiceStub(channel)
            request = node_pb2.NodeInferenceMetricsRequest()
            response = await stub.GetInferenceMetrics(request)
            return response

    async def run(self):
        """Start the gRPC server and inference queue processing."""
        await self.wait_for_network_ready()

        # Start the inference queue processor as a background task
        await self.process_inference_queue()
        

    async def send_for_inference(self, inference_task: InferenceTask):
        try:
            input_tensor = inference_task.input_tensor
            task_id = inference_task.task_id

            async with grpc.aio.insecure_channel(self.first_node_ip) as channel:
                node_stub = node_pb2_grpc.NodeServiceStub(channel)

                request = node_pb2.InferenceRequest(input_tensor=input_tensor) 
                response: node_pb2.InferenceResponse = await node_stub.Infer(request)

                if(response.status_code != status.INFERENCE_ACCEPTED):
                    logger.log(f"There was an error starting inference Task ID: {response.task_id}, Status: {response.status_code}, Message: {response.message}")
                    return server_pb2.StartInferenceResponse(status_code=status.SERVER_ERROR, message="There was an error starting inference")  

                logger.log(f"Task ID [{task_id}]: Inference accepted")
        except:
            traceback.logger.log_exc() 

    async def Ping(self, request: server_pb2.ServerPingRequest, context: grpc.RpcContext) -> server_pb2.ServerPingResponse:
        return server_pb2.ServerPingResponse()
    

async def serve():
    """
    Start the gRPC server for the coordinator.
    """
    port = "50051"
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    with open('config.json') as f:
            config = json.load(f)
            server_config = config.get('server_config')
            node_config = config.get('node_config')

    port = server_config.get('grpc_server_port', 55001)
    dashboard_server_port = server_config.get('dashboard_server_port', 4010)
    coordinator_node = Server(node_config=node_config, server_config=server_config)

    # GRPC Server
    server_pb2_grpc.add_ServerServicer_to_server(coordinator_node, server)
    server.add_insecure_port("0.0.0.0:" + str(port))
    logger.log(f"Coordinator node started, listening on port {port}")

    # Dashboard server
    logger.log(f"Starting dashboard server")
    def run_flask_server():
        dashboard_server.start_server(coordinator_node, {"port": dashboard_server_port})

    flask_thread = threading.Thread(
        target=run_flask_server, 
        daemon=True
    )
    flask_thread.start()

    try:
        await server.start()
        await coordinator_node.run()
    except:
        # Shutting down the server if there's a fatal error
        logger.log(f"Terminating coordinator node due to a fatal error. Please contact support.")
        await server.stop(grace=None)
        return

if __name__ == "__main__":
    asyncio.run(serve())