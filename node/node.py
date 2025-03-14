import os
import sys
import grpc
import asyncio
import numpy as np
import onnxruntime as ort
import time
import json

from util import status, logger
logger = logger.logger

module_path = os.path.abspath('../')
sys.path.insert(0, module_path)

from modules.SparsityEngine import SparsityEngine, SparsityEngineDefaults
from modules.EncoderDecoder import EncoderDecoderManager
from modules.NetworkObservabilityTracker import NetworkObservabilityTracker
from node.node_server import main as serve
from generated.node_pb2 import InferenceRequest, InferenceResponse, PingRequest
from generated.node_pb2_grpc import NodeServiceStub
from generated.server_pb2_grpc import ServerStub 
from generated.server_pb2 import EndInferenceRequest, RegisterRequest, RegisterResponse, Test, ReadyRequest, ServerPingRequest

class NodeNotConnected(Exception):
    def __init__(self, task_id, port, *args):
        logger.log(f"NodeNotConnected: task_id {task_id} {port}")
        super().__init__(*args)

class Node:
    def __init__(self, coordinator_address: str, node_ip: str):
        """
        Initialize the node object.
        """
        self.coordinator_address = coordinator_address  # Coordinator gRPC address (e.g., "192.168.1.1:50051")
        self.node_ip = node_ip                          # Node's IP address
        self.prev_node = None                           # Predecessor node
        self.model_part_id = None
        self.port = None
        self.network_observability = NetworkObservabilityTracker()
        self.sparsity_engine = SparsityEngine(network_observer=self.network_observability)
        self.encoderDecoder = EncoderDecoderManager(network_observer=self.network_observability, sparsity_engine=self.sparsity_engine)
        self.connection = asyncio.Future() 
        self.next_node = asyncio.Future() 
        self.encoding_strategy = "adaptive"
        self.config = {}

    async def register(self, stub: ServerStub):
        """
        Register the node with the coordination server and handle streamed model parts.
        """
        request = RegisterRequest()

        model_file_path = f"model_part_{self.model_part_id}.onnx"
        with open(model_file_path, "wb") as model_file:
            async for response in stub.RegisterNode(request):
                if response.status_code != status.REGISTER_SUCCESS:
                    message = f"Failed to register node: {response.message}"
                    print(message)
                    raise RuntimeError(message) 

                if response.chunk:
                    model_file.write(response.chunk)

                if response.prev_node:
                    self.prev_node = response.prev_node

                if response.first_node:
                    self.first_node = response.first_node

                if response.model_part_id:
                    self.model_part_id = response.model_part_id
                
                if response.port:
                    self.port = response.port
                    if self.port == 50055:
                        self.encoding_strategy = "huffman"
                
                if response.config:
                    self.config = json.loads(response.config)
                    self.load_config()

        logger.set_logger_id(self.port) 

        # Rename the downloaded file
        updated_model_file_path = f"model_part_{self.model_part_id}.onnx"
        os.rename(model_file_path, updated_model_file_path)
        logger.log(f"Model part saved at {updated_model_file_path}")
        
        # Loading the ONNX Model partition.
        logger.log("Loading model for inference")
        self.ort_session = ort.InferenceSession(updated_model_file_path)
        logger.log("Model successfully loaded for inference")

        # Store the input shape from the model's input metadata
        input_metadata = self.ort_session.get_inputs()[0]
        self.input_name = input_metadata.name
        self.input_shape = input_metadata.shape
        logger.log(f"Model input shape: {self.input_shape}, {self.config}")

        return (updated_model_file_path, self.port, self.config.get("node_id", ""))

    def load_config(self):
        node_config = self.config
        self.encoding_strategy = node_config.get('encoding_strategy', self.encoding_strategy)
        
        # Update the defaults
        self.sparsity_engine.update_defaults_from_node_config(node_config=node_config)


    def infer(self, task_id, input):
        # Update ingress metrics
        self.network_observability.update_ingress_metrics(input)

        # Convert to torch tensor
        reconstructed_activations = self.encoderDecoder.decode(input)

        # Load the ONNX model and perform the inference
        ort_session = self.ort_session
        ort_inputs = {ort_session.get_inputs()[0].name: reconstructed_activations.astype(np.float32)}

        # Schedule an inference task
        task = self.async_inference(task_id=task_id, ort_inputs=ort_inputs)
        asyncio.create_task(task)

    async def async_inference(self, task_id: int, ort_inputs):
        # Run inference in the ONNX session asynchronously
        loop = asyncio.get_event_loop()
        output_tensor = await loop.run_in_executor(None, lambda: self.ort_session.run(None, ort_inputs))

        logger.log(f'[taskId: {task_id}] output_tensor', output_tensor[0].shape)

        # Propagate to the next node
        await self.forward_to_next_node(task_id=task_id, input_tensor=output_tensor[0])

    async def forward_to_next_node(self, task_id: int, input_tensor):
        connection = await self.connection 
        next_node = await self.next_node

        # Encode the output tensor
        tensor = self.encoderDecoder.encode(self.encoding_strategy, input_tensor)

        # Update egress metrics
        self.network_observability.update_egress_metrics(tensor)

        logger.log(f"Forwarding to next node {next_node}")
        if(next_node == None or next_node == ""):
            await self.finish_inference(task_id=task_id, result=tensor) 
            print(f"Inference complete {task_id}")
            return
        
        # TODO: Reuse the channel
        stub = NodeServiceStub(connection)

        request = InferenceRequest(
            # next_model_part_id=self.model_part_id+1, # This is not ideal
            task_id= task_id, 
            input_tensor=tensor
        )

        message = f"[{task_id}]: Forwarding task to next node; {self.next_node}"
        logger.log(message)
        response: InferenceResponse = await stub.Infer(request) 

        if(response.status_code != status.INFERENCE_ACCEPTED):
            message = f"[{task_id}]: There was an error forwarding inference task to: {self.next_node}"
            logger.log(message)

        logger.log(f"Inference successfully propagated to next node with status: {response.status_code}")

    async def finish_inference(self, task_id: int, result):
        connection = await self.connection

        # Assuming the self.connection is assigned with a aio channel prior to reaching this
        stub  = ServerStub(connection)
        request = EndInferenceRequest(result=result, task_id=task_id)
        response = await stub.EndInference(request)
        
        if(response.status_code != status.INFERENCE_END_ACCEPTED):
            message = f"There was an error finalizing the inference end, {response.status_code}" 
            logger.log(message)

    async def ping_in_background(self, ping_timeout: int = 3):
        await self.ping_next_node()

        while True:
            await asyncio.sleep(ping_timeout)
            await self.ping_next_node()

    async def ping_next_node(self):
        next_node = await self.next_node
        connection = await self.connection
        
        start_time = time.time_ns()

        if(next_node == None or next_node == ''):
            # Send the ping request to server stub
            stub = ServerStub(connection)
            request = ServerPingRequest()
            await stub.Ping(request)

        else:
            # Send the ping request to node stub
            stub = NodeServiceStub(connection)
            request = PingRequest()
            await stub.Ping(request)

        end_time = time.time_ns()
        rtt = (end_time - start_time) /1000_000 # Convert to milliseconds
        logger.log(f"RTT updated: {rtt}ms")

        self.network_observability.update_rtt_with_next_node(rtt=rtt)

    async def connect_to_coordinator_node(self):
        logger.log(f"Initializing connection to coordinator node: {self.coordinator_address}")
        channel = grpc.aio.insecure_channel(self.coordinator_address)
        await channel.channel_ready()
        self.connection.set_result(channel) 
        logger.log("Channel initialized", self.connection)

    async def connect_to_next_node(self):
        next_node = await self.next_node
        logger.log(f"Initializing connection to next node: {next_node}")
        channel = grpc.aio.insecure_channel(next_node)
        await channel.channel_ready()
        self.connection.set_result(channel)
        logger.log("Channel initialized")

async def main(coordinator_ip="127.0.0.1:50051"):
    try: 
        print(f'Initializing node...')
        node = Node(coordinator_address=coordinator_ip, node_ip="")

        async with grpc.aio.insecure_channel(node.coordinator_address) as channel:
            stub = ServerStub(channel)

            temp_request = Test()
            await stub.Hello(temp_request)

            print('Registering node...')
            (model_file_path, port, node_id) = await node.register(stub)
            print('Successfully registered node...')

            # Start the gRPC server to allow other nodes to send input tensors for inference
            logger.log('Starting the gRPC server...')
            server = await serve(model_path=model_file_path, node=node, port=port)

            # The server registers the ip of the current node as the last ip
            # Could be an issue - verify
            logger.log(f'Informing node is ready {node.model_part_id}')
            readyRequest = ReadyRequest(port=port, node_id=node_id)
            await stub.InformReady(readyRequest)

            # Wait for the next node to be initialized 
            next_node = await node.next_node

            if(next_node == None or next_node == ""):
                logger.log("Next node is None")
                await node.connect_to_coordinator_node()
            else:
                logger.log(f"Connecting to next node: {next_node}")
                await node.connect_to_next_node()

            logger.log('Connected to next node')

            # Start the ping in background
            # Assuming fire and forget in create_task
            logger.log('Starting ping task')
            await asyncio.create_task(node.ping_in_background())
            logger.log('Started ping task in the background')

            # Block till server termination
            await server.wait_for_termination()

            # TODO : This is not an ideal approach.. please check for better solutions
            # await asyncio.gather(server_start_async, connection_task, ping_task)
            node_connection = await node.connection
            if(node_connection != None):
                node_connection.close()
                node.connection = None
            
    except RuntimeError as e:
        print("Couldn't initialise the node")
        print(e)


if __name__ == '__main__':
    asyncio.run(main())
