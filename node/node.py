import os
import sys
import grpc
import asyncio
import struct
import numpy as np
import zlib

from util import status, top_k_sparsify

module_path = os.path.abspath('../')
sys.path.insert(0, module_path)

from modules.EncoderDecoder import EncoderDecoderManager
from node.node_server import main as serve
from generated.node_pb2 import InferenceRequest, InferenceResponse
from generated.node_pb2_grpc import NodeServiceStub
from generated.server_pb2_grpc import ServerStub 
from generated.server_pb2 import EndInferenceRequest, RegisterRequest, Test, ReadyRequest

class Node:
    def __init__(self, coordinator_address: str, node_ip: str):
        """
        Initialize the node object.
        """
        self.coordinator_address = coordinator_address  # Coordinator gRPC address (e.g., "192.168.1.1:50051")
        self.node_ip = node_ip                          # Node's IP address
        self.prev_node = None                           # Predecessor node
        self.next_node = None
        self.model_part_id = None
        self.port = None
        self.encoderDecoder = EncoderDecoderManager()


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

        # Rename the downloaded file
        updated_model_file_path = f"model_part_{self.model_part_id}.onnx"
        os.rename(model_file_path, updated_model_file_path)

        print(f"Model part saved at {updated_model_file_path}")
        print(f"Next node: {self.next_node}")

        return (updated_model_file_path, self.port)

    async def forward_to_next_node(self, task_id: int, input_tensor):
        # Apply sparsification
        # values, indices = top_k_sparsify(input_tensor, k=1000) 

        # # Encode values and indices
        # encoded_values, delta_indices = self.encode(values, indices)

        # # Compress the encoded data
        # compressed_data = self.compress(encoded_values, delta_indices)

        # # Convert to bytes
        # input_tensor = input_tensor.tobytes()

        tensor = self.encoderDecoder.encode('sparse', input_tensor)

        if(self.next_node == None):
            await self.finish_inference(task_id=task_id, result=tensor) 
            return
        
        # TODO: Reuse the channel
        async with grpc.aio.insecure_channel(self.next_node) as channel:
            stub = NodeServiceStub(channel)

            request = InferenceRequest(
                # next_model_part_id=self.model_part_id+1, # This is not ideal
                task_id= task_id, 
                input_tensor=tensor
            )

            message = f"[{task_id}]: Forwarding task to next node; {self.next_node}"
            print(message)

            response: InferenceResponse = await stub.Infer(request) 

            if(response.status_code != status.INFERENCE_ACCEPTED):
                message = f"[{task_id}]: There was an error forwarding inference task to: {self.next_node}"
                print(message)

            print(f"Inference successfully propagated to next node with status: {response.status_code}")

    async def finish_inference(self, task_id: int, result):
        async with grpc.aio.insecure_channel(self.coordinator_address) as channel:
            stub  = ServerStub(channel)
            request = EndInferenceRequest(result=result, task_id=task_id)
            response = await stub.EndInference(request)
            
            if(response.status_code != status.INFERENCE_END_ACCEPTED):
                message = f"There was an error finalizing the inference end, {response.status_code}" 

async def main():
    try: 
        print(f'Initializing node...')
        node = Node(coordinator_address="127.0.0.1:50051", node_ip="")

        async with grpc.aio.insecure_channel(node.coordinator_address) as channel:
            stub = ServerStub(channel)

            temp_request = Test()
            await stub.Hello(temp_request)

            print('Registering node...')
            (model_file_path, port) = await node.register(stub)
            print('Successfully registered node...')

            # Start the gRPC server to allow other nodes to send input tensors for inference
            print('Starting the gRPC server...')
            server_start_async = serve(model_path=model_file_path, node=node, port=port)

            # The server registers the ip of the current node as the last ip
            # Could be an issue - verify
            print(f'Informing node is ready {node.model_part_id}')
            readyRequest = ReadyRequest(port=port)
            await stub.InformReady(readyRequest)

            # # TODO : This is not an ideal approach.. please check for better solutions
            await asyncio.gather(server_start_async)
            
    except RuntimeError as e:
        print("Couldn't initialise the node")
        print(e)


if __name__ == '__main__':
    asyncio.run(main())
