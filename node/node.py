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

    def encode(self, values, indices):
        """
        Encode the values and indices for efficient transmission.
        - No quantization applied; values remain as float32.
        """
        # Use float32 for values and int32 for indices
        values = values.astype(np.float32)
        delta_indices = np.diff(np.insert(indices, 0, 0)).astype(np.int32)

        return values, delta_indices

    def compress(self, values, indices):
        """
        Compress the encoded data using zlib.
        """
        # Serialize data for compression
        serialized_data = struct.pack(f"{len(values)}f", *values) + \
                          struct.pack(f"{len(indices)}i", *indices)
        compressed_data = zlib.compress(serialized_data)

        return compressed_data

    def decompress(self, compressed_data):
        """
        Decompress the zlib-compressed data.
        """
        decompressed_data = zlib.decompress(compressed_data)
        return decompressed_data

    def decode(self, compressed_data, original_shape):
        """
        Decode and reconstruct the sparse tensor from compressed data.
        """
        # Decompress the data
        decompressed_data = self.decompress(compressed_data)

        # Deserialize the values and indices
        num_values = len(decompressed_data) // 8  # 4 bytes per value + 4 bytes per index
        values = struct.unpack(f"{num_values}f", decompressed_data[:4 * num_values])
        indices = struct.unpack(f"{num_values}i", decompressed_data[4 * num_values:])

        # Convert to numpy arrays
        values = np.array(values, dtype=np.float32)
        indices = np.cumsum(indices).astype(np.int32)  # Reconstruct original indices

        # Reconstruct the sparse tensor
        sparse_tensor = np.zeros(np.prod(original_shape), dtype=np.float32)
        sparse_tensor[indices] = values
        return sparse_tensor.reshape(original_shape)

    async def forward_to_next_node(self, task_id: int, input_tensor):
        # Apply sparsification
        values, indices = top_k_sparsify(input_tensor, k=1000) 

        

        # Encode values and indices
        encoded_values, delta_indices = self.encode(values, indices)

        # Compress the encoded data
        compressed_data = self.compress(encoded_values, delta_indices)

        # Convert to bytes
        # input_tensor = input_tensor.tobytes()

        if(self.next_node == None):
            await self.finish_inference(task_id=task_id, result=compressed_data) 
            return
        
        # TODO: Reuse the channel
        async with grpc.aio.insecure_channel(self.next_node) as channel:
            stub = NodeServiceStub(channel)

            request = InferenceRequest(
                # next_model_part_id=self.model_part_id+1, # This is not ideal
                task_id= task_id, 
                input_tensor=input_tensor
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
                message = f"There was an error finalising the inference end, {response.status_code}" 


    
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
            inform_ready_async = await stub.InformReady(readyRequest)

            # # TODO : This is not an idea approach.. please check for better solutions
            await asyncio.gather(server_start_async, 
                        #    inform_ready_async
                           )
            
    except RuntimeError as e:
        print("Couldn't initialise the node")
        print(e)


if __name__ == '__main__':
    main()
