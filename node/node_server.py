from concurrent import futures
import asyncio
import grpc
import onnxruntime as ort

import generated.node_pb2 as node_pb2
import generated.node_pb2_grpc as node_pb2_grpc
from util import to_numpy, status
import traceback
import numpy as np

class NodeServer(node_pb2_grpc.NodeServiceServicer): 
    def __init__(self, model_path, node: any):
        self.model_path = model_path
        self.node = node
        self.current_task_id = 0

        # Loading the ONNX Model partition.
        # Assuming the model part is downloaded and verify when the node server
        # is instantiated.
        print("Loading model for inference")
        self.ort_session = ort.InferenceSession(model_path)
        print("Model successfully loaded for inference")

        # Store the input shape from the model's input metadata
        input_metadata = self.ort_session.get_inputs()[0]
        self.input_name = input_metadata.name
        self.input_shape = input_metadata.shape  # e.g., [batch_size, 3, 32, 32]
        print(f"Model input shape: {self.input_shape}")

        super().__init__()


     
    async def Infer(self, request: node_pb2.InferenceRequest, context: grpc.RpcContext):
        """
        Perform inference using the assigned model part.
        """
        model_part_id = self.node.model_part_id

        try:
            input = request.input_tensor
            task_id = request.task_id

            if not input:
                message = f"Task ID: {task_id} | Input tensor is not provided"
                return node_pb2.InferenceResponse(
                    status_code=status.BAD_REQUEST,
                    message=message
                )

            if not task_id:
                task_id = self.current_task_id
                self.current_task_id += 1

            print(f"Inference task received: {task_id}")
            print(f"Input tensor type: {type(input)}, length: {len(input)}")

            # # Convert byte stream to NumPy array
            # input_array = np.frombuffer(input, dtype=np.float32)
            # reshaped_input = input_array.reshape(self.input_shape)  # Skip batch dimension if dynamic
            # # reshaped_input = input_array  # Skip batch dimension if dynamic
            # print(f"Reshaped input: {reshaped_input.shape}")
            # Decode and reconstruct the sparse tensor
            reconstructed_activations = self.node.decode(input, self.input_shape)

            # Convert to torch tensor
            reconstructed_activations = np.asanyarray(reconstructed_activations, dtype=np.float32)

            # Load the ONNX model and perform the inference
            ort_session = self.ort_session
            ort_inputs = {ort_session.get_inputs()[0].name: reconstructed_activations}

            # Schedule an inference task
            task = self.async_inference(task_id=task_id, ort_inputs=ort_inputs)
            asyncio.create_task(task)

            message = f"[id:{model_part_id}] Inference accepted for task_id: {task_id}"

            return node_pb2.InferenceResponse(
                status_code=status.INFERENCE_ACCEPTED,
                message=message,
                task_id=task_id
            )

        except Exception as e:
            message = f"[id: {model_part_id}] Inference failed: {str(e)}"
            print(message)
            traceback.print_exc()
            context.set_details(message)
            context.set_code(grpc.StatusCode.INTERNAL)
            return node_pb2.InferenceResponse(status_code=status.SERVER_ERROR)

    async def UpdateNextNode(self, request: node_pb2.UpdateNextNodeRequest, context: grpc.RpcContext):
        """
        Update the next node in the network.
        """
        try:
            self.node.next_node = request.next_node
            print(f"Next node updated to: {self.node.next_node}")

            return node_pb2.UpdateNextNodeResponse(
                status_code=status.NODE_UPDATE_SUCCESS,
            )
        except Exception as e:
            message=f"There was an error updating next node reference {e}"
            print(message)
            return node_pb2.UpdateNextNodeResponse(
                status_code=status.NODE_UPDATE_ERROR,
                message=message
            ) 

    async def async_inference(self, task_id: int, ort_inputs):
        # Run inference in the ONNX session asynchronously
        loop = asyncio.get_event_loop()
        output_tensor = await loop.run_in_executor(None, lambda: self.ort_session.run(None, ort_inputs))

        print(f'[taskId: {task_id}]output_tensor', output_tensor[0].shape)
        # Propagate to the next node
        await self.node.forward_to_next_node(task_id=task_id, input_tensor=output_tensor[0])


async def main(model_path, node, port=55001): 
    """
    Start the gRPC server for the node.
    """
    # Initialize the gRPC server
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    node_service = NodeServer(model_path=model_path, node=node)
    node_pb2_grpc.add_NodeServiceServicer_to_server(node_service, server)
    server.add_insecure_port(f"[::]:{port}")
    await server.start()

    print(f"Node server started on port {port}")
    await server.wait_for_termination()    


if __name__ == '__main__':
    main()
