from concurrent import futures
import asyncio
import grpc
import onnxruntime as ort


import generated.node_pb2 as node_pb2
import generated.node_pb2_grpc as node_pb2_grpc
from util import to_numpy, status

class NodeServer(node_pb2_grpc.NodeServiceServicer): 
    def __init__(self, model_path, node: any):
        self.model_path = model_path
        self.node = node

        # Loading the ONNX Model partition.
        # Assuming the model part is downloaded and verify when the node server
        # is instantiated.
        print("Loading model for inference")
        self.ort_session = ort.InferenceSession(model_path)
        print("Model successfully loaded for inference")
        super().__init__()


    async def Infer(self, request: node_pb2.InferenceRequest, context: grpc.RpcContext):
        """
        Perform inference using the assigned model part.
        """
        model_part_id = self.node.model_part_id

        try:
            input = request.input_tensor
            task_id = request.task_id
            
            if(input == None):
               return

            if(task_id == None):
                task_id = self.current_task_id
                self.current_task_id +=1

            # Load the onnx and perform the inference 
            ort_session = self.ort_session
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}

            # Schedule an inference task
            self.async_inference(task_id=task_id, ort_inputs=ort_inputs)
            
            message = f"[id:{model_part_id}] Inference accepted for task_id: {task_id}"

            return node_pb2.InferenceResponse(
                status_code=status.INFERENCE_ACCEPTED,
                message="",
                model_part_id=model_part_id, 
                task_id=task_id
            )
        except Exception as e:
            message = f"[id:{model_part_id}] Inference failed: {str(e)}"
            print(message)
            context.set_details(message)
            context.set_code(grpc.StatusCode.INTERNAL)
            return node_pb2.InferenceResponse(status_code=status.SERVER_ERROR) 


    async def UpdateNextNode(self, request: node_pb2.UpdateNextNodeRequest, context: grpc.RpcContext):
        """
        Update the next node in the network.
        """
        try:
            self.node.next_node = request.next_node_address
            print(f"Next node updated to: {self.node.next_node}")

            return node_pb2.UpdateNextNodeResponse(
                status_code=status.NODE_UPDATE_SUCCESS,
            )
        except Exception as e:
            return node_pb2.UpdateNextNodeResponse(
                status_code=status.NODE_UPDATE_ERROR,
                message=str(e)
            ) 

    async def async_inference(self, task_id: int, ort_inputs):
        # Run inference in the ONNX session asynchronously
        loop = asyncio.get_event_loop()
        output_tensor = await loop.run_in_executor(None, lambda: self.ort_session.run(None, ort_inputs))

        # Propagate to the next node
        await self.node.forward_to_next_node(task_id=task_id, input_tensor=output_tensor)


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