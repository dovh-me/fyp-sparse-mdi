import sys
import os
import asyncio

from util import status

module_path = os.path.abspath('../')
sys.path.insert(0, module_path)

from concurrent import futures
import grpc
import generated.server_pb2 as server_pb2
import generated.server_pb2_grpc as server_pb2_grpc
import generated.node_pb2 as node_pb2
import generated.node_pb2_grpc as node_pb2_grpc

node_ports = 50052

class Server(server_pb2_grpc.ServerServicer):
    def __init__(self):
        super().__init__()
        self.node_registry = {}  # Maps node IPs to their assigned model parts
        self.model_partitions_dir = "./model_parts"
        self.first_node_ip = None
        self.last_node_ip = None
        self.task_registry = {}
        self.task_track_counter = 0

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
        print(f"Registering new node with IP: {new_node_ip}")

        if new_node_ip in self.node_registry:
            print(f"IP is already registered: {new_node_ip}")
            yield server_pb2.RegisterResponse(
                status_code=status.IP_ALREADY_REGISTERED,
            )
            return

        # Allocate the next available model part
        model_parts = sorted([os.path.splitext(filename)[0] for filename in os.listdir(self.model_partitions_dir)])
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
        print(f"Node {new_node_ip} assigned model part: {assigned_part}")

        # Determine the initial next node
        if len(self.node_registry) == 1:
            self.first_node_ip = new_node_ip
            self.last_node_ip = new_node_ip

        # Read and stream the model part back to the new node
        model_part_path = os.path.join(self.model_partitions_dir, assigned_part + '.onnx')

        try:
            with open(model_part_path, "rb") as f:
                while chunk := f.read(1024 * 1024):  # Stream 1MB chunks
                    yield server_pb2.RegisterResponse(
                        status_code=status.REGISTER_SUCCESS,
                        port=node_port,
                        prev_node=self.last_node_ip,
                        model_part_id=assigned_part,
                        chunk=chunk
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
        
        ready_node_ip += f":{request.port}"
        print(f"Node ready informed: {ready_node_ip}")

        # Early termination upon receiving the confirmation from the first node
        if(ready_node_ip == self.last_node_ip):
            print(f"Last node is already ip: {ready_node_ip}")
            return server_pb2.ReadyResponse(status_code=status.NODE_READY_INFORM_SUCCESS) 
        
        current_last_node = self.last_node_ip
        self.last_node_ip = ready_node_ip
        print(f"Last node ip updated: {self.last_node_ip}")

        print(f"Informing node of the last_node_ip update: {current_last_node}")
        async with grpc.aio.insecure_channel(current_last_node) as channel:
            stub = node_pb2_grpc.NodeServiceStub(channel)
            request = node_pb2.UpdateNextNodeRequest(next_node=self.last_node_ip)
            response : node_pb2.UpdateNextNodeResponse = await stub.UpdateNextNode(request)

            if(response.status_code != status.NODE_UPDATE_SUCCESS): 
                print(f"Error updating the next node for ip:{current_last_node}\n{response.message}")
                self.last_node_ip = current_last_node
                print(f"Reverted last_node_ip to {current_last_node}")
                return server_pb2.ReadyResponse(status_code=status.NODE_UPDATE_ERROR, message=response.message) 

        # Update the last node
        return server_pb2.ReadyResponse(status_code=status.NODE_READY_INFORM_SUCCESS) 

    async def Hello(self, request, context):
        print("Client says Hello")
        return server_pb2.Test() 


    async def StartInference(self, request: server_pb2.StartInferenceRequest, context):
        task_id = self.task_track_counter
        self.task_track_counter += 1
        print(f"Inference request received! Starting new inference task with id: {task_id}") 

        if(self.task_registry.get(task_id) != None):
            print(f"Task ids overlapped. {task_id}: is already in use.")
            return server_pb2.StartInferenceResponse(status_code=status.SERVER_ERROR, message="An internal server error occurred. Please try again.")

        input_tensor = request.input_tensor
        if(input_tensor == None):
            message=f"Inference Error: Input tensor is not provided for the inference task. Task ID: {task_id}"
            return server_pb2.StartInferenceResponse(status_code=status.BAD_REQUEST, message=message)

        async with grpc.aio.insecure_channel(self.first_node_ip) as channel:
            node_stub = node_pb2_grpc.NodeServiceStub(channel)
            request = node_pb2.InferenceRequest(input_tensor=input_tensor) 

            response: node_pb2.InferenceResponse = await node_stub.Infer(request)

            if(response.status_code != status.INFERENCE_ACCEPTED):
                print(f"There was an error starting inference Task ID: {response.task_id}, Status: {response.status_code}, Message: {response.message}")
                return server_pb2.StartInferenceResponse(status_code=status.SERVER_ERROR, message="There was an error starting inference")  

            print(f"Task ID [{task_id}]: Inference accepted")

            task = asyncio.Future()
            self.task_registry.set(task_id, task) 

            print(f"Waiting for inference result: Task ID: {task_id}")
            result = await task 
        return server_pb2.StartInferenceResponse(status_code=status.INFERENCE_ACCEPTED, result=result)

    async def EndInference(self, request: server_pb2.EndInferenceRequest, context):
        task_id = request.task_id
        if(task_id == None):
            print("Not processing end inference request since task_id is not defined")
            return server_pb2.EndInferenceResponse(status_code=status.BAD_REQUEST, message="Task id is not defined")

        task = self.task_registry.get()

        if(task == None): 
            print(f"Received task id is not valid {task_id}")
            return server_pb2.EndInferenceResponse(status_code=status.SERVER_ERROR, message="Invalid task id.") 

        result = request.result
        if(result == None): 
            print(f"Invalid inference end request without results received for  {task_id}")
            return server_pb2.EndInferenceResponse(status_code=status.SERVER_ERROR, message="Invalid task id.") 

        print(f"task_id: {task_id} future has been resolved")
        task.set_result(result)

        return server_pb2.EndInferenceResponse(status_code=status.INFERENCE_END_ACCEPTED) 

async def serve():
    """
    Start the gRPC server for the coordinator.
    """
    port = "50051"
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    server_pb2_grpc.add_ServerServicer_to_server(Server(), server)
    server.add_insecure_port("[::]:" + port)
    print(f"Coordinator server started, listening on port {port}")
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())