import sys
import os
import asyncio

from util import status

module_path = os.path.abspath('../')
sys.path.insert(0, module_path)

from concurrent import futures
import gdown
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

    async def RegisterNode(self, request, context):
        """
        Register a new node and stream the assigned model part.
        """
        peer_info = context.peer().split(":") # Example: context.peer() "ipv4:192.168.1.5:50432"
        new_node_ip = peer_info[1]  # Extract the IP address with port => 192.168.1.5:50432 
        print(f"Registering new node with IP: {new_node_ip}")

        if new_node_ip in self.node_registry:
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

        # This is to make sure the ports don't overlap when running with mininet
        # Not required in the final implementation
        # TODO: Remove
        global node_ports
        node_ports+=1
        node_port = node_ports

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
        
        print(f"Node ready informed: {ready_node_ip}")

        # Early termination upon receiving the confirmation from the first node
        if(ready_node_ip == self.last_node_ip):
            print(f"Last node is already ip: {ready_node_ip}")
            return server_pb2.ReadyResponse(status_code=status.NODE_READY_INFORM_SUCCESS) 
        
        current_last_node = self.last_node_ip
        self.last_node_ip = ready_node_ip
        print(f"Last node ip updated: {self.last_node_ip}")

        print(f"Informing node of the last_node_ip update: {current_last_node}")
        with grpc.insecure_channel(current_last_node) as channel:
            stub = node_pb2_grpc.NodeServiceStub(channel)
            request = node_pb2.UpdateNextNodeRequest(next_node=self.last_node_ip)
            response : node_pb2.UpdateNextNodeResponse = stub.UpdateNextNode(request)

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