from concurrent import futures
import asyncio
import grpc
import traceback

import generated.node_pb2 as node_pb2
NodeInferenceMetricsResponse = node_pb2.NodeInferenceMetricsResponse
NodeInferenceMetricsRequest =  node_pb2.NodeInferenceMetricsRequest

PingRequest = node_pb2.PingRequest
PingResponse = node_pb2.PingResponse

import generated.node_pb2_grpc as node_pb2_grpc
from util import status, logger
logger = logger.logger

class NodeServer(node_pb2_grpc.NodeServiceServicer): 
    def __init__(self, model_path, node: any):
        self.node = node
        self.current_task_id = 0
        super().__init__()

    async def Ping(self, request:PingRequest, context: grpc.RpcContext) -> PingResponse:
        return PingResponse();

    async def GetInferenceMetrics(self, request: NodeInferenceMetricsRequest, context: grpc.RpcContext) -> NodeInferenceMetricsResponse:
       inference_metrics = self.node.network_observability.get_inference_metrics()

       return node_pb2.NodeInferenceMetricsResponse(
        #    values_bytes=inference_metrics['values'],
        #    indices_bytes=inference_metrics['indices'], 
           ingress_bytes=inference_metrics['ingress'], 
           egress_bytes=inference_metrics['egress']
        )
     
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

            logger.log(f"Inference task received: {task_id}")
            logger.log(f"Input tensor type: {type(input)}, length: {len(input)}")

            self.node.infer(task_id, input)            

            message = f"[id:{model_part_id}] Inference accepted for task_id: {task_id}"

            return node_pb2.InferenceResponse(
                status_code=status.INFERENCE_ACCEPTED,
                message=message,
                task_id=task_id
            )

        except Exception as e:
            message = f"[id: {model_part_id}] Inference failed: {str(e)}"
            logger.log(message)
            traceback.logger.log_exc()
            context.set_details(message)
            context.set_code(grpc.StatusCode.INTERNAL)
            return node_pb2.InferenceResponse(status_code=status.SERVER_ERROR)

    async def UpdateNextNode(self, request: node_pb2.UpdateNextNodeRequest, context: grpc.RpcContext):
        """
        Update the next node in the network.
        """
        try:
            next_node = request.next_node
            logger.log(f"Next node updated to: {next_node}")
            if next_node == None:
               self.node.next_node.set_result(None)
            else:
               self.node.next_node.set_result(next_node)
                
            return node_pb2.UpdateNextNodeResponse(
                status_code=status.NODE_UPDATE_SUCCESS,
            )
        except Exception as e:
            message=f"There was an error updating next node reference {e}"
            logger.log(message)
            return node_pb2.UpdateNextNodeResponse(
                status_code=status.NODE_UPDATE_ERROR,
                message=message
            ) 

    
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

    logger.log(f"Node server started on port {port}")
    return server    

if __name__ == '__main__':
    asyncio.run(main())
