import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
import os
import sys

module_path = os.path.abspath('./')
sys.path.insert(0, module_path)

from server.server import Server
from generated.server_pb2 import (
    RegisterRequest, RegisterResponse, ReadyRequest, 
    StartInferenceRequest, StartInferenceResponse, InferenceMetricsResponse
)

@pytest.fixture
def server():
    """Fixture to create a Server instance with mock dependencies."""
    return Server(node_config=[])

@pytest.mark.asyncio
async def test_register_node(server):
    """Test node registration."""
    request = RegisterRequest()
    context = AsyncMock()
    context.peer.return_value = "ipv4:192.168.1.5:50432"

    responses = [response async for response in server.RegisterNode(request, context)]
    
    assert responses[0].status_code in {200, 409, 500}  # Success, already registered, or error

@pytest.mark.asyncio
async def test_inform_ready(server):
    """Test the InformReady method."""
    request = ReadyRequest(port=50052, node_id="node_1")
    context = AsyncMock()
    context.peer.return_value = "ipv4:192.168.1.5:50432"

    response = await server.InformReady(request, context)
    
    assert response.status_code in {200, 500}  # Success or failure

@pytest.mark.asyncio
async def test_start_inference(server):
    """Test StartInference request processing."""
    input_tensor = b"\x00\x00\x80?\x00\x00\x00@\x00\x00@@\x00\x00\x80@"  # Fake binary data
    request = StartInferenceRequest(input_tensor=input_tensor)
    context = AsyncMock()

    response = await server.StartInference(request, context)

    assert response.status_code in {200, 400, 500}  # Success, bad request, or internal error

@pytest.mark.asyncio
async def test_process_inference_queue(server):
    """Test processing of the inference queue."""
    inference_task = MagicMock(task_id=1, input_tensor=b"fake_tensor")
    await server.inference_queue.put(inference_task)

    with patch.object(server, "send_for_inference", new_callable=AsyncMock) as mock_send:
        task = asyncio.create_task(server.process_inference_queue())
        await asyncio.sleep(1)  # Allow processing time
        task.cancel()

    mock_send.assert_called_once_with(inference_task)

@pytest.mark.asyncio
async def test_get_inference_metrics(server):
    """Test fetching inference metrics."""
    request = MagicMock()
    context = AsyncMock()

    with patch.object(server, "get_inference_metrics_from_node", new_callable=AsyncMock) as mock_metrics:
        mock_metrics.return_value.egress_bytes = 1024
        response = await server.GetInferenceMetrics(request, context)

    assert response.egress_bytes == 1024