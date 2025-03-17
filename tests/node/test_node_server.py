import pytest
import asyncio
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
import os
import sys

module_path = os.path.abspath('./')
sys.path.insert(0, module_path)

from modules.EncoderDecoder import HuffmanEncoding
from node.node import Node
from generated.node_pb2 import InferenceRequest, InferenceResponse
from generated.node_pb2_grpc import NodeServiceStub

@pytest.fixture
def node():
    """Fixture to create a Node instance with mock dependencies."""
    return Node(coordinator_address="127.0.0.1:50051", node_ip="192.168.1.10")

@pytest.mark.asyncio
@patch.object(Node, "encoderDecoder", create=True)
async def test_infer(mock_encoder_decoder, node):
    """Test asynchronous inference initiation."""
    encoder = HuffmanEncoding()

    # Create a valid encoded tensor
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
    mock_encoded_tensor = b"huffman|" + encoder.encode(tensor)

    # Mock decode() method to return the original tensor
    mock_encoder_decoder.decode.return_value = tensor

    node.ort_session = MagicMock()
    node.ort_session.get_inputs.return_value = [MagicMock(name="input", shape=(2,2))]
    node.ort_session.run.return_value = [np.array([[10.0, 20.0], [30.0, 40.0]])]

    # Use correctly formatted encoded data
    await node.infer(task_id=1, input=mock_encoded_tensor)

    assert mock_encoder_decoder.decode.called
    assert node.ort_session.run.called

@pytest.mark.asyncio
@patch.object(Node, "connection", new_callable=AsyncMock)
@patch("modules.node.NodeServiceStub")
async def test_forward_to_next_node(mock_stub, mock_connection, node):
    """Test forwarding inference result to the next node."""
    mock_stub.return_value.Infer = AsyncMock()

    node.connection.set_result(mock_connection)
    node.next_node.set_result("192.168.1.20:50052")

    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]])

    await node.forward_to_next_node(task_id=1, input_tensor=input_tensor)

    assert mock_stub.called
    mock_stub.return_value.Infer.assert_called_once()

@pytest.mark.asyncio
@patch.object(Node, "ping_next_node", new_callable=AsyncMock)
async def test_ping_in_background(mock_ping, node):
    """Ensure the ping background task is executed."""
    mock_ping.side_effect = [None, None, None]  # Simulating 3 pings

    task = asyncio.create_task(node.ping_in_background(ping_timeout=1))
    await asyncio.sleep(3.5)
    task.cancel()

    assert mock_ping.call_count >= 3  # Should have pinged at least 3 times

@pytest.mark.asyncio
@patch.object(Node, "connection", new_callable=AsyncMock)
@patch("modules.node.NodeServiceStub")
async def test_finish_inference(mock_stub, mock_connection, node):
    """Test finalizing inference results."""
    mock_stub.return_value.EndInference = AsyncMock()

    node.connection.set_result(mock_connection)

    await node.finish_inference(task_id=1, result=b"final_result")

    mock_stub.return_value.EndInference.assert_called_once()