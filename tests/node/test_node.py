import pytest
import asyncio
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

import os
import sys

module_path = os.path.abspath('./')
sys.path.insert(0, module_path)

from node.node import Node, NodeNotConnected
from modules.NetworkObservabilityTracker import NetworkObservabilityTracker
from modules.SparsityEngine import SparsityEngine
from modules.EncoderDecoder import EncoderDecoderManager

@pytest.fixture
def node():
    """Fixture to create a Node instance with mock dependencies."""
    return Node(coordinator_address="127.0.0.1:50051", node_ip="192.168.1.10")

@patch("src.modules.node.grpc.aio.insecure_channel")
async def test_connect_to_coordinator_node(mock_channel, node):
    """Test connection initialization to the coordinator node."""
    mock_channel.return_value.channel_ready = AsyncMock()
    
    await node.connect_to_coordinator_node()
    
    assert node.connection.done()  # Future should be resolved

@patch("src.modules.node.grpc.aio.insecure_channel")
async def test_connect_to_next_node(mock_channel, node):
    """Test connection initialization to the next node."""
    node.next_node.set_result("192.168.1.20:50052")  # Simulating next node setup
    mock_channel.return_value.channel_ready = AsyncMock()
    
    await node.connect_to_next_node()
    
    assert node.connection.done()  # Future should be resolved

def test_load_config(node):
    """Test loading node configuration."""
    node.config = {
        "encoding_strategy": "huffman",
        "sparsity_factor": 1.5
    }
    
    node.load_config()
    
    assert node.encoding_strategy == "huffman"
    assert node.sparsity_engine.defaults.sparsity_factor == 1.5

def test_update_network_metrics(node):
    """Ensure network observability metrics update correctly."""
    node.network_observability.update_ingress_metrics(b"test_data")
    node.network_observability.update_egress_metrics(b"outgoing_data")

    assert node.network_observability.inference_ingress == len(b"test_data")
    assert node.network_observability.inference_egress == len(b"outgoing_data")

@patch.object(Node, "encoderDecoder", create=True)
def test_infer(mock_encoder_decoder, node):
    """Test inference initiation."""
    mock_encoder_decoder.decode.return_value = np.array([[1.0, 2.0], [3.0, 4.0]])
    
    node.ort_session = MagicMock()
    node.ort_session.get_inputs.return_value = [MagicMock(name="input", shape=(2,2))]
    node.ort_session.run.return_value = [np.array([[10.0, 20.0], [30.0, 40.0]])]

    input_tensor = b"encoded_input"
    
    node.infer(task_id=1, input=input_tensor)

    assert mock_encoder_decoder.decode.called
    assert node.ort_session.run.called

@patch.object(Node, "connection", new_callable=AsyncMock)
@patch("src.modules.node.NodeServiceStub")
async def test_forward_to_next_node(mock_stub, mock_connection, node):
    """Test forwarding inference result to the next node."""
    mock_stub.return_value.Infer = AsyncMock()
    
    node.connection.set_result(mock_connection)
    node.next_node.set_result("192.168.1.20:50052")
    
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
    
    await node.forward_to_next_node(task_id=1, input_tensor=input_tensor)

    assert mock_stub.called
    mock_stub.return_value.Infer.assert_called_once()

@patch.object(Node, "connection", new_callable=AsyncMock)
@patch("src.modules.node.ServerStub")
async def test_finish_inference(mock_stub, mock_connection, node):
    """Test finalizing inference results."""
    mock_stub.return_value.EndInference = AsyncMock()

    node.connection.set_result(mock_connection)
    
    await node.finish_inference(task_id=1, result=b"final_result")
    
    mock_stub.return_value.EndInference.assert_called_once()

@patch.object(Node, "ping_next_node", new_callable=AsyncMock)
async def test_ping_in_background(mock_ping, node):
    """Ensure the ping background task is executed."""
    mock_ping.side_effect = [None, None, None]  # Simulating 3 pings

    task = asyncio.create_task(node.ping_in_background(ping_timeout=1))
    await asyncio.sleep(3.5)
    task.cancel()

    assert mock_ping.call_count >= 3  # Should have pinged at least 3 times