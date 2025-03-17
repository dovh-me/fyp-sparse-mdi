import pytest
import os
import sys

module_path = os.path.abspath('./')
sys.path.insert(0, module_path)

from modules.NetworkObservabilityTracker import NetworkObservabilityTracker

@pytest.fixture
def tracker():
    """Fixture to create a new instance of NetworkObservabilityTracker before each test"""
    return NetworkObservabilityTracker()

def test_initial_state(tracker):
    """Check if all initial values are set correctly"""
    assert tracker.inference_ingress == 0
    assert tracker.inference_egress == 0
    assert tracker.rtt['min'] == 0
    assert tracker.rtt['max'] == 0
    assert tracker.rtt['current'] == 0
    assert tracker.packet_loss == 0
    assert tracker.inference_counter == 0
    assert tracker.sparsity_counter == 0

def test_update_rtt_with_next_node(tracker):
    """Ensure RTT updates correctly"""
    tracker.update_rtt_with_next_node(100)
    assert tracker.rtt['current'] == 100
    assert tracker.rtt['max'] == 100
    assert tracker.rtt['min'] == 0

    tracker.update_rtt_with_next_node(50)
    assert tracker.rtt['min'] == 0
    assert tracker.rtt['max'] == 100

    tracker.update_rtt_with_next_node(150)
    assert tracker.rtt['max'] == 150

def test_update_ingress_metrics(tracker):
    """Verify inference ingress metric updates correctly"""
    tracker.update_ingress_metrics(b"data_packet_123")
    assert tracker.inference_ingress == len(b"data_packet_123")

    tracker.update_ingress_metrics(b"more_data")
    assert tracker.inference_ingress == len(b"data_packet_123") + len(b"more_data")

def test_update_egress_metrics(tracker):
    """Verify inference egress metric updates correctly"""
    tracker.update_egress_metrics(b"sent_data")
    assert tracker.inference_egress == len(b"sent_data")

    tracker.update_egress_metrics(b"extra_packet")
    assert tracker.inference_egress == len(b"sent_data") + len(b"extra_packet")

def test_update_inference_counter(tracker):
    """Ensure inference counter increments correctly"""
    tracker.update_inference_counter()
    assert tracker.inference_counter == 1

    tracker.update_inference_counter()
    assert tracker.inference_counter == 2

def test_update_sparsity_counter(tracker):
    """Ensure sparsity counter increments correctly"""
    tracker.update_sparsity_counter()
    assert tracker.sparsity_counter == 1

    tracker.update_sparsity_counter()
    assert tracker.sparsity_counter == 2

def test_update_packet_loss(tracker):
    """Ensure packet loss updates correctly"""
    tracker.update_packet_loss(5)
    assert tracker.packet_loss == 5

    tracker.update_packet_loss(10)
    assert tracker.packet_loss == 10

def test_get_inference_metrics(tracker):
    """Verify that inference metrics return expected values"""
    tracker.update_ingress_metrics(b"test_data")
    tracker.update_egress_metrics(b"outgoing_data")
    tracker.update_rtt_with_next_node(75)

    metrics = tracker.get_inference_metrics()
    assert metrics['ingress'] == len(b"test_data")
    assert metrics['egress'] == len(b"outgoing_data")
    assert metrics['rtt'] == 75

def test_get_rtt(tracker):
    """Ensure RTT retrieval works correctly"""
    tracker.update_rtt_with_next_node(200)
    rtt_data = tracker.get_rtt()
    assert rtt_data['current'] == 200
    assert rtt_data['max'] == 200
    assert rtt_data['min'] == 0