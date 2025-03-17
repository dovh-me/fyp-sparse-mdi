import pytest
import numpy as np
import os
import sys

module_path = os.path.abspath('./')
sys.path.insert(0, module_path)

from modules.NetworkObservabilityTracker import NetworkObservabilityTracker
from modules.SparsityEngine import SparsityEngine, SparsityEngineDefaults

@pytest.fixture
def tracker():
    """Fixture to create a NetworkObservabilityTracker instance for testing."""
    networkTracker = NetworkObservabilityTracker()
    networkTracker.update_rtt_with_next_node(20)
    return networkTracker 

@pytest.fixture
def sparsity_engine(tracker):
    """Fixture to create a SparsityEngine instance for testing."""
    return SparsityEngine(tracker)

def test_compute_tensor_sparsity(sparsity_engine):
    """Test sparsity computation for different tensors."""
    tensor1 = np.array([0, 1, 0, 2, 3, 0, 4])
    tensor2 = np.zeros((3, 3))
    tensor3 = np.array([1, 2, 3, 4, 5])

    assert sparsity_engine.compute_tensor_sparsity(tensor1) == 3
    assert sparsity_engine.compute_tensor_sparsity(tensor2) == 9  # All zeros
    assert sparsity_engine.compute_tensor_sparsity(tensor3) == 0  # No zeros

def test_get_rtt(sparsity_engine, tracker):
    """Ensure RTT normalization works correctly."""
    tracker.update_rtt_with_next_node(10)
    assert 0 <= sparsity_engine.get_rtt() <= 1

    tracker.update_rtt_with_next_node(15)
    assert 0 <= sparsity_engine.get_rtt() <= 1

def test_compute_sparsity_level(sparsity_engine):
    """Test sparsity level computation based on network and tensor conditions."""
    tensor = np.array([[1, 0, 3], [0, 2, 0], [4, 5, 0]])

    k, sparsity_level = sparsity_engine.compute_sparsity_level(tensor)
    assert 0 <= sparsity_level <= 1  # Sparsity should be within valid range

def test_compute_network_factor(sparsity_engine):
    """Ensure network factor calculation is within expected bounds."""
    factor = sparsity_engine.compute_network_factor()
    assert factor >= 0  # Should not be negative

def test_update_defaults_from_node_config(sparsity_engine):
    """Ensure default values are correctly updated from a node configuration."""
    node_config = {
        "sparsity_factor": 1.5,
        "max_sparsity": 0.6,
        "min_rtt": 5,
        "max_rtt": 12,
        "min_pl": 0.02,
        "rtt_factor": 0.7,
        "pl_factor": 0.1
    }

    sparsity_engine.update_defaults_from_node_config(node_config)
    assert sparsity_engine.defaults.sparsity_factor == 1.5
    assert sparsity_engine.defaults.max_sparsity == 0.6
    assert sparsity_engine.defaults.min_rtt == 5
    assert sparsity_engine.defaults.max_rtt == 12
    assert sparsity_engine.defaults.min_pl == 0.02
    assert sparsity_engine.defaults.rtt_factor == 0.7
    assert sparsity_engine.defaults.pl_factor == 0.1