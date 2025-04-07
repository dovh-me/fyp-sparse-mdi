import pytest
import numpy as np
import os
import sys

module_path = os.path.abspath('./')
sys.path.insert(0, module_path)

from modules.EncoderDecoder import HuffmanEncoding, SparseEncoding, EncoderDecoderManager
from modules.NetworkObservabilityTracker import NetworkObservabilityTracker
from modules.AdaptiveSparsityEngine import SparsityEngine

@pytest.fixture
def network_observer():
    """Fixture to create a NetworkObservabilityTracker instance for testing."""
    networkTracker = NetworkObservabilityTracker()
    networkTracker.update_rtt_with_next_node(20)
    return networkTracker 

@pytest.fixture
def sparsity_engine(network_observer):
    """Fixture to create a SparsityEngine instance for testing."""
    return SparsityEngine(network_observer)

@pytest.fixture
def huffman_encoder():
    """Fixture to create a HuffmanEncoding instance."""
    return HuffmanEncoding()

@pytest.fixture
def sparse_encoder(network_observer):
    """Fixture to create a SparseEncoding instance."""
    return SparseEncoding(network_observer=network_observer)

@pytest.fixture
def encoder_decoder_manager(network_observer, sparsity_engine):
    """Fixture to create an EncoderDecoderManager instance."""
    return EncoderDecoderManager(network_observer, sparsity_engine)

def test_huffman_encoding(huffman_encoder):
    """Test Huffman encoding and decoding of a tensor."""
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    encoded = huffman_encoder.encode(tensor)
    decoded = huffman_encoder.decode(encoded)
    
    assert np.allclose(tensor, decoded)

def test_sparse_encoding(sparse_encoder):
    """Test Sparse encoding and decoding of a tensor."""
    tensor = np.array([[1.0, 0.0], [0.0, 4.0]], dtype=np.float32)
    encoded = sparse_encoder.encode(tensor, sparsity_level=2)
    decoded = sparse_encoder.decode(encoded)
    
    assert np.allclose(tensor, decoded)

def test_sparse_encoding_empty_tensor(sparse_encoder):
    """Ensure sparse encoding handles an empty tensor correctly."""
    tensor = np.zeros((2, 2), dtype=np.float32)
    encoded = sparse_encoder.encode(tensor, sparsity_level=2)
    decoded = sparse_encoder.decode(encoded)
    
    assert np.allclose(tensor, decoded)

def test_encoder_decoder_manager_registration(encoder_decoder_manager):
    """Ensure encoding strategies are properly registered."""
    assert "huffman" in encoder_decoder_manager.strategies
    assert "sparse" in encoder_decoder_manager.strategies

def test_encoder_decoder_manager_encoding_decoding(encoder_decoder_manager):
    """Test encoding and decoding via EncoderDecoderManager."""
    tensor = np.array([[5.0, 0.0], [0.0, -3.0]], dtype=np.float32)
    encoded = encoder_decoder_manager.encode("sparse", tensor)
    decoded = encoder_decoder_manager.decode(encoded)
    
    assert np.allclose(tensor, decoded)

def test_encoder_decoder_manager_adaptive_encoding(encoder_decoder_manager):
    """Ensure adaptive encoding selects the correct strategy."""
    tensor = np.random.rand(4, 4).astype(np.float32)
    encoded = encoder_decoder_manager.encode("adaptive", tensor)
    decoded = encoder_decoder_manager.decode(encoded)
    
    assert np.allclose(tensor, decoded)