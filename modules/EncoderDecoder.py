from typing import Dict
import struct
import numpy as np

import os
import sys

module_path = os.path.abspath('../')
sys.path.insert(0, module_path)

from modules.NetworkObservabilityTracker import NetworkObservabilityTracker
from modules.AdaptiveSparsityEngine import SparsityEngine
from util.logger import logger

# Base Encoding Strategy
class EncodingStrategy:
    def encode(self, tensor: np.ndarray) -> bytes:
        raise NotImplementedError
    
    def decode(self, encoded_data: bytes) -> np.ndarray:
        raise NotImplementedError

# Huffman Encoding (Example Strategy using struct for efficiency)
class HuffmanEncoding(EncodingStrategy):
    def encode(self, tensor: np.ndarray) -> bytes:
        shape = tensor.shape
        shape_size = len(shape)
        flattened = tensor.flatten()
        packed_shape = struct.pack(f"{shape_size}I", *shape)  # Pack shape as unsigned ints
        packed_data = struct.pack(f"{len(flattened)}f", *flattened)  # Pack tensor values as floats
        return struct.pack("I", shape_size) + packed_shape + packed_data
    
    def decode(self, encoded_data: bytes) -> np.ndarray:
        shape_size = struct.unpack("I", encoded_data[:4])[0]  # Extract number of dimensions
        shape = struct.unpack(f"{shape_size}I", encoded_data[4:4 + shape_size * 4])
        data = struct.unpack(f"{np.prod(shape)}f", encoded_data[4 + shape_size * 4:])
        return np.array(data).reshape(shape)

# Sparse Encoding Strategy using struct
class SparseEncoding(EncodingStrategy):
    def __init__(self, network_observer):
        super().__init__()
        self.sparsity_engine = SparsityEngine(network_observer=network_observer)
        self.network_observer: NetworkObservabilityTracker = network_observer

    def top_k_sparsify(self, activations, k):
        """
        Create a sparse tensor by keeping only the top-k values.
        """
        activations_np = activations.flatten()
        k = min(k, activations_np.size)

        top_k_indices = np.argpartition(np.abs(activations_np), -k)[-k:]
        top_k_values = activations_np[top_k_indices]

        # Sort by magnitude for stable results
        sorted_indices = np.argsort(np.abs(top_k_values))
        top_k_values = top_k_values[sorted_indices]
        top_k_indices = top_k_indices[sorted_indices]

        return top_k_values, top_k_indices

    def encode(self, tensor: np.ndarray, sparsity_level: int) -> bytes:
        # sparsity_level = self.sparsity_engine.compute_tensor_sparsity(tensor) 
        # k = int(round((tensor.size * sparsity_level), 0))
        k = int(round((sparsity_level), 0))
        logger.log(f"k: {k} | sparsity_level: {sparsity_level}")

        shape = tensor.shape
        shape_size = len(shape)
        
        values, flat_indices = self.top_k_sparsify(tensor, k=k)
        num_elements = len(values)  # Explicitly store number of nonzero elements

        packed_shape = struct.pack(f"{shape_size}I", *shape)
        packed_num_elements = struct.pack("I", num_elements)
        packed_indices = struct.pack(f"{num_elements}I", *flat_indices)
        packed_values = struct.pack(f"{num_elements}f", *values)

        return struct.pack("I", shape_size) + packed_shape + packed_num_elements + packed_indices + packed_values

    def decode(self, encoded_data: bytes) -> np.ndarray:
        shape_size = struct.unpack("I", encoded_data[:4])[0]  # Extract number of dimensions
        shape = struct.unpack(f"{shape_size}I", encoded_data[4:4 + shape_size * 4])
        offset = 4 + shape_size * 4

        num_elements = struct.unpack("I", encoded_data[offset:offset + 4])[0]  # Retrieve stored num_elements
        offset += 4

        indices = struct.unpack(f"{num_elements}I", encoded_data[offset:offset + num_elements * 4])
        offset += num_elements * 4
        values = struct.unpack(f"{num_elements}f", encoded_data[offset:offset + num_elements * 4])

        tensor = np.zeros(shape)
        tensor.flat[list(indices)] = values  # Use `.flat[]` for 1D indexing

        return tensor

# Encoder-Decoder Manager
class EncoderDecoderManager:
    def __init__(self, network_observer: NetworkObservabilityTracker, sparsity_engine: SparsityEngine):
        self.network_observer = network_observer
        self.strategies: Dict[str, EncodingStrategy] = {}
        self.threshold = 0.55
        self.register_strategy('huffman', HuffmanEncoding())
        self.register_strategy('sparse', SparseEncoding(network_observer=network_observer))

        self.sparsity_engine = sparsity_engine 

    def register_strategy(self, name: str, strategy: EncodingStrategy):
        self.strategies[name] = strategy
    
    def encode(self, strategy_name: str, tensor: np.ndarray) -> bytes:
        is_adaptive = strategy_name == 'adaptive'

        if is_adaptive: 
            k, sparsity_ratio = self.sparsity_engine.compute_sparsity_level(inference_tensor=tensor)

            # This ensures that sparse encoding is not applied if minimum required level of sparsity is not available.
            # Here the 0.5 is obtained based on the mathematical proof that was obtained regarding the COO format.
            # Please refer to the thesis Chapter 6 for the detailed explanation.
            strategy_name = 'sparse' if sparsity_ratio >= self.threshold else 'huffman'

            encoded_tensor = None

            if strategy_name == 'sparse':
                encoded_tensor = self.strategies[strategy_name].encode(tensor, sparsity_level=k)
            else: 
                encoded_tensor = self.strategies[strategy_name].encode(tensor)
            
            logger.log(f"strategy_name: {strategy_name}, sparsity_level: {sparsity_ratio}, k: {k}, tensor_size: {tensor.size}")

            return strategy_name.encode() + b'|' + encoded_tensor
  
        if strategy_name not in self.strategies:
            raise ValueError(f"Encoding strategy '{strategy_name}' not found.")

        encoded_tensor = None 
        if strategy_name == 'sparse':
            sparsity_level = self.sparsity_engine.defaults.sparsity_level;
            total_activations = tensor.size

            k = round((1 - sparsity_level) * total_activations)
            encoded_tensor = self.strategies[strategy_name].encode(tensor, sparsity_level=k) # Hardcoded for now

        else: 
            encoded_tensor = self.strategies[strategy_name].encode(tensor)

        return strategy_name.encode() + b'|' + encoded_tensor
    
    def decode(self, encoded_data: bytes) -> np.ndarray:
        strategy_name, encoded_tensor = encoded_data.split(b'|', 1)
        strategy_name = strategy_name.decode()
        
        if strategy_name not in self.strategies:
            raise ValueError(f"Decoding strategy '{strategy_name}' not found.")
        
        return self.strategies[strategy_name].decode(encoded_tensor)