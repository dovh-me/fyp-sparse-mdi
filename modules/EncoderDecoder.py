from typing import Dict, Type
import struct
import numpy as np

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

    def encode(self, tensor: np.ndarray) -> bytes:
        shape = tensor.shape
        shape_size = len(shape)
        
        values, flat_indices = self.top_k_sparsify(tensor, k=5000)
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
    def __init__(self):
        self.strategies: Dict[str, EncodingStrategy] = {}
        self.register_strategy('huffman', HuffmanEncoding())
        self.register_strategy('sparse', SparseEncoding())
    
    def register_strategy(self, name: str, strategy: EncodingStrategy):
        self.strategies[name] = strategy
    
    def encode(self, strategy_name: str, tensor: np.ndarray) -> bytes:
        if strategy_name not in self.strategies:
            raise ValueError(f"Encoding strategy '{strategy_name}' not found.")

        encoded_tensor = self.strategies[strategy_name].encode(tensor)
        return strategy_name.encode() + b'|' + encoded_tensor
    
    def decode(self, encoded_data: bytes) -> np.ndarray:
        strategy_name, encoded_tensor = encoded_data.split(b'|', 1)
        strategy_name = strategy_name.decode()
        
        if strategy_name not in self.strategies:
            raise ValueError(f"Decoding strategy '{strategy_name}' not found.")
        
        return self.strategies[strategy_name].decode(encoded_tensor)