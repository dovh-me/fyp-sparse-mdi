import zlib
import heapq
from collections import Counter, namedtuple
from typing import Any, Tuple

class EncodingStrategy:
    """Abstract base class for encoding strategies."""
    def encode(self, data: Any) -> Any:
        raise NotImplementedError

    def decode(self, data: Any) -> Any:
        raise NotImplementedError

# -------------------- ZLIB Encoding --------------------
class ZlibEncoding(EncodingStrategy):
    def encode(self, data: bytes) -> bytes:
        return zlib.compress(data)

    def decode(self, data: bytes) -> bytes:
        return zlib.decompress(data)

# -------------------- Run-Length Encoding (RLE) --------------------
class RLEEncoding(EncodingStrategy):
    def encode(self, data: str) -> str:
        encoded = []
        count = 1
        for i in range(1, len(data)):
            if data[i] == data[i-1]:
                count += 1
            else:
                encoded.append(data[i-1] + str(count))
                count = 1
        encoded.append(data[-1] + str(count))
        return "".join(encoded)

    def decode(self, data: str) -> str:
        decoded = []
        i = 0
        while i < len(data):
            char = data[i]
            count = ""
            i += 1
            while i < len(data) and data[i].isdigit():
                count += data[i]
                i += 1
            decoded.append(char * int(count))
        return "".join(decoded)

# -------------------- Huffman Encoding --------------------
class HuffmanEncoding(EncodingStrategy):
    class Node(namedtuple("Node", ["char", "freq", "left", "right"])):
        def __lt__(self, other):
            return self.freq < other.freq

    def _build_tree(self, data: str):
        frequency = Counter(data)
        heap = [self.Node(char, freq, None, None) for char, freq in frequency.items()]
        heapq.heapify(heap)
        
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merged = self.Node(None, left.freq + right.freq, left, right)
            heapq.heappush(heap, merged)

        return heap[0]

    def _generate_codes(self, root, prefix="", code_map={}):
        if root is None:
            return
        if root.char is not None:
            code_map[root.char] = prefix
        self._generate_codes(root.left, prefix + "0", code_map)
        self._generate_codes(root.right, prefix + "1", code_map)
        return code_map

    def encode(self, data: str) -> Tuple[str, dict]:
        root = self._build_tree(data)
        code_map = self._generate_codes(root)
        encoded_data = "".join(code_map[char] for char in data)
        return encoded_data, code_map

    def decode(self, encoded_data: str, code_map: dict) -> str:
        reverse_map = {v: k for k, v in code_map.items()}
        decoded_data = ""
        buffer = ""
        for bit in encoded_data:
            buffer += bit
            if buffer in reverse_map:
                decoded_data += reverse_map[buffer]
                buffer = ""
        return decoded_data

# -------------------- Encoder-Decoder Class --------------------
class EncoderDecoder:
    def __init__(self, strategy: EncodingStrategy):
        """
        Initialize with a specific encoding strategy.
        :param strategy: Instance of EncodingStrategy (e.g., ZlibEncoding, RLEEncoding).
        """
        self.strategy = strategy

    def encode(self, data: Any) -> Any:
        return self.strategy.encode(data)

    def decode(self, data: Any) -> Any:
        return self.strategy.decode(data)