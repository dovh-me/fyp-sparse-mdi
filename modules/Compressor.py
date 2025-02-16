import zlib

class Compressor:
    def compress():
        raise NotImplementedError()
    def decompress():
        raise NotImplementedError()

# -------------------- ZLIB Encoding --------------------
class ZlibEncoding(Compressor):
    def compress(self, data: bytes) -> bytes:
        return zlib.compress(data)

    def compress(self, data: bytes) -> bytes:
        return zlib.decompress(data)

