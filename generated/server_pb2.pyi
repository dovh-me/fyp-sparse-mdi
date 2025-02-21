from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Test(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class StartInferenceRequest(_message.Message):
    __slots__ = ("input_tensor",)
    INPUT_TENSOR_FIELD_NUMBER: _ClassVar[int]
    input_tensor: bytes
    def __init__(self, input_tensor: _Optional[bytes] = ...) -> None: ...

class StartInferenceResponse(_message.Message):
    __slots__ = ("status_code", "message", "result")
    STATUS_CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    status_code: int
    message: str
    result: bytes
    def __init__(self, status_code: _Optional[int] = ..., message: _Optional[str] = ..., result: _Optional[bytes] = ...) -> None: ...

class EndInferenceRequest(_message.Message):
    __slots__ = ("task_id", "result")
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    task_id: int
    result: bytes
    def __init__(self, task_id: _Optional[int] = ..., result: _Optional[bytes] = ...) -> None: ...

class EndInferenceResponse(_message.Message):
    __slots__ = ("status_code", "message")
    STATUS_CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    status_code: int
    message: str
    def __init__(self, status_code: _Optional[int] = ..., message: _Optional[str] = ...) -> None: ...

class ReadyRequest(_message.Message):
    __slots__ = ("port",)
    PORT_FIELD_NUMBER: _ClassVar[int]
    port: int
    def __init__(self, port: _Optional[int] = ...) -> None: ...

class ReadyResponse(_message.Message):
    __slots__ = ("status_code", "message")
    STATUS_CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    status_code: int
    message: str
    def __init__(self, status_code: _Optional[int] = ..., message: _Optional[str] = ...) -> None: ...

class RegisterRequest(_message.Message):
    __slots__ = ("ip",)
    IP_FIELD_NUMBER: _ClassVar[int]
    ip: str
    def __init__(self, ip: _Optional[str] = ...) -> None: ...

class RegisterResponse(_message.Message):
    __slots__ = ("status_code", "message", "first_node", "prev_node", "model_part_id", "port", "chunk")
    STATUS_CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    FIRST_NODE_FIELD_NUMBER: _ClassVar[int]
    PREV_NODE_FIELD_NUMBER: _ClassVar[int]
    MODEL_PART_ID_FIELD_NUMBER: _ClassVar[int]
    PORT_FIELD_NUMBER: _ClassVar[int]
    CHUNK_FIELD_NUMBER: _ClassVar[int]
    status_code: int
    message: str
    first_node: str
    prev_node: str
    model_part_id: str
    port: int
    chunk: bytes
    def __init__(self, status_code: _Optional[int] = ..., message: _Optional[str] = ..., first_node: _Optional[str] = ..., prev_node: _Optional[str] = ..., model_part_id: _Optional[str] = ..., port: _Optional[int] = ..., chunk: _Optional[bytes] = ...) -> None: ...

class InferenceMetricsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class InferenceMetricsResponse(_message.Message):
    __slots__ = ("ingress_bytes", "egress_bytes", "values_bytes", "indices_bytes")
    INGRESS_BYTES_FIELD_NUMBER: _ClassVar[int]
    EGRESS_BYTES_FIELD_NUMBER: _ClassVar[int]
    VALUES_BYTES_FIELD_NUMBER: _ClassVar[int]
    INDICES_BYTES_FIELD_NUMBER: _ClassVar[int]
    ingress_bytes: int
    egress_bytes: int
    values_bytes: int
    indices_bytes: int
    def __init__(self, ingress_bytes: _Optional[int] = ..., egress_bytes: _Optional[int] = ..., values_bytes: _Optional[int] = ..., indices_bytes: _Optional[int] = ...) -> None: ...
