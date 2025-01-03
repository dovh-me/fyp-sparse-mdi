from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Test(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class StartInferenceRequest(_message.Message):
    __slots__ = ("input_tensor",)
    INPUT_TENSOR_FIELD_NUMBER: _ClassVar[int]
    input_tensor: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, input_tensor: _Optional[_Iterable[float]] = ...) -> None: ...

class StartInferenceResponse(_message.Message):
    __slots__ = ("status_code", "message", "result")
    STATUS_CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    status_code: int
    message: str
    result: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, status_code: _Optional[int] = ..., message: _Optional[str] = ..., result: _Optional[_Iterable[float]] = ...) -> None: ...

class EndInferenceRequest(_message.Message):
    __slots__ = ("task_id", "result")
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    task_id: int
    result: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, task_id: _Optional[int] = ..., result: _Optional[_Iterable[float]] = ...) -> None: ...

class EndInferenceResponse(_message.Message):
    __slots__ = ("status_code", "message")
    STATUS_CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    status_code: int
    message: str
    def __init__(self, status_code: _Optional[int] = ..., message: _Optional[str] = ...) -> None: ...

class ReadyRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

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
