from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class InferenceRequest(_message.Message):
    __slots__ = ("next_model_part_id", "task_id", "input_tensor")
    NEXT_MODEL_PART_ID_FIELD_NUMBER: _ClassVar[int]
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    INPUT_TENSOR_FIELD_NUMBER: _ClassVar[int]
    next_model_part_id: int
    task_id: int
    input_tensor: bytes
    def __init__(self, next_model_part_id: _Optional[int] = ..., task_id: _Optional[int] = ..., input_tensor: _Optional[bytes] = ...) -> None: ...

class InferenceResponse(_message.Message):
    __slots__ = ("status_code", "message", "task_id", "current_model_part_id")
    STATUS_CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    CURRENT_MODEL_PART_ID_FIELD_NUMBER: _ClassVar[int]
    status_code: int
    message: str
    task_id: int
    current_model_part_id: int
    def __init__(self, status_code: _Optional[int] = ..., message: _Optional[str] = ..., task_id: _Optional[int] = ..., current_model_part_id: _Optional[int] = ...) -> None: ...

class NodeInferenceMetricsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class NodeInferenceMetricsResponse(_message.Message):
    __slots__ = ("values_bytes", "indices_bytes", "ingress_bytes", "egress_bytes")
    VALUES_BYTES_FIELD_NUMBER: _ClassVar[int]
    INDICES_BYTES_FIELD_NUMBER: _ClassVar[int]
    INGRESS_BYTES_FIELD_NUMBER: _ClassVar[int]
    EGRESS_BYTES_FIELD_NUMBER: _ClassVar[int]
    values_bytes: int
    indices_bytes: int
    ingress_bytes: int
    egress_bytes: int
    def __init__(self, values_bytes: _Optional[int] = ..., indices_bytes: _Optional[int] = ..., ingress_bytes: _Optional[int] = ..., egress_bytes: _Optional[int] = ...) -> None: ...

class UpdateNextNodeRequest(_message.Message):
    __slots__ = ("next_node",)
    NEXT_NODE_FIELD_NUMBER: _ClassVar[int]
    next_node: str
    def __init__(self, next_node: _Optional[str] = ...) -> None: ...

class UpdateNextNodeResponse(_message.Message):
    __slots__ = ("status_code", "message")
    STATUS_CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    status_code: int
    message: str
    def __init__(self, status_code: _Optional[int] = ..., message: _Optional[str] = ...) -> None: ...

class PingRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class PingResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
