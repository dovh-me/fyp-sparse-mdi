#!/bin/bash

PROTO_DIR="./protos"
OUTPUT_DIR="./generated"

mkdir -p $OUTPUT_DIR
# python -m grpc_tools.protoc \
#     -I$PROTO_DIR \
#     --python_out=$OUTPUT_DIR \
#     --grpc_python_out=$OUTPUT_DIR \
#     $PROTO_DIR/server.proto

# python -m grpc_tools.protoc \
#     -I$PROTO_DIR \
#     --python_out=$OUTPUT_DIR \
#     --grpc_python_out=$OUTPUT_DIR \
#     $PROTO_DIR/node.proto

python -m grpc_tools.protoc -I./protos --python_out=generated --pyi_out=generated --grpc_python_out=generated ./protos/server.proto
python -m grpc_tools.protoc -I./protos --python_out=generated --pyi_out=generated --grpc_python_out=generated ./protos/node.proto