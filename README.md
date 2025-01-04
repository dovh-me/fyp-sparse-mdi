# Step 1: Check Python version or install Python via Homebrew

python3 --version
brew install python # Optional: If Python 3 is not installed

# Step 2: Create virtual environment

python3 -m venv myenv # Using venv (recommended) - use this

# Step 3: Activate the virtual environment

source myenv/bin/activate

# Step 4: Install packages using pip

pip install grpcio grpcio-tools torch torchvision

# Step 5: Deactivate when done

deactivate

# Generate the grpc protocols

python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. sparsify.proto

# Limitations

- Only supports a single model part per node.
- Model partitioning and allocation is static (for now).
- Model distribution, allocation and partitioning happens through the server

# Commands

## gRPC file generation

### Sample

```bash
python -m grpc_tools.protoc \
    -I. \
    --python_out=./generated \
    --grpc_python_out=./generated \
    your_service.proto
```

### Server proto

```bash
python -m grpc_tools.protoc -I./protos --python_out=generated --pyi_out=generated --grpc_python_out=generated ./protos/server.proto
```

### Node proto

```bash
python -m grpc_tools.protoc -I./protos --python_out=generated --pyi_out=generated --grpc_python_out=generated ./protos/node.proto
```

## Debug Run

```bash
GRPC_VERBOSITY=debug GRPC_TRACE=all python <script name>.py
```
