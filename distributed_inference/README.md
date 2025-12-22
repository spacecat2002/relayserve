# Distributed Inference System

This module implements a distributed inference system that uses both CPU and GPU nodes for efficient model serving with lazy loading and migration capabilities.

## Architecture

The system consists of:

1. **CPU Backend**: Loads model weights immediately and starts prefill inference
2. **GPU Backend**: Uses lazy loading to load weights in the background
3. **Migration Coordinator**: Manages the migration of inference from CPU to GPU
4. **Client**: Sends requests to the distributed system

## Workflow

1. Client sends a request to the migration coordinator
2. CPU backend loads weights and starts prefill inference
3. GPU backend loads weights in parallel (lazy loading)
4. Once GPU weights are loaded, inference migrates from CPU to GPU
5. GPU backend continues generation from where CPU left off

## Setup

### 1. Start Ray Head Node

```bash
ray start --head --port=10000
```

### 2. Start CPU and GPU Nodes

You can start CPU and GPU nodes in separate terminals:

**Terminal 1 - Start CPU Node:**
```bash
# Make script executable
chmod +x scripts/start_cpu_node.sh

# Set virtual environment path (optional)
export VLLM_CPU_ENV=/path/to/vllm-cpu/env

# Set head node address (if not localhost:10000)
export HEAD_NODE_ADDRESS=localhost:10000

# Start CPU node
./scripts/start_cpu_node.sh
```

**Terminal 2 - Start GPU Node:**
```bash
# Make script executable
chmod +x scripts/start_gpu_node.sh

# Set virtual environment path (optional)
export VLLM_GPU_ENV=/path/to/vllm-gpu/env

# Set head node address (if not localhost:10000)
export HEAD_NODE_ADDRESS=localhost:10000

# Start GPU node
./scripts/start_gpu_node.sh
```

**Note:** Both nodes will connect to the same Ray head node specified by `HEAD_NODE_ADDRESS`.

### 3. Start the Distributed Inference System

```bash
python scripts/start_system.py \
    --model facebook/opt-125m \
    --coordinator-name migration_coordinator \
    --cpu-config configs/cpu_backend_config.json \
    --gpu-config configs/gpu_backend_config.json
```

### 4. Run Example Client

```bash
python scripts/example_client.py
```

## Configuration

### CPU Backend Config

- `enforce_eager`: Whether to use eager execution
- `enable_prefix_caching`: Enable prefix caching
- `task`: Task type (auto, text-generation, etc.)
- `torch_dtype`: Data type for tensors
- `device`: Device type (cpu)

### GPU Backend Config

- `enforce_eager`: Whether to use eager execution
- `enable_prefix_caching`: Enable prefix caching
- `task`: Task type (auto, text-generation, etc.)
- `torch_dtype`: Data type for tensors
- `lazy_load`: Enable lazy loading (must be True)

## Usage

### Python API

```python
import ray
from distributed_inference.client.inference_client import InferenceClient

# Initialize Ray
ray.init(address="auto")

# Create client
client = InferenceClient(coordinator_name="migration_coordinator")
client.connect()

# Send request
result = await client.generate(
    prompt="What is machine learning?",
    max_tokens=100,
    temperature=0.7,
)

print(result)
```

### Batch Requests

```python
prompts = [
    "What is AI?",
    "Explain deep learning.",
]

results = await client.generate_batch(
    prompts=prompts,
    max_tokens=100,
    temperature=0.7,
)
```

## Components

### Backends

- `backends/cpu_backend.py`: CPU backend implementation
- `backends/gpu_backend.py`: GPU backend with lazy loading

### Coordinator

- `migration_coordinator.py`: Manages CPU-GPU migration

### Client

- `client/inference_client.py`: Client for sending requests

### Scripts

- `scripts/start_system.py`: Start the distributed system
- `scripts/start_cpu_node.sh`: Start CPU Ray node (run in Terminal 1)
- `scripts/start_gpu_node.sh`: Start GPU Ray node (run in Terminal 2)
- `scripts/example_client.py`: Example client usage

## Notes

- The CPU backend loads weights immediately and starts prefill
- The GPU backend uses lazy loading to load weights in the background
- Migration happens automatically when GPU weights are ready
- The system handles concurrent requests and manages migration state

