#!/bin/bash
# ---------------------------------------------------------------------------- #
#  Distributed Inference - GPU Node Startup Script                            #
#  Starts GPU Ray node with vllm-gpu virtual environment                      #
# ---------------------------------------------------------------------------- #


export VLLM_TORCH_PROFILER_DIR=/home/zwh/workspace/relayserve/gpu-trace

set -e

# Start GPU node (with GPU resources)
ray start \
    --address="localhost:10000" \
    --num-cpus=4 \
    --num-gpus=2 \
    --resources='{"worker_node": 1, "gpu_worker_0": 1}' \
    --disable-usage-stats

echo ""
echo "=========================================="
echo "GPU Node started successfully!"
echo "=========================================="
echo "Port: $GPU_NODE_PORT"
echo ""
echo "To stop this node, run:"
echo "  ray stop --port=$GPU_NODE_PORT"
echo ""

