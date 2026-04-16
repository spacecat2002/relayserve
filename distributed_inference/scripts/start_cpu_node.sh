#!/bin/bash
# ---------------------------------------------------------------------------- #
#  Distributed Inference - CPU Node Startup Script                            #
#  Starts CPU Ray node with vllm-cpu virtual environment                      #
# ---------------------------------------------------------------------------- #

export VLLM_TORCH_PROFILER_DIR=/home/zwh/workspace/relayserve/cpu-trace

set -e


# Start CPU node (no GPU resources)
ray start \
    --address="localhost:10000" \
    --num-cpus=64 \
    --num-gpus=0 \
    --resources='{"worker_node": 1, "cpu_worker_0": 1}' \

echo ""
echo "=========================================="
echo "CPU Node started successfully!"
echo "=========================================="
echo "Port: $CPU_NODE_PORT"
echo ""
echo "To stop this node, run:"
echo "  ray stop --port=$CPU_NODE_PORT"
echo ""

