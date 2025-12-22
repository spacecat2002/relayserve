#!/usr/bin/env python3
# ---------------------------------------------------------------------------- #
#  Distributed Inference - System Startup Script                              #
#  Starts CPU and GPU nodes with Ray and initializes the system              #
# ---------------------------------------------------------------------------- #
import argparse
import asyncio
import logging
import os
import sys
import time
import json
import ray

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from distributed_inference.migration_coordinator import MigrationCoordinator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def start_ray_cluster():
    """Initialize Ray cluster."""
    if not ray.is_initialized():
        ray.init(
            address="auto",  # Connect to existing cluster or start new one
            ignore_reinit_error=True,
        )
        logger.info("Ray cluster initialized")
    else:
        logger.info("Ray cluster already initialized")


async def start_distributed_inference(
    model_name: str,
    cpu_backend_config: dict,
    gpu_backend_config: dict,
    coordinator_name: str = "migration_coordinator",
):
    """
    Start the distributed inference system.

    Args:
        model_name: Name of the model to load
        cpu_backend_config: Configuration for CPU backend
        gpu_backend_config: Configuration for GPU backend
        coordinator_name: Name for the migration coordinator actor
    """
    logger.info("Starting distributed inference system...")

    # Create migration coordinator
    coordinator = MigrationCoordinator.options(
        name=coordinator_name,
        namespace="vllm",
        num_cpus=1,
        resources={"control_node": 0.1},
    ).remote(model_name, cpu_backend_config, gpu_backend_config)

    # Initialize coordinator
    await coordinator.initialize.remote()

    logger.info(f"Distributed inference system started with coordinator: {coordinator_name}")
    logger.info("CPU backend: Loading weights and ready for prefill")
    logger.info("GPU backend: Lazy loading weights in background")

    return coordinator


def main():
    parser = argparse.ArgumentParser(
        description="Start distributed inference system"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to load",
    )
    parser.add_argument(
        "--coordinator-name",
        type=str,
        default="migration_coordinator",
        help="Name for the migration coordinator actor",
    )
    parser.add_argument(
        "--cpu-config",
        type=str,
        help="Path to CPU backend config JSON file",
    )
    parser.add_argument(
        "--gpu-config",
        type=str,
        help="Path to GPU backend config JSON file",
    )

    args = parser.parse_args()

    # Default configurations
    cpu_backend_config = {
        "enforce_eager": False,
        "task": "auto",
    }

    gpu_backend_config = {
        "enforce_eager": False,
        "task": "auto",
        "lazy_load": True,
    }

    # Load configs from files if provided
    if args.cpu_config:
        with open(args.cpu_config, "r") as f:
            cpu_backend_config.update(json.load(f))

    if args.gpu_config:
        with open(args.gpu_config, "r") as f:
            gpu_backend_config.update(json.load(f))

    # Start Ray cluster
    start_ray_cluster()

    # Start distributed inference system
    coordinator = asyncio.run(
        start_distributed_inference(
            args.model,
            cpu_backend_config,
            gpu_backend_config,
            args.coordinator_name,
        )
    )

    logger.info("System started. Press Ctrl+C to stop.")
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        print("DEBUG: KeyboardInterrupt caught, starting shutdown...")
        # coordinator.shutdown.remote() returns ObjectRef, need to await it
        async def shutdown_coordinator():
            try:
                print(f"DEBUG: About to call coordinator.shutdown.remote(), coordinator={coordinator}")
                logger.info(f"About to call coordinator.shutdown.remote(), coordinator={coordinator}")
                result = await coordinator.shutdown.remote()
                print(f"DEBUG: coordinator.shutdown.remote() completed, result={result}")
                logger.info(f"coordinator.shutdown.remote() completed, result={result}")
            except Exception as e:
                print(f"DEBUG: Exception in coordinator.shutdown.remote(): {e}")
                logger.error(f"Exception in coordinator.shutdown.remote(): {e}")
                import traceback
                traceback.print_exc()
                raise
        try:
            asyncio.run(shutdown_coordinator())
            print("DEBUG: shutdown_coordinator() completed")
        except Exception as e:
            print(f"DEBUG: Exception in shutdown_coordinator(): {e}")
            logger.error(f"Exception in shutdown_coordinator(): {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("DEBUG: Calling ray.shutdown()...")
            ray.shutdown()
            logger.info("System shut down")


if __name__ == "__main__":
    main()

