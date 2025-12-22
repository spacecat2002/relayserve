# ---------------------------------------------------------------------------- #
#  Distributed Inference - Start Backend                                      #
#  Remote function to start CPU or GPU backend actors                          #
# ---------------------------------------------------------------------------- #
import os
import sys
import ray

# Add sllm directory to path for imports
# This ensures distributed_inference module can be found in Ray remote functions
current_dir = os.path.dirname(os.path.abspath(__file__))
sllm_dir = os.path.join(current_dir, "../..")
if sllm_dir not in sys.path:
    sys.path.insert(0, sllm_dir)

from sllm.logger import init_logger

logger = init_logger(__name__)


@ray.remote
def start_backend(
    backend_name: str,
    backend_type: str,  # "cpu" or "gpu"
    model_name: str,
    backend_config: dict,
    startup_config: dict,
):
    """
    Start a backend actor (CPU or GPU) with the given name and configuration.

    Args:
        backend_name: Name for the backend actor
        backend_type: Type of backend ("cpu" or "gpu")
        model_name: Name of the model to load
        backend_config: Backend configuration
        startup_config: Startup configuration (resources, etc.)

    Returns:
        The created backend actor
    """
    logger.info(
        f"Starting {backend_type.upper()} backend {backend_name} for model {model_name}"
    )

    if backend_type == "cpu":
        from distributed_inference.backends.cpu_backend import CpuBackend

        # CpuBackend is already a Ray remote actor class
        backend_cls = CpuBackend
    elif backend_type == "gpu":
        from distributed_inference.backends.gpu_backend import GpuBackend

        # GpuBackend is already a Ray remote actor class
        backend_cls = GpuBackend
    else:
        logger.error(f"Unknown backend type: {backend_type}")
        raise ValueError(f"Unknown backend type: {backend_type}")

    # Create backend actor with specified resources and name
    # Backend classes are already decorated with @ray.remote, so use them directly
    return backend_cls.options(
        name=backend_name,
        **startup_config,
        max_concurrency=10,
        lifetime="detached",
    ).remote(model_name, backend_config)

