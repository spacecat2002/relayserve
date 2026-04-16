# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
#  you may obtain a copy of the license at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/license-2.0                  #
#                                                                              #
#  unless required by applicable law or agreed to in writing, software         #
#  distributed under the license is distributed on an "as is" basis,           #
#  without warranties or conditions of any kind, either express or implied.    #
#  see the license for the specific language governing permissions and         #
#  limitations under the license.                                              #
# ---------------------------------------------------------------------------- #
from abc import ABC, abstractmethod
from typing import Dict, Optional

import ray

from sllm.logger import init_logger

logger = init_logger(__name__)


class SllmRouter(ABC):
    """Abstract router for a model (CPU cold-start or GPU pool)."""

    @abstractmethod
    async def start(self, auto_scaling_config: Optional[Dict[str, int]] = None):
        pass

    @abstractmethod
    async def shutdown(self):
        pass

    @abstractmethod
    async def inference(self, request_data: dict, action: str):
        pass

@ray.remote
def start_instance(
    instance_id, backend, model_name, backend_config, startup_config, device
):
    logger.info(f"Starting instance {instance_id} with backend {backend}")
    if backend == "vllm":
        if device == "cpu":
            from sllm.backends.cpu_backend import CPUBackend

            model_backend_cls = CPUBackend
        elif device == "gpu":
            from sllm.backends.gpu_backend import GPUBackend

            model_backend_cls = GPUBackend
        else:
            raise ValueError(f"Unknown device for vllm: {device}")
    else:
        logger.error(f"Unknown backend: {backend}")
        raise ValueError(f"Unknown backend: {backend}")

    model_actor_cls = ray.remote(model_backend_cls)

    runtime_env = startup_config.get("runtime_env")
    return model_actor_cls.options(
        name=instance_id,
        **startup_config,
        max_concurrency=10,
        lifetime="detached",
    ).remote(instance_id, model_name, device, backend_config, runtime_env)