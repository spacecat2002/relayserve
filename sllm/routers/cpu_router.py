# ---------------------------------------------------------------------------- #
#  CPU-side model router (single vLLM instance, SHM KV probe for GPU alignment)  #
# ---------------------------------------------------------------------------- #
import asyncio
import time
import uuid
from typing import Dict, Optional, Tuple

import ray

from sllm.logger import init_logger
from sllm.utils import InstanceHandle, ray_cpu_actor_placement_resources

from .router_utils import SllmRouter, start_instance

logger = init_logger(__name__)


class CpuModelRouter(SllmRouter):
    """One CPU inference actor per model; exposes SHM KV layout to the GPU router."""

    def __init__(
        self,
        model_name: str,
        resource_requirements: Dict[str, int],
        backend: str,
        backend_config: Dict,
    ) -> None:
        self.model_name = model_name
        self.resource_requirements = resource_requirements
        self.backend = backend
        self.backend_config = backend_config

        self._cached_shm_kv_cache_info: Optional[Tuple[int, int, int, int]] = None
        self.loop = asyncio.get_running_loop()
        self.starting_inference_instances: Dict[str, InstanceHandle] = {}
        self.ready_inference_instances: Dict[str, InstanceHandle] = {}
        self.instance_management_lock = asyncio.Lock()

        self.auto_scaling_config: Dict[str, int] = {}
        self.auto_scaling_lock = asyncio.Lock()

        self.running = False
        self.running_lock = asyncio.Lock()

        self.cpu_instance_id: Optional[str] = None
        self.model_loading_scheduler = None  # type: ignore

        logger.info(f"Created CPU handler for model {self.model_name}")

    async def start(
        self,
        auto_scaling_config: Optional[Dict[str, int]] = None,
        mode: str = "inference",
    ):
        if mode == "inference":
            async with self.auto_scaling_lock:
                self.auto_scaling_config = {}
            self.cpu_instance_id = await self._create_instance()
            deadline = time.monotonic() + float(
                self.backend_config.get("cpu_shm_kv_wait_timeout_s", 600.0)
            )
            while self._cached_shm_kv_cache_info is None:
                if time.monotonic() > deadline:
                    logger.error(
                        "Timed out waiting for CPU shm kv cache info "
                        f"(instance_id={self.cpu_instance_id})"
                    )
                    break
                await asyncio.sleep(0.05)
        async with self.running_lock:
            self.running = True
        logger.info(f"Started CPU handler for model {self.model_name}")

    def _new_instance_id(self):
        pattern = "{model_name}_{id}"
        return pattern.format(model_name=self.model_name, id=uuid.uuid4())

    async def _create_instance(self) -> str:
        instance_id = self._new_instance_id()
        logger.info(
            f"Creating new CPU instance {instance_id} for model {self.model_name}"
        )
        max_request_length = 5
        instance = InstanceHandle(
            instance_id=instance_id,
            max_queue_length=max_request_length,
            num_gpu=0,
            empty_instance=False,
        )
        async with self.instance_management_lock:
            self.starting_inference_instances[instance_id] = instance
        self.loop.create_task(self._start_instance(instance_id))
        return instance_id

    async def _start_instance(self, instance_id: str):
        async with self.instance_management_lock:
            if instance_id not in self.starting_inference_instances:
                logger.error(f"Instance {instance_id} not found")
                return
            instance = self.starting_inference_instances[instance_id]

        logger.info(
            f"Allocating resources for model {self.model_name} on instance {instance_id}"
        )
        backend_config = dict(self.backend_config)
        startup_resources, placement_node_id = ray_cpu_actor_placement_resources(
            backend_config
        )
        num_gpus = 0
        startup_config = {
            "resources": startup_resources,
            "num_gpus": num_gpus,
        }
        logger.info(
            f"CPU startup config: {startup_config}, placement_node_id={placement_node_id}"
        )

        await start_instance.options(resources=startup_resources).remote(
            instance_id,
            self.backend,
            self.model_name,
            backend_config,
            startup_config,
            "cpu",
        )
        logger.info(
            f"Started CPU instance {instance_id} for model {self.model_name}"
        )
        instance.backend_instance = ray.get_actor(instance_id)
        async with instance.lock:
            instance.ready = True
            instance.node_id = placement_node_id
        await instance.backend_instance.init_backend.remote()

        try:
            shm_info = await instance.backend_instance.get_shm_kv_cache_info.remote()
            if (
                shm_info is not None
                and isinstance(shm_info, (list, tuple))
                and len(shm_info) == 4
            ):
                self._cached_shm_kv_cache_info = (
                    int(shm_info[0]),
                    int(shm_info[1]),
                    int(shm_info[2]),
                    int(shm_info[3]),
                )
                logger.info(
                    f"Cached shm kv cache info from CPU backend: "
                    f"{self._cached_shm_kv_cache_info}"
                )
            else:
                logger.warning(
                    f"Unexpected shm kv cache info from CPU backend: {shm_info!r}"
                )
        except Exception as exc:
            logger.error("get_shm_kv_cache_info from CPU backend failed: %s", exc)

        async with self.instance_management_lock:
            self.ready_inference_instances[instance_id] = instance
            self.starting_inference_instances.pop(instance_id)

    async def _shutdown_instance(self, instance_id: str):
        logger.info(
            f"Force deleting CPU instance for model {self.model_name} id={instance_id}"
        )
        async with self.instance_management_lock:
            pool = self.ready_inference_instances
            if instance_id not in pool:
                logger.error(f"Instance {instance_id} not found")
                return
            instance = pool.pop(instance_id)
        async with instance.lock:
            instance.status = False

        await instance.backend_instance.shutdown.remote()
        ray.kill(instance.backend_instance)

    async def get_cached_shm_kv_cache_info(
        self,
    ) -> Optional[Tuple[int, int, int, int]]:
        return self._cached_shm_kv_cache_info

    async def get_cpu_placement_node_id(self) -> Optional[str]:
        """Logical Ray worker id (``cpu_worker_<id>`` suffix) for the ready CPU instance, if any."""
        async with self.instance_management_lock:
            for _instance_id, inst in self.ready_inference_instances.items():
                nid = inst.node_id
                if nid is None:
                    return None
                return str(nid)
        return None

    async def get_instance(self):
        async with self.instance_management_lock:
            for _instance_id, inst in self.ready_inference_instances.items():
                return inst
        return None

    async def inference(self, request_data: dict, action: str):
        async with self.running_lock:
            if not self.running:
                return {"error": "Instance stopped"}

        instance_id = self.cpu_instance_id
        if instance_id is None:
            return {"error": "CPU instance not ready"}

        payload_keys = sorted(
            k for k in request_data.keys() if k not in ("prompt", "messages")
        )
        logger.info(
            "CPU inference action=%s model=%s request_id=%s payload_keys=%s",
            action,
            self.model_name,
            request_data.get("request_id"),
            payload_keys,
        )
        async with self.instance_management_lock:
            if instance_id not in self.ready_inference_instances:
                logger.error(f"Instance {instance_id} not found")
                return {"error": "Instance not found"}
            instance = self.ready_inference_instances[instance_id]

        slot_taken = False
        try:
            if not await instance.add_requests(1):
                return {"error": "Instance busy"}
            slot_taken = True

            if action == "generate":
                result = await instance.backend_instance.generate.remote(
                    request_data=request_data
                )
            else:
                result = {"error": "Invalid action"}
            logger.info("Finished processing CPU request")
            return result
        finally:
            if slot_taken:
                await instance.add_requests(-1)

    async def shutdown(self):
        async with self.running_lock:
            self.running = False

        async with self.instance_management_lock:
            deleted_instance_id = list(self.ready_inference_instances.keys())
        delete_tasks = [
            self._shutdown_instance(instance_id)
            for instance_id in deleted_instance_id
        ]
        await asyncio.gather(*delete_tasks)

        return deleted_instance_id
