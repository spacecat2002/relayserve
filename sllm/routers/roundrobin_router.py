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
import asyncio
import logging
import time
import uuid
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import ray

from sllm.logger import init_logger

from ..utils import InstanceHandle, get_worker_nodes
from .router_utils import SllmRouter

logger = init_logger(__name__)


@ray.remote
def start_instance(
    instance_id, backend, model_name, backend_config, startup_config, device
):
    logger.info(f"Starting instance {instance_id} with backend {backend}")
    if backend == "vllm":
        from sllm.backends import VllmBackend
        model_backend_cls = VllmBackend
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

async def _build_pp_placement_group(
    allocations: List[Dict[str, Any]], tp_size: int
) -> Optional[Any]:
    """Create a Ray PlacementGroup that pins each PP stage to its designated node.

    Bundle layout (strategy=PACK):
      - Bundle 0 : {"CPU": 1, "node:<stage0_IP>": 0.001}
        → coordinator (GPUBackend actor + start_instance task)
      - Bundles 1 .. tp_size : {"GPU": 1.0, "node:<stage0_IP>": 0.001}
        → stage-0 GPU workers (vLLM)
      - Bundles tp_size+1 .. 2*tp_size : {"GPU": 1.0, "node:<stage1_IP>": 0.001}
        → stage-1 GPU workers (vLLM)
      - … one block of tp_size GPU bundles per PP stage …
    """
    try:
        from ray.util.placement_group import placement_group as ray_placement_group
    except ImportError:
        logger.error("ray.util.placement_group unavailable; cannot build PP placement group")
        return None

    if not allocations:
        return None

    stage0_addr = allocations[0].get("address", "")
    if stage0_addr:
        bundles: List[Dict[str, Any]] = [{"CPU": 1, f"node:{stage0_addr}": 0.001}]
    else:
        logger.warning(
            "PP placement group: stage-0 node address unknown; "
            "coordinator bundle is not node-pinned"
        )
        bundles = [{"CPU": 1}]

    for stage in allocations:
        addr = stage.get("address", "")
        for _ in range(tp_size):
            if addr:
                bundles.append({"GPU": 1.0, f"node:{addr}": 0.001})
            else:
                logger.warning(
                    "PP placement group: stage %d address unknown; GPU bundle is not node-pinned",
                    stage.get("stage_idx", -1),
                )
                bundles.append({"GPU": 1.0})

    logger.info(
        "Creating PP placement group with %d bundles (1 CPU coordinator + %d GPU workers): %s",
        len(bundles), len(bundles) - 1, bundles,
    )
    pg = ray_placement_group(bundles, strategy="PACK")
    await pg.ready()
    logger.info("PP placement group is ready")
    return pg


async def auto_scaler(
    auto_scaling_metrics: Dict[str, int], auto_scaling_config: Dict[str, int]
) -> int:
    """
    Returns desired number of instances for a model based on the auto-scaling policy
    """

    request_count = auto_scaling_metrics.get("request_count", 0)

    min_instances = auto_scaling_config.get("min_instances", 0)
    max_instances = auto_scaling_config.get("max_instances", 10)
    target_ongoing_requests = auto_scaling_config.get("target", 10)

    desired_instances = (
        request_count + target_ongoing_requests - 1
    ) // target_ongoing_requests
    desired_instances = min(
        max_instances, max(min_instances, desired_instances)
    )

    return desired_instances


def _remove_placement_group(instance: "InstanceHandle", instance_id: str) -> None:
    """Best-effort removal of a Ray PlacementGroup stored on *instance*."""
    pg = instance.placement_group
    if pg is None:
        return
    try:
        from ray.util.placement_group import remove_placement_group
        remove_placement_group(pg)
        instance.placement_group = None
        logger.info("Removed placement group for instance %s", instance_id)
    except Exception as exc:
        logger.warning(
            "Failed to remove placement group for instance %s: %s", instance_id, exc
        )


class RoundRobinRouter(SllmRouter):
    def __init__(
        self,
        model_name: str,
        resource_requirements: Dict[str, int],
        backend: str,
        backend_config: Dict,
        device: str,
    ) -> None:
        self.model_name = model_name
        self.resource_requirements = resource_requirements
        self.backend = backend
        self.backend_config = backend_config
        self.device = device

        self.loop_interval = 1
        self.loop = asyncio.get_running_loop()
        self.request_queue = asyncio.Queue()  # type:ignore
        # Inference instance pools
        self.starting_inference_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.deleting_inference_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.ready_inference_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.instance_management_lock = asyncio.Lock()

        self.req_to_instance_id: Dict[str, str] = {}
        self.instance_to_load_status: Dict[str, bool] = {}

        self.gpu_lock = asyncio.Lock()

        self.auto_scaling_config = {}
        self.auto_scaling_lock = asyncio.Lock()

        self.request_count = 0
        self.request_count_lock = asyncio.Lock()

        self.running = False
        self.running_lock = asyncio.Lock()

        self.idle_time = 0
        self.idle_time_lock = asyncio.Lock()

        self.auto_scaler = None
        self.load_balancer = None
        self.round_robin_index = 0
        self.request_arrivals = deque()

        self.load_window_seconds = int(
            self.backend_config.get("load_window_seconds", 120)
        )
        self.forecast_horizon_seconds = int(
            self.backend_config.get("forecast_horizon_seconds", 20)
        )
        self.predictive_prewarm_threshold = float(
            self.backend_config.get("predictive_prewarm_threshold", 0.7)
        )
        logger.info(f"Created new handler for model {self.model_name}")

    async def start(
        self, auto_scaling_config: Dict[str, int], mode: str = "inference"
    ):
        self.model_loading_scheduler = ray.get_actor("model_loading_scheduler")
        if mode == "inference":
            if self.device == "gpu":
                async with self.auto_scaling_lock:
                    self.auto_scaling_config = auto_scaling_config
                prewarm = self.backend_config.get(
                    "prewarm_gpu_instances", 1
                )
                for _ in range(prewarm):
                    await self._try_rebalance_for_tp()
                    await self._create_instance(empty_instance=True)
                self.auto_scaler = asyncio.create_task(self._auto_scaler_loop())
                self.load_balancer = asyncio.create_task(self._load_balancer_loop())
            elif self.device == "cpu":
                self.cpu_instance_id = await self._create_instance()
        async with self.running_lock:
            self.running = True
        logger.info(f"Started handler for model {self.model_name}")

    async def update(
        self,
        auto_scaling_config: Optional[Dict[str, int]] = None,
    ):
        if auto_scaling_config is not None:
            async with self.auto_scaling_lock:
                self.auto_scaling_config = auto_scaling_config

        logger.info(
            f"Model {self.model_name}'s auto scaling config updated to {auto_scaling_config}"
        )

    def _new_instance_id(self):
        pattern = "{model_name}_{id}"
        return pattern.format(model_name=self.model_name, id=uuid.uuid4())

    async def _create_instance(
        self, empty_instance: bool = False, preferred_gpu_ids: Optional[List[int]] = None
    ):
        instance_id = self._new_instance_id()
        logger.info(
            f"Creating new instance {instance_id} for model {self.model_name}"
        )
        # get max_queue_length from auto_scaling_config
        if self.auto_scaling_config.get("metric", "") == "concurrency":
            max_request_length = self.auto_scaling_config.get("target", 1)
        else:
            max_request_length = 1
        logger.info(
            f"Creating new instance {instance_id} for model {self.model_name}, max queue length is {max_request_length}"
        )
        instance = InstanceHandle(
            instance_id=instance_id,
            max_queue_length=max_request_length,
            num_gpu=self.resource_requirements["num_gpus"],
            empty_instance=empty_instance,
            preferred_gpu_ids=preferred_gpu_ids,
        )
        async with self.instance_management_lock:
            self.starting_inference_instances[instance_id] = instance
        self.loop.create_task(self._start_instance(instance_id))

        return instance_id

    async def _start_instance(self, instance_id):
        async with self.instance_management_lock:
            if instance_id not in self.starting_inference_instances:
                logger.error(f"Instance {instance_id} not found")
                return
            instance = self.starting_inference_instances[instance_id]
        # Now ask model loading scheduler to load the model
        logger.info(
            f"Allocating resources for model {self.model_name} on instance {instance_id}"
        )
        backend_config = dict(self.backend_config)
        if instance.empty_instance:
            backend_config["lazy_load"] = True
        else:
            backend_config["lazy_load"] = False

        if self.device == "gpu":
            scheduler_resources = dict(self.resource_requirements)
            scheduler_resources["empty_instance"] = instance.empty_instance

            # Derive pp_size / tp_size from backend_config so the scheduler can plan
            # multi-node pipeline stages correctly.
            total_gpus = int(self.resource_requirements.get("num_gpus", 1))
            pp_size = int(
                self.backend_config.get(
                    "pipeline_parallel_size",
                    self.resource_requirements.get("pipeline_parallel_size", 1),
                )
                or 1
            )
            tp_size_cfg = int(self.backend_config.get("tensor_parallel_size", 0) or 0)
            if tp_size_cfg > 0:
                scheduler_resources["tp_size"] = tp_size_cfg
            elif total_gpus > 1:
                # Infer per-stage TP from total GPU count divided by PP stages.
                scheduler_resources["tp_size"] = max(1, total_gpus // max(pp_size, 1))
            if pp_size > 1:
                scheduler_resources["pipeline_parallel_size"] = pp_size
            if instance.empty_instance:
                scheduler_resources["num_gpus"] = 0.001
            if instance.preferred_gpu_ids:
                scheduler_resources["preferred_gpu_ids"] = instance.preferred_gpu_ids

            allocation_info = await self.model_loading_scheduler.allocate_resource.remote(
                self.model_name, instance_id, scheduler_resources
            )

            # Decode what the scheduler actually allocated.
            if isinstance(allocation_info, dict):
                startup_node = allocation_info.get("node_id")
                gpu_ids = allocation_info.get("gpu_ids", [])
                actual_pp = int(allocation_info.get("pipeline_parallel_size", 1))
                actual_tp = int(allocation_info.get("tensor_parallel_size", 1))
                stage_allocations = allocation_info.get("allocations", [])
            else:
                startup_node = allocation_info
                gpu_ids = []
                actual_pp = 1
                actual_tp = 1
                stage_allocations = []

            # ------------------------------------------------------------------ #
            #  Pipeline-parallel path: build a Placement Group that pins each    #
            #  stage to its designated Ray node.                                  #
            # ------------------------------------------------------------------ #
            if actual_pp > 1 and stage_allocations:
                logger.info(
                    "Instance %s is PP=%d TP=%d; building placement group for stages: %s",
                    instance_id, actual_pp, actual_tp,
                    [(s.get("stage_idx"), s.get("node_id"), s.get("address")) for s in stage_allocations],
                )
                from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

                pg = await _build_pp_placement_group(stage_allocations, actual_tp)
                if pg is None:
                    logger.error(
                        "Failed to build PP placement group for instance %s; aborting", instance_id
                    )
                    return

                async with instance.lock:
                    instance.placement_group = pg
                    instance.node_id = startup_node

                # The GPUBackend actor lands on bundle 0 (CPU coordinator on stage-0 node).
                # With capture_child_tasks=True, vLLM's GPU worker actors are automatically
                # placed in the remaining GPU bundles (1 … tp_size*pp_size).
                pg_strategy = PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=0,
                    placement_group_capture_child_tasks=True,
                )
                startup_config = {
                    "num_cpus": 0,
                    "num_gpus": 0,
                    "scheduling_strategy": pg_strategy,
                    # No runtime_env: Ray assigns CUDA_VISIBLE_DEVICES per worker via PG.
                }

                logger.info("Startup config (PP/PG): %s", startup_config)
                await start_instance.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=0,
                        placement_group_capture_child_tasks=True,
                    )
                ).remote(
                    instance_id,
                    self.backend,
                    self.model_name,
                    backend_config,
                    startup_config,
                    self.device,
                )

                logger.info("Started PP instance %s for model %s", instance_id, self.model_name)
                instance.backend_instance = ray.get_actor(instance_id)
                async with instance.lock:
                    instance.ready = True
                await instance.backend_instance.init_backend.remote()

                async with self.instance_management_lock:
                    self.ready_inference_instances[instance_id] = instance
                    self.starting_inference_instances.pop(instance_id)
                    lazy_load = backend_config.get("lazy_load", False)
                    self.instance_to_load_status[instance_id] = not lazy_load

                return instance_id

            # ------------------------------------------------------------------ #
            #  TP-only (single-node) path: use custom Ray resource constraints.  #
            # ------------------------------------------------------------------ #
            startup_resources = {
                "gpu_worker_node": 0.001,
                f"worker_id_{startup_node}": 0.001,
            }
            num_gpus = 0.001

        else:
            startup_node = None
            gpu_ids = []
            startup_resources = {
                "cpu_worker_node": 0.001,
            }
            num_gpus = 0

        startup_config = {
            "resources": startup_resources,
            "num_gpus": num_gpus,
        }
        logger.info(f"Startup config: {startup_config}, {backend_config}")

        if gpu_ids:
            runtime_env = {"env_vars": {"CUDA_VISIBLE_DEVICES": ",".join(map(str, gpu_ids))}}
            startup_config["runtime_env"] = runtime_env
            instance.gpu_group = gpu_ids
            logger.info(f"Assigned GPU group {gpu_ids} to instance {instance_id}")

        await start_instance.options(resources=startup_resources).remote(
            instance_id,
            self.backend,
            self.model_name,
            backend_config,
            startup_config,
            self.device,
        )
        logger.info(
            f"Started instance {instance_id} for model {self.model_name}"
        )
        instance.backend_instance = ray.get_actor(instance_id)
        async with instance.lock:
            instance.ready = True
            instance.node_id = startup_node
        await instance.backend_instance.init_backend.remote()

        # Get visible devices and store in instance handle
        if self.device == "gpu":
            gpu_ids = await instance.backend_instance.get_visible_devices.remote()
            async with instance.lock:
                instance.gpu_group = gpu_ids
                logger.info(f"Instance {instance_id} initialized on GPUs {gpu_ids}")

        async with self.instance_management_lock:
            self.ready_inference_instances[instance_id] = instance
            self.starting_inference_instances.pop(instance_id)
            if self.device == "gpu":
                lazy_load = backend_config.get("lazy_load", False)
                self.instance_to_load_status[instance_id] = (
                    False if lazy_load else True
                )
        return instance_id

    def _build_deallocate_resources(self, instance: InstanceHandle) -> Dict:
        resources = dict(self.resource_requirements)
        if self.device == "gpu":
            if instance.empty_instance:
                resources["num_gpus"] = 0.001
                resources["tp_size"] = self.resource_requirements.get("num_gpus", 1)
        return resources

    async def _predictive_load_stats(self) -> Dict[str, float]:
        now = time.time()
        while self.request_arrivals and self.request_arrivals[0] < now - self.load_window_seconds:
            self.request_arrivals.popleft()
        short_window = min(30, self.load_window_seconds)
        short_count = len([ts for ts in self.request_arrivals if ts >= now - short_window])
        long_count = len(self.request_arrivals)
        short_rate = short_count / max(float(short_window), 1.0)
        long_rate = long_count / max(float(self.load_window_seconds), 1.0)
        trend = short_rate - long_rate
        predicted_rate = max(0.0, short_rate + trend)
        predicted_requests = predicted_rate * self.forecast_horizon_seconds
        return {
            "short_rate": short_rate,
            "long_rate": long_rate,
            "trend": trend,
            "predicted_requests": predicted_requests,
        }

    async def _try_rebalance_for_tp(self):
        """Hook for subclasses (e.g. MigrationRouter) to rebalance GPU
        placement across NUMA nodes before creating a TP instance.

        The default implementation is a no-op.
        """
        pass

    async def _migrate_empty_instance(
        self, instance_id: str, target_gpu_ids: List[int]
    ) -> bool:
        """Hook for subclasses to migrate an empty instance to *target_gpu_ids*.

        The default implementation is a no-op that returns ``False``.
        """
        return False

    async def _stop_instance(self, instance_id: Optional[str] = None):
        while len(self.ready_inference_instances) <= 0:
            await asyncio.sleep(1)

        async with self.instance_management_lock:
            if instance_id is None:
                instance_id, instance = self.ready_inference_instances.popitem()
            elif instance_id in self.ready_inference_instances:
                instance = self.ready_inference_instances.pop(instance_id)
            else:
                logger.error(f"Instance {instance_id} not found")
                return
            self.deleting_inference_instances[instance_id] = instance
        logger.info(
            f"Stopping instance {instance_id} for model {self.model_name}"
        )
        self.loop.create_task(self._finish_instance(instance_id))

    async def _finish_instance(self, instance_id: str):
        async with self.instance_management_lock:
            if instance_id not in self.deleting_inference_instances:
                logger.error(f"Instance {instance_id} not found")
                return
            instance = self.deleting_inference_instances.pop(instance_id)
        async with instance.lock:
            instance.status = False

        # Release GPU lock if held
        if self.device == "gpu" and instance.gpu_locked and instance.gpu_group and instance.node_id:
            try:
                await self.model_loading_scheduler.release_gpu_lock.remote(
                    instance.node_id, instance.gpu_group
                )
                logger.info(f"Released GPU lock for instance {instance_id}")
            except Exception as e:
                logger.error(f"Failed to release GPU lock for instance {instance_id}: {e}")
            instance.gpu_locked = False

        await instance.backend_instance.stop.remote()
        ray.kill(instance.backend_instance)
        if self.device == "gpu":
            await self.model_loading_scheduler.deallocate_resource.remote(
                self.model_name,
                instance_id,
                self._build_deallocate_resources(instance),
            )
        _remove_placement_group(instance, instance_id)

    async def _shutdown_instance(self, instance_id: str):
        logger.info(
            f"Force deleting an instance (even if it is busy) for model {self.model_name}"
        )
        async with self.instance_management_lock:
            pool = self.ready_inference_instances
            if instance_id not in pool:
                logger.error(f"Instance {instance_id} not found")
                return
            instance = pool.pop(instance_id)
            async with instance.lock:
                instance.status = False
        
        # Release GPU lock if held
        if self.device == "gpu" and instance.gpu_locked and instance.gpu_group and instance.node_id:
            try:
                await self.model_loading_scheduler.release_gpu_lock.remote(
                    instance.node_id, instance.gpu_group
                )
                logger.info(f"Released GPU lock for instance {instance_id}")
            except Exception as e:
                logger.error(f"Failed to release GPU lock for instance {instance_id}: {e}")
            instance.gpu_locked = False

        await instance.backend_instance.shutdown.remote()
        ray.kill(instance.backend_instance)
        if self.device == "gpu":
            await self.model_loading_scheduler.deallocate_resource.remote(
                self.model_name,
                instance_id,
                self._build_deallocate_resources(instance),
            )
        _remove_placement_group(instance, instance_id)
        return

    async def lazy_load_weights(
        self,
        layer_idxes: List[int],
        load_method: str = "tokenwise",
        request_id: Optional[str] = None,
    ):
        if self.device == "cpu":
            return

        async with self.idle_time_lock:
            self.idle_time = 0

        # Try to find an existing instance to load weights
        # If all instances are busy (GPU locked), wait or create new one
        instance_id = None
        while instance_id is None:
            # Check existing empty instances
            potential_instances = []
            async with self.instance_management_lock:
                for i_id, instance in self.ready_inference_instances.items():
                    if self.instance_to_load_status.get(i_id, False) == False:
                        potential_instances.append(instance)
            
            # Try to lock GPU for an instance
            async with self.gpu_lock:
                for instance in potential_instances:
                    async with instance.lock:
                        if instance.gpu_group and instance.node_id:
                            # Try to acquire global GPU lock
                            lock_acquired = await self.model_loading_scheduler.acquire_gpu_lock.remote(
                                instance.node_id, instance.gpu_group
                            )
                            if lock_acquired:
                                instance.gpu_locked = True
                                instance_id = instance.instance_id
                                logger.info(f"Locked GPUs {instance.gpu_group} on node {instance.node_id} for instance {instance_id}")
                                break
            
            if instance_id is None:
                # No available instance, wait or trigger scaling (not implemented here)
                logger.info("No available GPU slots for loading weights, waiting...")
                await asyncio.sleep(0.1)

        async with self.instance_management_lock:
            if instance_id not in self.ready_inference_instances:
                logger.error(f"Instance {instance_id} not found")
                return
            instance = self.ready_inference_instances[instance_id]
            if request_id is not None:
                self.req_to_instance_id[request_id] = instance_id
            logger.info(f"Lazy load weights request for model {self.model_name}")
            
        await instance.backend_instance.lazy_load_weights.remote(layer_idxes)
        if load_method == "tokenwise":
            await self.notify_weights_loaded(instance_id)


    async def notify_weights_loaded(self, instance_id: str):
        logger.info(f"Received weight loading notification for instance {instance_id}")
        async with self.instance_management_lock:
            self.instance_to_load_status[instance_id] = True
            logger.info(f"Lazy load weights request for model {self.model_name} completed")

    async def get_load_status(self, request_id: str):
        instance_id = self.req_to_instance_id.get(request_id, None)
        if instance_id is None:
            return False
        async with self.instance_management_lock:
            return self.instance_to_load_status.get(instance_id, False)

    async def has_loaded_instance(self) -> bool:
        async with self.instance_management_lock:
            for instance_id in self.ready_inference_instances.keys():
                if self.instance_to_load_status.get(instance_id, False):
                    return True
        return False

    async def get_instance_pool_status(self) -> Dict[str, int]:
        """Return routing state for controller-side load decisions (GPU router)."""
        if self.device != "gpu":
            return {
                "loaded_ready": 0,
                "loaded_available": 0,
                "empty_ready": 0,
                "empty_starting": 0,
            }
        async with self.instance_management_lock:
            ready_instances = list(self.ready_inference_instances.values())
            starting_instances = list(self.starting_inference_instances.values())
            loaded_instances = [
                inst
                for inst in ready_instances
                if self.instance_to_load_status.get(inst.instance_id, False)
            ]
            empty_ready = [
                inst
                for inst in ready_instances
                if inst.empty_instance
                and not self.instance_to_load_status.get(inst.instance_id, False)
            ]
            empty_starting = [inst for inst in starting_instances if inst.empty_instance]

        loaded_available = 0
        for inst in loaded_instances:
            try:
                if await inst.check_request_queue():
                    loaded_available += 1
            except Exception:
                continue

        return {
            "loaded_ready": len(loaded_instances),
            "loaded_available": loaded_available,
            "empty_ready": len(empty_ready),
            "empty_starting": len(empty_starting),
        }

    async def ensure_one_instance(self) -> bool:
        """Ensure at least one GPU instance exists or is starting. Returns True if created."""
        if self.device != "gpu":
            return False
        status = await self.get_instance_pool_status()
        if status["empty_ready"] > 0 or status["empty_starting"] > 0:
            return False
        await self._create_instance(empty_instance=False)
        return True

    async def get_instance(self):
        async with self.instance_management_lock:
            for instance_id in self.ready_inference_instances.keys():
                return self.ready_inference_instances[instance_id]
        return None

    async def inference(self, request_data: dict, action: str):
        async with self.running_lock:
            if not self.running:
                return {"error": "Instance stopped"}

        self.request_arrivals.append(time.time())
        async with self.request_count_lock:
            self.request_count += 1

        if self.device == "gpu":
            async with self.idle_time_lock:
                self.idle_time = 0
            request_id = request_data.get("request_id", str(uuid.uuid4()))
            instance_id = self.req_to_instance_id.get(request_id, None)
            if instance_id is None:
                instance_allocation = self.loop.create_future()
                await self.request_queue.put(instance_allocation)
                logger.info(f"Enqueued {action} request for model {self.model_name}")
                instance_id = await instance_allocation
            if instance_id is None:
                return {"error": "Instance cancelled"}

        elif self.device == "cpu":
            instance_id = self.cpu_instance_id

        logger.info(f"{request_data}, type: {type(request_data)}")
        async with self.instance_management_lock:
            if instance_id not in self.ready_inference_instances:
                logger.error(f"Instance {instance_id} not found")
                return {"error": "Instance not found"}
            instance = self.ready_inference_instances[instance_id]

        # NOTE: `.remote(request_data)` does not work, don't know why.
        # Looks like a known issue:
        # https://github.com/ray-project/ray/issues/26283#issuecomment-1780691475
        if action == "generate":
            result = await instance.backend_instance.generate.remote(
                request_data=request_data
            )
        else:
            result = {"error": "Invalid action"}
        logger.info(f"Finished processing request")
        await instance.add_requests(-1)
        async with self.request_count_lock:
            self.request_count -= 1
        return result

    async def _teardown_instance_for_shutdown(
        self, instance_id: str, instance: InstanceHandle
    ) -> None:
        """Best-effort teardown for any instance handle (ready / starting / deleting)."""
        async with instance.lock:
            instance.status = False
        if self.device == "gpu" and instance.gpu_locked and instance.gpu_group and instance.node_id:
            try:
                await self.model_loading_scheduler.release_gpu_lock.remote(
                    instance.node_id, instance.gpu_group
                )
            except Exception:
                pass
            instance.gpu_locked = False
        be = instance.backend_instance
        if be is not None:
            try:
                await be.shutdown.remote()
            except Exception:
                pass
            try:
                ray.kill(be)
            except Exception:
                pass
        else:
            try:
                ray.kill(ray.get_actor(instance_id))
            except Exception:
                pass
        if self.device == "gpu":
            try:
                await self.model_loading_scheduler.deallocate_resource.remote(
                    self.model_name,
                    instance_id,
                    self._build_deallocate_resources(instance),
                )
            except Exception as exc:
                logger.warning(
                    "deallocate_resource on shutdown failed for %s: %s",
                    instance_id,
                    exc,
                )
        _remove_placement_group(instance, instance_id)

    async def shutdown(self):
        async with self.running_lock:
            self.running = False

        for task in (self.auto_scaler, self.load_balancer):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        while not self.request_queue.empty():
            try:
                fut = self.request_queue.get_nowait()
                fut.set_result(None)
            except Exception:
                break

        async with self.instance_management_lock:
            all_items: List[Tuple[str, InstanceHandle]] = []
            for pool in (
                self.ready_inference_instances,
                self.starting_inference_instances,
                self.deleting_inference_instances,
            ):
                all_items.extend(list(pool.items()))
                pool.clear()
            self.instance_to_load_status.clear()
            self.req_to_instance_id.clear()

        await asyncio.gather(
            *[
                self._teardown_instance_for_shutdown(iid, inst)
                for iid, inst in all_items
            ],
            return_exceptions=True,
        )
        return [iid for iid, _ in all_items]

    async def _load_balancer_loop(self):
        while True:
            instance_allocation = await self.request_queue.get()
            allocated = False
            logger.info(f"A request is waiting for model {self.model_name}")
            while not allocated:
                # 1. get ready instances
                instance_options = None
                while not instance_options:
                    await asyncio.sleep(1)
                    async with self.instance_management_lock:
                        instance_options = list(
                            self.ready_inference_instances.keys()
                        )
                    logger.info(f"{instance_options}")
                logger.info(f"Got ready instances {instance_options}")
                instance_id = instance_options[
                    self.round_robin_index % len(instance_options)
                ]
                self.round_robin_index += 1
                async with self.instance_management_lock:
                    if instance_id not in self.ready_inference_instances:
                        continue
                    instance = self.ready_inference_instances[instance_id]
                    # check if the request queue reaches max length
                    if await instance.check_request_queue():
                        allocated = await instance.add_requests(1)
                        if allocated:
                            instance_allocation.set_result(instance_id)
                    else:
                        logger.info(
                            f"Instance {instance_id} cannot add another request"
                        )
                if not allocated:
                    await asyncio.sleep(self.loop_interval)

    async def _auto_scaler_loop(self):
        while True:
            # logger.info(f"Auto-scaling for model {self.model_name}")
            async with self.auto_scaling_lock:
                auto_scaling_config = self.auto_scaling_config.copy()
            async with self.request_count_lock:
                request_count = self.request_count
            await self.model_loading_scheduler.report_model_load.remote(
                self.model_name,
                float(request_count),
            )
            auto_scaling_metrics = {"request_count": request_count}
            desired_instances = await auto_scaler(
                auto_scaling_metrics, auto_scaling_config
            )
            predictive_stats = await self._predictive_load_stats()
            target_per_instance = max(float(auto_scaling_config.get("target", 1)), 1.0)
            predicted_instances = int(
                (predictive_stats["predicted_requests"] + target_per_instance - 1)
                // target_per_instance
            )
            if predictive_stats["predicted_requests"] > 0:
                desired_instances = max(desired_instances, predicted_instances)
            async with self.instance_management_lock:
                num_running_instances = len(
                    self.starting_inference_instances
                ) + len(self.ready_inference_instances)
            if desired_instances != num_running_instances:
                logger.info(
                    "%s: %s instances, need %s instances",
                    self.model_name,
                    num_running_instances,
                    desired_instances,
                )
            if desired_instances > num_running_instances:
                logger.info("Creating new instance")
                await self._try_rebalance_for_tp()
                await self._create_instance()
            elif desired_instances < num_running_instances:
                keep_alive = auto_scaling_config.get("keep_alive", 0)
                if self.idle_time >= keep_alive:
                    logger.info(
                        f"Stopping instance, idle_time: {self.idle_time}, keep_alive: {keep_alive}"
                    )
                    await self._stop_instance()
                    async with self.idle_time_lock:
                        self.idle_time = 0
                else:
                    logger.info(
                        f"idle_time: {self.idle_time}, keep_alive: {keep_alive}"
                    )
                    async with self.idle_time_lock:
                        self.idle_time += self.loop_interval
            else:
                async with self.instance_management_lock:
                    empty_instance_count = sum(
                        1
                        for instance in self.ready_inference_instances.values()
                        if instance.empty_instance
                        and not self.instance_to_load_status.get(
                            instance.instance_id, False
                        )
                    )
                predicted_ratio = predictive_stats["predicted_requests"] / target_per_instance
                if (
                    self.device == "gpu"
                    and predicted_ratio > num_running_instances * self.predictive_prewarm_threshold
                    and empty_instance_count == 0
                ):
                    await self._try_rebalance_for_tp()
                    await self._create_instance(empty_instance=True)
            await asyncio.sleep(self.loop_interval)
