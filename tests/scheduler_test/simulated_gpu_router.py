# ---------------------------------------------------------------------------- #
#  Simulates RoundRobinRouter's scheduler I/O without Ray. #
# ---------------------------------------------------------------------------- #
from __future__ import annotations

from typing import Any, Dict, List, Optional

from sllm.schedulers.fcfs_scheduler import FcfsScheduler
from sllm.utils import InstanceHandle


class SimulatedGpuRouter:
    """Mirror *resource* construction and scheduler calls from ``RoundRobinRouter``.

    Does not start Ray actors, call ``start_instance``, or touch ``.remote`` handles.
    Use in tests with a local :class:`FcfsScheduler` and patched ``get_worker_nodes``.
    """

    def __init__(
        self,
        scheduler: FcfsScheduler,
        model_name: str,
        resource_requirements: Dict[str, Any],
        backend_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.scheduler = scheduler
        self.model_name = model_name
        self.resource_requirements = resource_requirements
        self.backend_config = dict(backend_config or {})

    def build_scheduler_resources(self, instance: InstanceHandle) -> Dict[str, Any]:
        """Same shape as ``RoundRobinRouter._start_instance`` before ``allocate_resource``."""
        scheduler_resources = dict(self.resource_requirements)
        scheduler_resources["empty_instance"] = instance.empty_instance
        tp_size = int(
            self.resource_requirements.get(
                "tp_size", self.resource_requirements.get("num_gpus", 1)
            )
        )
        pp_size = int(self.resource_requirements.get("pp_size", 1))
        scheduler_resources["tp_size"] = tp_size
        scheduler_resources["pp_size"] = pp_size
        scheduler_resources["tensor_parallel_size"] = tp_size
        scheduler_resources["pipeline_parallel_size"] = pp_size
        if instance.preferred_gpu_ids:
            scheduler_resources["preferred_gpu_ids"] = list(instance.preferred_gpu_ids)
        return scheduler_resources

    def build_deallocate_resources(self) -> Dict[str, Any]:
        """Matches ``RoundRobinRouter._build_deallocate_resources``."""
        resources = dict(self.resource_requirements)
        resources["tp_size"] = self.resource_requirements.get(
            "tp_size", self.resource_requirements.get("num_gpus", 1)
        )
        resources["pp_size"] = self.resource_requirements.get("pp_size", 1)
        return resources

    async def allocate_via_scheduler(
        self, instance_id: str, instance: InstanceHandle
    ) -> Any:
        sched_res = self.build_scheduler_resources(instance)
        info = await self.scheduler.allocate_resource(
            self.model_name, instance_id, sched_res
        )
        if isinstance(info, dict):
            instance.node_id = info.get("node_id")
            instance.gpu_group = list(info.get("gpu_ids", []))
            instance.allocation_info = info
        else:
            instance.node_id = info
            instance.allocation_info = None
            instance.gpu_group = []
        return info

    def build_runtime_env_vars(self, allocation_info: Dict[str, Any]) -> Dict[str, str]:
        """Subset of env vars ``RoundRobinRouter`` sets when ``gpu_ids`` is non-empty."""
        tp_size = int(
            self.resource_requirements.get(
                "tp_size", self.resource_requirements.get("num_gpus", 1)
            )
        )
        pp_size = int(self.resource_requirements.get("pp_size", 1))
        startup_node = allocation_info.get("node_id")
        gpu_ids = allocation_info.get("gpu_ids", [])
        pipeline_allocations = list(allocation_info.get("allocations", []))
        env_vars: Dict[str, str] = {
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpu_ids)),
            "SLLM_TP_SIZE": str(tp_size),
            "SLLM_PP_SIZE": str(pp_size),
            "SLLM_PP0_NODE_ID": str(startup_node),
        }
        local_numa_nodes: List[int] = []
        if pipeline_allocations:
            local_numa_nodes = list(pipeline_allocations[0].get("numa_nodes", []))
        if local_numa_nodes:
            env_vars["SLLM_LOCAL_NUMA_NODE"] = str(local_numa_nodes[0])
        return env_vars

    async def deallocate_via_scheduler(self, instance: InstanceHandle) -> None:
        if instance.gpu_locked and instance.gpu_group and instance.node_id:
            await self.scheduler.release_gpu_lock(
                str(instance.node_id), list(instance.gpu_group)
            )
            instance.gpu_locked = False
        await self.scheduler.deallocate_resource(
            self.model_name,
            instance.instance_id,
            self.build_deallocate_resources(),
        )

    async def acquire_gpu_lock_for_instance(self, instance: InstanceHandle) -> bool:
        if not instance.gpu_group or instance.node_id is None:
            return False
        ok = await self.scheduler.acquire_gpu_lock(
            str(instance.node_id), list(instance.gpu_group)
        )
        instance.gpu_locked = bool(ok)
        return bool(ok)

    async def simulate_lazy_load_weight_hooks(self, instance: InstanceHandle) -> None:
        """Order aligned with ``lazy_load_weights`` after lock: commit + loaded (+ unlock).

        Skips backend ``lazy_load_weights.remote`` and migration.
        """
        await self.acquire_gpu_lock_for_instance(instance)
        await self.scheduler.mark_cold_start_committed(instance.instance_id)
        await self.scheduler.mark_weights_loaded(instance.instance_id)
        if instance.gpu_group and instance.node_id:
            await self.scheduler.release_gpu_lock(
                str(instance.node_id), list(instance.gpu_group)
            )
            instance.gpu_locked = False

    async def read_cluster_capacity(self) -> Dict[str, int]:
        return await self.scheduler.get_cluster_gpu_capacity()

    async def sync_empty_prewarm_pool(
        self,
        pool: List[InstanceHandle],
        target_count: int,
        *,
        id_prefix: str = "prewarm",
    ) -> None:
        """Grow or shrink the set of *empty_instance* handles to *target_count*.

        Used to simulate autoscaler-driven prewarm. Only pops from the end when
        shrinking—caller's responsibility that those instances are still pure
        prewarm (not cold-started).
        """
        while len(pool) < target_count:
            idx = len(pool)
            iid = f"{id_prefix}-{idx}"
            h = InstanceHandle(
                instance_id=iid,
                max_queue_length=4,
                num_gpu=int(
                    self.resource_requirements.get(
                        "tp_size",
                        self.resource_requirements.get("num_gpus", 1),
                    )
                ),
                empty_instance=True,
            )
            await self.allocate_via_scheduler(iid, h)
            pool.append(h)
        while len(pool) > target_count:
            h = pool.pop()
            await self.deallocate_via_scheduler(h)
