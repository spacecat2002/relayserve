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
import copy
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from sllm.logger import init_logger
from sllm.utils import get_worker_nodes

from .scheduler_utils import SllmScheduler, load_gpu_numa_affinity, resolve_gpu_numa_affinity

logger = init_logger(__name__)


class FcfsScheduler(SllmScheduler):
    """First-Come-First-Served scheduler for GPU resource allocation.

    Design principles (simplified):
    - Every instance (including prewarm/empty) immediately occupies its GPUs
      at ``allocate_resource`` time.  There is no "soft" prewarm quota that
      doesn't count against capacity.
    - One GPU slot per instance: no ``max_instances_per_gpu`` oversubscription.
    - TP GPUs are spread evenly across NUMA domains within the node.
    - PP stages are placed on distinct Ray nodes (enforced by default).
    - Same TP/PP configuration models are preferentially co-located on the same
      nodes for better resource organisation.
    """

    def __init__(self, scheduler_config: Optional[Mapping] = None):
        super().__init__()
        self.scheduler_config = scheduler_config

        self.queue_lock = asyncio.Lock()
        self.model_loading_queues: Dict[str, List] = {}

        self.metadata_lock = asyncio.Lock()
        self.worker_nodes: Dict[str, Dict] = {}
        self.model_instance: Dict[str, Dict[str, str]] = {}

        # Round-robin counters for GPU cycling per node (used when NUMA info is absent).
        self.node_gpu_counters: Dict[str, int] = {}

        # GPU locks: acquired during weight loading to serialise CPU→GPU transfers.
        # Structure: {node_id: {gpu_id: bool}}
        self.gpu_locks: Dict[str, Dict[int, bool]] = {}
        self.gpu_locks_lock = asyncio.Lock()

        # Per-node cold-start serialisation (shared CPU memory bandwidth).
        self.cold_start_state_lock = asyncio.Lock()
        self.cold_start_status: Dict[Tuple[str, str], Dict] = {}
        self.node_cold_start_ready_events: Dict[str, asyncio.Event] = {}

        self.loop = asyncio.get_running_loop()
        self.loop_task: Optional[asyncio.Task] = None

        self.running_lock = asyncio.Lock()
        self.running = False

        self.load_window_seconds = int(
            self.scheduler_config.get("load_window_seconds", 120)
        ) if self.scheduler_config else 120
        self.forecast_horizon_seconds = int(
            self.scheduler_config.get("forecast_horizon_seconds", 20)
        ) if self.scheduler_config else 20
        # Control loop polling interval; reduce in tests for faster feedback.
        self.control_loop_interval_s = float(
            self.scheduler_config.get("control_loop_interval_s", 1.0)
        ) if self.scheduler_config else 1.0
        self.model_load_history: Dict[str, Deque[Tuple[float, float]]] = defaultdict(deque)

        # instance_id → allocation metadata
        self.instance_allocations: Dict[str, Dict] = {}

        self.gpu_numa_affinity = load_gpu_numa_affinity(
            config=dict(self.scheduler_config) if self.scheduler_config else None
        )
        self.auto_pipeline_split = bool(
            (self.scheduler_config or {}).get("auto_pipeline_split", True)
        )
        logger.info("FCFS scheduler auto_pipeline_split=%s", self.auto_pipeline_split)

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        async with self.running_lock:
            if self.running:
                logger.error("FCFS scheduler already started")
                return
            self.running = True
        logger.info("Starting FCFS scheduler")
        self.loop_task = self.loop.create_task(self._control_loop())

    async def shutdown(self) -> None:
        async with self.running_lock:
            if not self.running:
                logger.error("FCFS scheduler not running")
                return
            self.running = False
        async with self.queue_lock:
            self.model_loading_queues = {}
        if self.loop_task is not None:
            await self.loop_task

    # ------------------------------------------------------------------ #
    #  Load tracking / forecasting                                        #
    # ------------------------------------------------------------------ #

    async def report_model_load(self, model_name: str, current_load: float) -> None:
        now = time.time()
        history = self.model_load_history[model_name]
        history.append((now, float(current_load)))
        cutoff = now - self.load_window_seconds
        while history and history[0][0] < cutoff:
            history.popleft()

    def _predict_model_load(self, model_name: str) -> Dict[str, float]:
        history = self.model_load_history.get(model_name, deque())
        if not history:
            return {"current_load": 0.0, "predicted_load": 0.0, "trend": 0.0}
        now = time.time()
        recent = [(t, v) for t, v in history if t >= now - self.load_window_seconds]
        if not recent:
            return {"current_load": 0.0, "predicted_load": 0.0, "trend": 0.0}
        current_load = recent[-1][1]
        if len(recent) == 1:
            return {"current_load": current_load, "predicted_load": current_load, "trend": 0.0}
        start_t, start_v = recent[0]
        end_t, end_v = recent[-1]
        dt = max(end_t - start_t, 1e-6)
        trend = (end_v - start_v) / dt
        predicted = max(0.0, end_v + trend * self.forecast_horizon_seconds)
        return {"current_load": current_load, "predicted_load": predicted, "trend": trend}

    async def get_model_load_stats(self, model_name: str) -> Dict[str, float]:
        return self._predict_model_load(model_name)

    # ------------------------------------------------------------------ #
    #  Cluster capacity                                                   #
    # ------------------------------------------------------------------ #

    async def get_cluster_gpu_capacity(self) -> Dict[str, int]:
        worker_nodes = await self._get_worker_nodes()
        total_gpus = sum(int(n.get("total_gpu", 0)) for n in worker_nodes.values())
        free_gpus = sum(int(n.get("remaining_gpu_slots", 0)) for n in worker_nodes.values())
        return {
            "total_gpus": total_gpus,
            "total_nodes": len(worker_nodes),
            "capacity_free_gpus": free_gpus,
        }

    async def instance_eligible_for_request(self, instance_id: str) -> bool:
        async with self.metadata_lock:
            return instance_id in self.instance_allocations

    # ------------------------------------------------------------------ #
    #  Request normalisation                                              #
    # ------------------------------------------------------------------ #

    def _normalize_parallel_request(self, resources: Mapping) -> Dict:
        requested_total_gpus = float(resources.get("num_gpus", 0) or 0)
        pipeline_parallel_size = int(
            resources.get("pipeline_parallel_size", resources.get("pp_size", 1)) or 1
        )
        if pipeline_parallel_size < 1:
            pipeline_parallel_size = 1

        has_tp_key = "tensor_parallel_size" in resources or "tp_size" in resources
        if has_tp_key:
            raw_tp = resources.get("tensor_parallel_size", resources.get("tp_size"))
            tensor_parallel_size = int(raw_tp) if raw_tp is not None else None
        else:
            tensor_parallel_size = None

        if tensor_parallel_size is None or tensor_parallel_size <= 0:
            tensor_parallel_size = (
                max(1, int(round(requested_total_gpus / pipeline_parallel_size)))
                if requested_total_gpus >= 1
                else 1
            )

        logical_total_gpus = max(
            requested_total_gpus,
            float(tensor_parallel_size * pipeline_parallel_size),
        )

        return {
            "num_gpus": logical_total_gpus,
            "tp_size": tensor_parallel_size,
            "pp_size": pipeline_parallel_size,
            "empty_instance": bool(resources.get("empty_instance", False)),
            "preferred_gpu_ids": resources.get("preferred_gpu_ids"),
            "preferred_pp0_node_id": resources.get("preferred_pp0_node_id"),
            # PP stages must be on distinct Ray nodes by default.
            "allow_stage_colocation": bool(resources.get("allow_stage_colocation", False)),
        }

    # ------------------------------------------------------------------ #
    #  GPU accounting                                                     #
    # ------------------------------------------------------------------ #

    def _count_gpu_instances_on_node(self, node_id: str) -> Dict[int, int]:
        """Per-GPU occupied-instance count on *node_id*.

        Every allocated instance (including prewarm/empty) is counted immediately;
        there is no separate capacity-committed distinction.
        """
        out: Dict[int, int] = defaultdict(int)
        for info in self.instance_allocations.values():
            for allocation in info.get("allocations", []):
                if allocation.get("node_id") != node_id:
                    continue
                for gid in allocation.get("gpu_ids", []):
                    out[int(gid)] += 1
        return out

    def _effective_gpu_counts(
        self,
        node_id: str,
        total_gpus: int,
        reserved_gpu_ids: Optional[Sequence[int]],
    ) -> Dict[int, int]:
        """Instance counts per GPU, plus *reserved_gpu_ids* from the current allocation loop."""
        counts = self._count_gpu_instances_on_node(node_id)
        effective: Dict[int, int] = {g: counts.get(g, 0) for g in range(total_gpus)}
        for gid in reserved_gpu_ids or ():
            ig = int(gid)
            if 0 <= ig < total_gpus:
                effective[ig] = effective.get(ig, 0) + 1
        return effective

    def _compute_node_free_gpus(self, node_id: str, total_gpus: int) -> int:
        """Number of unlocked, unoccupied GPUs on *node_id*."""
        counts = self._count_gpu_instances_on_node(node_id)
        node_locks = self.gpu_locks.get(node_id, {})
        return sum(
            1
            for g in range(total_gpus)
            if not node_locks.get(g, False) and not counts.get(g, 0)
        )

    def _build_per_gpu_snapshot(self, node_id: str, total_gpus: int) -> List[Dict[str, Any]]:
        eff = self._effective_gpu_counts(node_id, total_gpus, None)
        node_locks = self.gpu_locks.get(node_id, {})
        return [
            {
                "gpu_id": g,
                "occupied": bool(eff.get(g, 0)),
                "locked": bool(node_locks.get(g, False)),
                "available": not eff.get(g, 0) and not node_locks.get(g, False),
            }
            for g in range(total_gpus)
        ]

    def _can_place_tp_on_node(
        self,
        node_id: str,
        total_gpus: int,
        tp_size: int,
        reserved_gpu_ids: Optional[Sequence[int]],
    ) -> bool:
        if tp_size <= 0:
            return True
        eff = self._effective_gpu_counts(node_id, total_gpus, reserved_gpu_ids)
        node_locks = self.gpu_locks.get(node_id, {})
        available = sum(
            1
            for g in range(total_gpus)
            if not node_locks.get(g, False) and not eff.get(g, 0)
        )
        return available >= tp_size

    def _free_gpu_ids_on_node(
        self,
        node_id: str,
        total_gpus: int,
        reserved_gpu_ids: Optional[Sequence[int]] = None,
    ) -> List[int]:
        """GPU indices on *node_id* that are unlocked and unoccupied."""
        eff = self._effective_gpu_counts(node_id, total_gpus, reserved_gpu_ids)
        node_locks = self.gpu_locks.get(node_id, {})
        return [
            g for g in range(total_gpus)
            if not node_locks.get(g, False) and not eff.get(g, 0)
        ]

    def _sort_free_gpus_numa_fair(
        self,
        node_id: str,
        total_gpus: int,
        free_gpu_ids: List[int],
        reserved_gpu_ids: Optional[Sequence[int]],
    ) -> List[int]:
        """Sort *free_gpu_ids* so that GPUs on less-loaded NUMA domains come first.

        This ensures that when ``_pick_balanced_numa_gpu_ids`` selects a TP group,
        it spreads across NUMA domains that are currently least occupied.
        """
        numa_map = self.gpu_numa_affinity.get(node_id, {})
        if not numa_map or not free_gpu_ids:
            return sorted(free_gpu_ids)
        eff = self._effective_gpu_counts(node_id, total_gpus, reserved_gpu_ids)
        per_numa_load: Dict[int, int] = defaultdict(int)
        for g in range(total_gpus):
            per_numa_load[numa_map.get(g, -1)] += eff.get(g, 0)

        def sort_key(gid: int) -> Tuple[int, int, int]:
            nd = numa_map.get(gid, -1)
            return (per_numa_load[nd], nd, gid)

        return sorted(free_gpu_ids, key=sort_key)

    def _build_stage_allocation(
        self, node_id: str, gpu_ids: List[int], stage_idx: int
    ) -> Dict:
        numa_map = self.gpu_numa_affinity.get(node_id, {})
        return {
            "stage_idx": stage_idx,
            "node_id": node_id,
            "gpu_ids": [int(gpu_id) for gpu_id in gpu_ids],
            "numa_nodes": sorted(
                {int(numa_map[gpu_id]) for gpu_id in gpu_ids if gpu_id in numa_map}
            ),
        }

    def _pick_balanced_numa_gpu_ids(
        self,
        node_id: str,
        candidate_gpu_ids: List[int],
        gpu_group_size: int,
    ) -> Optional[List[int]]:
        """Pick *gpu_group_size* GPUs spreading evenly across NUMA domains.

        Greedy selection: at each step pick from the NUMA domain with the fewest
        already-selected GPUs (ties broken by domain size then domain id).
        Falls back to round-robin when no NUMA affinity data is available.
        """
        if len(candidate_gpu_ids) < gpu_group_size:
            return None

        numa_map = self.gpu_numa_affinity.get(node_id, {})
        if not numa_map or gpu_group_size <= 1:
            # Round-robin to cycle through GPUs across successive allocations.
            if node_id not in self.node_gpu_counters:
                self.node_gpu_counters[node_id] = 0
            ordered = sorted(candidate_gpu_ids)
            n = len(ordered)
            start = self.node_gpu_counters[node_id] % n
            assigned = [ordered[(start + i) % n] for i in range(gpu_group_size)]
            self.node_gpu_counters[node_id] = (start + gpu_group_size) % n
            return assigned

        numa_to_gpu: Dict[int, List[int]] = defaultdict(list)
        for gpu_id in candidate_gpu_ids:
            numa_to_gpu[numa_map.get(gpu_id, -1)].append(gpu_id)
        for gpu_list in numa_to_gpu.values():
            gpu_list.sort()

        selected: List[int] = []
        selected_numa_counts: Dict[int, int] = defaultdict(int)
        while len(selected) < gpu_group_size:
            available_numas = [numa for numa, gpus in numa_to_gpu.items() if gpus]
            if not available_numas:
                break
            available_numas.sort(
                key=lambda numa: (selected_numa_counts[numa], -len(numa_to_gpu[numa]), numa)
            )
            chosen_numa = available_numas[0]
            selected.append(numa_to_gpu[chosen_numa].pop(0))
            selected_numa_counts[chosen_numa] += 1

        return selected if len(selected) == gpu_group_size else None

    def _select_gpu_ids(
        self,
        node_id: str,
        node_info: Mapping,
        gpu_group_size: int,
        preferred_gpu_ids: Optional[Sequence[int]] = None,
        reserved_gpu_ids: Optional[Sequence[int]] = None,
    ) -> Optional[List[int]]:
        """Select *gpu_group_size* free GPUs on *node_id* with NUMA load balancing."""
        total_gpus = int(node_info.get("total_gpu", 0))
        if gpu_group_size <= 0:
            return []
        free_gpu_ids = self._free_gpu_ids_on_node(node_id, total_gpus, reserved_gpu_ids)
        if preferred_gpu_ids:
            preferred = [int(gid) for gid in preferred_gpu_ids]
            free_set = set(free_gpu_ids)
            return preferred if all(gid in free_set for gid in preferred) else None
        ordered = self._sort_free_gpus_numa_fair(node_id, total_gpus, free_gpu_ids, reserved_gpu_ids)
        return self._pick_balanced_numa_gpu_ids(node_id, ordered, int(gpu_group_size))

    # ------------------------------------------------------------------ #
    #  Model / instance topology queries                                  #
    # ------------------------------------------------------------------ #

    def _node_ids_hosting_model(self, model_name: str) -> Set[str]:
        """Node IDs that already run at least one instance of *model_name*."""
        nodes: Set[str] = set()
        for info in self.instance_allocations.values():
            if info.get("model_name") != model_name:
                continue
            for alloc in info.get("allocations") or ():
                nid = alloc.get("node_id")
                if nid:
                    nodes.add(str(nid))
        return nodes

    def _node_ids_with_tp_config(self, tp_size: int, pp_size: int) -> Set[str]:
        """Node IDs already hosting instances with matching *tp_size* and *pp_size*.

        Used to concentrate models with the same parallelism configuration on the
        same nodes, making it easier to reason about GPU layout and reducing
        fragmentation across heterogeneous workloads.
        """
        nodes: Set[str] = set()
        for info in self.instance_allocations.values():
            if info.get("tp_size") == tp_size and info.get("pp_size") == pp_size:
                for alloc in info.get("allocations") or ():
                    nid = alloc.get("node_id")
                    if nid:
                        nodes.add(str(nid))
        return nodes

    # ------------------------------------------------------------------ #
    #  GPU locking (weight-loading serialisation)                        #
    # ------------------------------------------------------------------ #

    async def acquire_gpu_lock(self, node_id: str, gpu_ids: List[int]) -> bool:
        async with self.gpu_locks_lock:
            if node_id not in self.gpu_locks:
                self.gpu_locks[node_id] = {}
            for gpu_id in gpu_ids:
                if self.gpu_locks[node_id].get(gpu_id, False):
                    return False
            for gpu_id in gpu_ids:
                self.gpu_locks[node_id][gpu_id] = True
            logger.info("Acquired GPU lock: node=%s GPUs=%s", node_id, gpu_ids)
            return True

    async def release_gpu_lock(self, node_id: str, gpu_ids: List[int]) -> None:
        async with self.gpu_locks_lock:
            if node_id in self.gpu_locks:
                for gpu_id in gpu_ids:
                    self.gpu_locks[node_id][gpu_id] = False
                logger.info("Released GPU lock: node=%s GPUs=%s", node_id, gpu_ids)

    # ------------------------------------------------------------------ #
    #  Resource allocation / deallocation                                #
    # ------------------------------------------------------------------ #

    async def allocate_resource(
        self, model_name: str, instance_id: str, resources: Mapping
    ) -> Any:
        logger.info("Model %s allocation requested (instance %s)", model_name, instance_id)
        normalized = self._normalize_parallel_request(resources)
        async with self.queue_lock:
            if model_name not in self.model_loading_queues:
                self.model_loading_queues[model_name] = []
            allocation_result = self.loop.create_future()
            self.model_loading_queues[model_name].append(
                (time.time(), normalized, allocation_result)
            )
        logger.info("Model %s added to the loading queue", model_name)
        allocation_info = await allocation_result

        node_id = (
            allocation_info.get("node_id")
            if isinstance(allocation_info, dict)
            else allocation_info
        )

        async with self.metadata_lock:
            if model_name not in self.model_instance:
                self.model_instance[model_name] = {}
            self.model_instance[model_name][instance_id] = node_id

            allocations: List[Dict] = []
            gpu_ids: List[int] = []
            if isinstance(allocation_info, dict):
                allocations = list(allocation_info.get("allocations", []))
                gpu_ids = list(allocation_info.get("gpu_ids", []))
            if not allocations and node_id is not None:
                allocations = [self._build_stage_allocation(node_id, gpu_ids, stage_idx=0)]

            actual_tp = int(normalized["tp_size"])
            actual_pp = int(normalized["pp_size"])
            if isinstance(allocation_info, dict):
                actual_tp = int(allocation_info.get("tensor_parallel_size", actual_tp))
                actual_pp = int(allocation_info.get("pipeline_parallel_size", actual_pp))

            self.instance_allocations[instance_id] = {
                "model_name": model_name,
                "node_id": node_id,
                "gpu_ids": gpu_ids,
                "allocations": allocations,
                "empty_instance": normalized["empty_instance"],
                "tp_size": actual_tp,
                "pp_size": actual_pp,
                # Tracks whether model weights are loaded (used by the router for routing).
                "weights_loaded": not normalized["empty_instance"],
            }
        return allocation_info

    async def deallocate_resource(
        self, model_name: str, instance_id: str, resources: Mapping
    ) -> None:
        logger.info("Deallocating model %s instance %s", model_name, instance_id)
        release_node_id: Optional[str] = None
        gpu_ids_to_release: List[int] = []
        affected_nodes: Set[str] = set()

        async with self.metadata_lock:
            if model_name not in self.model_instance:
                logger.error("Model %s not found", model_name)
                return
            if instance_id not in self.model_instance[model_name]:
                logger.error("Instance %s not found for model %s", instance_id, model_name)
                return
            node_id = self.model_instance[model_name].pop(instance_id)
            alloc_info = self.instance_allocations.pop(instance_id, None)
            if alloc_info:
                gpu_ids_to_release = [int(g) for g in alloc_info.get("gpu_ids", [])]
                release_node_id = alloc_info.get("node_id", node_id)
                for allocation in alloc_info.get("allocations", []):
                    alloc_nid = allocation.get("node_id")
                    if alloc_nid:
                        affected_nodes.add(str(alloc_nid))
            logger.info("Removed instance %s from allocations (was on node %s)", instance_id, node_id)

        # Update free-GPU counts after removing the instance from allocations.
        async with self.gpu_locks_lock:
            new_slots: Dict[str, int] = {}
            for nid in affected_nodes:
                if nid in self.worker_nodes:
                    total = int(self.worker_nodes[nid].get("total_gpu", 0))
                    new_slots[nid] = self._compute_node_free_gpus(nid, total)
        async with self.metadata_lock:
            for nid, slots in new_slots.items():
                if nid in self.worker_nodes:
                    self.worker_nodes[nid]["remaining_gpu_slots"] = slots

        # Safety: release any GPU locks held by this instance.
        if gpu_ids_to_release and release_node_id:
            await self.release_gpu_lock(release_node_id, gpu_ids_to_release)

        logger.info("Deallocated model %s instance %s", model_name, instance_id)

    async def _patch_instance_allocation(
        self, instance_id: str, updates: Mapping[str, Any], log_line: str
    ) -> None:
        async with self.metadata_lock:
            info = self.instance_allocations.get(instance_id)
            if info is None:
                logger.warning("%s: unknown instance_id=%s", log_line, instance_id)
                return
            for key, value in updates.items():
                info[key] = value
        logger.info("Scheduler: instance %s marked %s", instance_id, log_line)

    async def mark_cold_start_committed(self, instance_id: str) -> None:
        """No-op: kept for API compatibility.

        With the simplified GPU accounting model every instance occupies its GPUs
        from the moment ``allocate_resource`` is called, so there is no separate
        "commit" step.
        """
        pass

    async def mark_weights_loaded(self, instance_id: str) -> None:
        await self._patch_instance_allocation(
            instance_id, {"weights_loaded": True}, "weights_loaded"
        )

    # ------------------------------------------------------------------ #
    #  Control loop                                                       #
    # ------------------------------------------------------------------ #

    async def _control_loop(self) -> None:
        logger.info("Starting control loop")
        while self.running:
            loading_requests: List[Tuple] = []
            reserved_gpu_ids_by_node: Dict[str, Set[int]] = defaultdict(set)

            async with self.queue_lock:
                for model_name, loading_queue in self.model_loading_queues.items():
                    for request_time, request_info, allocation_result in loading_queue:
                        loading_requests.append(
                            (model_name, request_time, request_info, allocation_result)
                        )

            if loading_requests:
                worker_nodes = await self._get_worker_nodes()
                logger.info("Worker nodes: %s", worker_nodes)
                loading_requests.sort(key=lambda x: x[1])  # FCFS ordering

                for model_name, request_time, request_info, allocation_result in loading_requests:
                    tp_size = int(request_info["tp_size"])
                    pp_size = int(request_info["pp_size"])
                    preferred_gpu_ids = request_info.get("preferred_gpu_ids")
                    preferred_pp0_node_id = request_info.get("preferred_pp0_node_id")
                    allow_stage_colocation = bool(request_info.get("allow_stage_colocation", False))

                    async with self.metadata_lock:
                        preferred_collocate_nodes = self._node_ids_hosting_model(model_name)
                        same_config_nodes = self._node_ids_with_tp_config(tp_size, pp_size)

                    allocation_info = None

                    if pp_size > 1:
                        allocation_info = self._allocate_pipeline_stages(
                            worker_nodes=worker_nodes,
                            tp_size=tp_size,
                            pp_size=pp_size,
                            reserved_gpu_ids_by_node=reserved_gpu_ids_by_node,
                            preferred_pp0_node_id=preferred_pp0_node_id,
                            allow_stage_colocation=allow_stage_colocation,
                            preferred_collocate_nodes=preferred_collocate_nodes,
                            model_name=model_name,
                        )
                    else:
                        # TP-only: rank nodes by same-config colocation first (to concentrate
                        # models with identical parallelism on the same nodes), then by
                        # model colocation (already-running model instances), then free GPUs.
                        def _tp_node_rank(item: Tuple[str, Mapping]) -> Tuple:
                            _nid, ninfo = item
                            return (
                                0 if _nid in same_config_nodes else 1,
                                0 if _nid in preferred_collocate_nodes else 1,
                                -int(ninfo.get("remaining_gpu_slots", 0)),
                                _nid,
                            )

                        for node_id, node_info in sorted(worker_nodes.items(), key=_tp_node_rank):
                            if not self._can_place_tp_on_node(
                                node_id,
                                int(node_info.get("total_gpu", 0)),
                                tp_size,
                                list(reserved_gpu_ids_by_node[node_id]),
                            ):
                                continue
                            candidate_gpu_ids = self._select_gpu_ids(
                                node_id=node_id,
                                node_info=node_info,
                                gpu_group_size=tp_size,
                                preferred_gpu_ids=preferred_gpu_ids,
                                reserved_gpu_ids=list(reserved_gpu_ids_by_node[node_id]),
                            )
                            if not candidate_gpu_ids:
                                continue
                            reserved_gpu_ids_by_node[node_id].update(candidate_gpu_ids)
                            allocation_info = {
                                "node_id": node_id,
                                "gpu_ids": candidate_gpu_ids,
                                "allocations": [
                                    self._build_stage_allocation(node_id, candidate_gpu_ids, 0)
                                ],
                                "tensor_parallel_size": tp_size,
                                "pipeline_parallel_size": 1,
                            }
                            node_info["remaining_gpu_slots"] = max(
                                0, int(node_info.get("remaining_gpu_slots", 0)) - len(candidate_gpu_ids)
                            )
                            break

                    if allocation_info:
                        for stage in allocation_info.get("allocations", []):
                            stage_nid = stage.get("node_id")
                            if stage_nid:
                                reserved_gpu_ids_by_node[stage_nid].update(stage.get("gpu_ids", []))
                        async with self.queue_lock:
                            if allocation_result.done():
                                break
                            try:
                                self.model_loading_queues[model_name].remove(
                                    (request_time, request_info, allocation_result)
                                )
                                allocation_result.set_result(allocation_info)
                            except ValueError:
                                break
                        logger.info(
                            "Allocated node=%s for model=%s TP=%s PP=%s",
                            allocation_info.get("node_id"), model_name, tp_size, pp_size,
                        )
                    else:
                        logger.info(
                            "No available node for model=%s (TP=%s PP=%s)",
                            model_name, tp_size, pp_size,
                        )

                await self._update_worker_nodes(worker_nodes)

            await asyncio.sleep(self.control_loop_interval_s)

    # ------------------------------------------------------------------ #
    #  Worker node queries / updates                                      #
    # ------------------------------------------------------------------ #

    async def _get_worker_nodes(self) -> Dict:
        worker_nodes = get_worker_nodes()
        resolved_affinity = resolve_gpu_numa_affinity(worker_nodes, self.gpu_numa_affinity)

        async with self.gpu_locks_lock:
            for node_id, node_info in worker_nodes.items():
                total_gpus = int(node_info.get("total_gpu", 0))
                node_info["remaining_gpu_slots"] = self._compute_node_free_gpus(node_id, total_gpus)
                node_info["per_gpu"] = self._build_per_gpu_snapshot(node_id, total_gpus)

        async with self.metadata_lock:
            updated = copy.deepcopy(self.worker_nodes)
            for node_id, node_info in worker_nodes.items():
                if node_id not in updated:
                    updated[node_id] = copy.deepcopy(node_info)
                else:
                    updated[node_id]["remaining_gpu_slots"] = node_info.get("remaining_gpu_slots", 0)
                    updated[node_id]["total_gpu"] = node_info["total_gpu"]
                    updated[node_id]["per_gpu"] = node_info.get("per_gpu", [])
            self.worker_nodes = updated
            self.gpu_numa_affinity = resolved_affinity

        return updated

    async def _update_worker_nodes(self, worker_nodes: Dict) -> None:
        async with self.metadata_lock:
            updated = copy.deepcopy(self.worker_nodes)
        for node_id, node_info in worker_nodes.items():
            if node_id not in updated:
                logger.error("Node %s not found in worker_nodes during update", node_id)
                continue
            updated[node_id] = copy.deepcopy(node_info)
        async with self.metadata_lock:
            self.worker_nodes = updated
        logger.info("Worker nodes updated: %s", updated)

    # ------------------------------------------------------------------ #
    #  Pipeline stage allocation                                          #
    # ------------------------------------------------------------------ #

    def _max_hardware_gpus_per_node(self, worker_nodes: Mapping[str, Mapping]) -> int:
        return max((int(n.get("total_gpu", 0)) for n in worker_nodes.values()), default=0)

    def _allocate_pipeline_stages(
        self,
        worker_nodes: Mapping[str, Mapping],
        tp_size: int,
        pp_size: int,
        reserved_gpu_ids_by_node: Mapping[str, Set[int]],
        preferred_pp0_node_id: Optional[str] = None,
        allow_stage_colocation: bool = False,
        preferred_collocate_nodes: Optional[Set[str]] = None,
        model_name: str = "",
    ) -> Optional[Dict]:
        """Allocate *pp_size* pipeline stages, each on *tp_size* NUMA-balanced GPUs.

        Each stage is placed on a **distinct** Ray node by default.  Stage colocation
        (multiple PP stages on the same node) is only allowed when
        *allow_stage_colocation* is True and the cluster has fewer nodes than stages.
        A warning is always logged when colocation occurs.
        """
        if tp_size <= 0 or pp_size <= 1:
            return None

        collocate = preferred_collocate_nodes or set()

        distinct_result = self._try_allocate_pipeline_stages(
            worker_nodes=worker_nodes,
            tp_size=tp_size,
            pp_size=pp_size,
            reserved_gpu_ids_by_node=reserved_gpu_ids_by_node,
            preferred_pp0_node_id=preferred_pp0_node_id,
            require_distinct_nodes=True,
            preferred_collocate_nodes=collocate,
            model_name=model_name,
        )
        if distinct_result:
            return distinct_result

        if not allow_stage_colocation:
            logger.warning(
                "PP allocation failed for model=%s (TP=%s PP=%s): not enough distinct Ray nodes "
                "(%d available). Set allow_stage_colocation=True in backend_config to permit "
                "multiple PP stages on the same node as a last-resort fallback.",
                model_name, tp_size, pp_size, len(worker_nodes),
            )
            return None

        logger.warning(
            "PP allocation for model=%s (TP=%s PP=%s) falling back to stage colocation: "
            "only %d Ray nodes available for %d PP stages. Performance may be suboptimal.",
            model_name, tp_size, pp_size, len(worker_nodes), pp_size,
        )
        return self._try_allocate_pipeline_stages(
            worker_nodes=worker_nodes,
            tp_size=tp_size,
            pp_size=pp_size,
            reserved_gpu_ids_by_node=reserved_gpu_ids_by_node,
            preferred_pp0_node_id=preferred_pp0_node_id,
            require_distinct_nodes=False,
            preferred_collocate_nodes=collocate,
            model_name=model_name,
        )

    def _try_allocate_pipeline_stages(
        self,
        worker_nodes: Mapping[str, Mapping],
        tp_size: int,
        pp_size: int,
        reserved_gpu_ids_by_node: Mapping[str, Set[int]],
        preferred_pp0_node_id: Optional[str],
        require_distinct_nodes: bool,
        preferred_collocate_nodes: Optional[Set[str]] = None,
        model_name: str = "",
    ) -> Optional[Dict]:
        stage_allocations: List[Dict] = []
        used_nodes: Set[str] = set()
        local_reserved: Dict[str, Set[int]] = defaultdict(set)
        collocate = preferred_collocate_nodes or set()

        for stage_idx in range(pp_size):
            def _stage_node_rank(item: Tuple[str, Mapping], _stage: int = stage_idx) -> Tuple:
                nid, ninfo = item
                # Stage-0: honour explicit pp0 node hint, then model colocation.
                pp0_match = (
                    0 if (_stage == 0 and preferred_pp0_node_id and nid == preferred_pp0_node_id)
                    else 1
                )
                collocate_pref = 0 if (_stage == 0 and nid in collocate) else 1
                # Penalise already-used nodes when distinct placement is required.
                already_used = 1 if (require_distinct_nodes and nid in used_nodes) else 0
                headroom = int(ninfo.get("remaining_gpu_slots", 0))
                return (pp0_match, collocate_pref, already_used, -headroom, nid)

            chosen_stage = None
            for node_id, node_info in sorted(worker_nodes.items(), key=_stage_node_rank):
                if require_distinct_nodes and node_id in used_nodes:
                    continue
                total_n = int(node_info.get("total_gpu", 0))
                merged_reserved = list(
                    set(reserved_gpu_ids_by_node.get(node_id, set())) | local_reserved[node_id]
                )
                if not self._can_place_tp_on_node(node_id, total_n, tp_size, merged_reserved):
                    continue
                candidate_gpu_ids = self._select_gpu_ids(
                    node_id=node_id,
                    node_info=node_info,
                    gpu_group_size=tp_size,
                    reserved_gpu_ids=merged_reserved,
                )
                if not candidate_gpu_ids:
                    continue
                chosen_stage = self._build_stage_allocation(node_id, candidate_gpu_ids, stage_idx)
                local_reserved[node_id].update(candidate_gpu_ids)
                used_nodes.add(node_id)
                break

            if chosen_stage is None:
                return None
            stage_allocations.append(chosen_stage)

        for stage in stage_allocations:
            nid = stage["node_id"]
            # Expose the node's IP address so the router can build PG node-affinity constraints.
            stage["address"] = str(worker_nodes.get(nid, {}).get("address") or "")
            worker_nodes[nid]["remaining_gpu_slots"] = max(
                0,
                int(worker_nodes[nid].get("remaining_gpu_slots", 0)) - len(stage["gpu_ids"]),
            )

        first_stage = stage_allocations[0]
        return {
            "node_id": first_stage["node_id"],
            "gpu_ids": first_stage["gpu_ids"],
            "allocations": stage_allocations,
            "tensor_parallel_size": tp_size,
            "pipeline_parallel_size": pp_size,
        }

    # ------------------------------------------------------------------ #
    #  NUMA migration suggestion                                          #
    # ------------------------------------------------------------------ #

    async def suggest_instance_migration(
        self,
        model_name: str,
        tp_size: int,
    ) -> Dict:
        """Suggest moving an instance when free GPUs are concentrated in a single NUMA domain.

        Prefers migrating empty/prewarm instances (no KV state to transfer) over
        loaded ones.  Only considers instances whose GPUs are not currently locked.
        """
        if tp_size <= 1:
            return {}
        worker_nodes = await self._get_worker_nodes()
        async with self.metadata_lock:
            allocations = copy.deepcopy(self.instance_allocations)

        for node_id, node_info in worker_nodes.items():
            numa_map = self.gpu_numa_affinity.get(node_id, {})
            if not numa_map:
                continue
            total_gpus = int(node_info.get("total_gpu", 0))
            free_gpu_ids = self._free_gpu_ids_on_node(node_id, total_gpus)
            if len(free_gpu_ids) < tp_size:
                continue

            free_numa_map: Dict[int, List[int]] = defaultdict(list)
            for gpu_id in free_gpu_ids:
                free_numa_map[numa_map.get(gpu_id, -1)].append(gpu_id)

            # If free GPUs already span enough distinct NUMA domains, no migration needed.
            if len([k for k, v in free_numa_map.items() if v]) >= tp_size:
                continue

            dominant_numa = max(free_numa_map.items(), key=lambda item: len(item[1]))[0]

            candidates: List[Tuple[str, Dict, List[int]]] = []
            for instance_id, info in allocations.items():
                if info.get("model_name") != model_name or info.get("node_id") != node_id:
                    continue
                gpu_ids = [int(gid) for gid in info.get("gpu_ids", [])]
                if not gpu_ids:
                    continue
                if all(numa_map.get(gid, -1) == dominant_numa for gid in gpu_ids):
                    continue
                async with self.gpu_locks_lock:
                    node_locks = self.gpu_locks.get(node_id, {})
                    if any(node_locks.get(gid, False) for gid in gpu_ids):
                        continue
                target_candidates = [
                    gid for gid in free_numa_map.get(dominant_numa, []) if gid not in gpu_ids
                ]
                if len(target_candidates) < len(gpu_ids):
                    continue
                candidates.append((instance_id, info, gpu_ids))

            if not candidates:
                continue

            # Prefer migrating empty (prewarm) instances — no KV state to restore.
            candidates.sort(key=lambda t: (0 if t[1].get("empty_instance") else 1, t[0]))
            instance_id, _info, gpu_ids = candidates[0]
            target_candidates = [
                gid for gid in free_numa_map.get(dominant_numa, []) if gid not in gpu_ids
            ]
            return {
                "node_id": node_id,
                "target_node_id": node_id,
                "instance_id": instance_id,
                "source_gpu_ids": gpu_ids,
                "target_gpu_ids": target_candidates[: len(gpu_ids)],
            }
        return {}

    # ------------------------------------------------------------------ #
    #  Per-node cold-start coordination                                   #
    # ------------------------------------------------------------------ #

    def _get_node_cold_start_event(self, node_id: str) -> asyncio.Event:
        if node_id not in self.node_cold_start_ready_events:
            event = asyncio.Event()
            event.set()
            self.node_cold_start_ready_events[node_id] = event
        return self.node_cold_start_ready_events[node_id]

    async def get_node_for_model(self, model_name: str) -> Optional[str]:
        async with self.metadata_lock:
            for _instance_id, node_id in self.model_instance.get(model_name, {}).items():
                return node_id
        return None

    async def start_cold_start(self, node_id: str, model_name: str, method: str) -> None:
        async with self.cold_start_state_lock:
            self.cold_start_status[(node_id, model_name)] = {"in_progress": True, "method": method}
            self._get_node_cold_start_event(node_id).clear()
        logger.info("Cold start started: node=%s model=%s method=%s", node_id, model_name, method)

    async def finish_cold_start(self, node_id: str, model_name: str) -> None:
        async with self.cold_start_state_lock:
            self.cold_start_status.pop((node_id, model_name), None)
            any_active = any(
                k[0] == node_id and v.get("in_progress")
                for k, v in self.cold_start_status.items()
            )
            if not any_active:
                self._get_node_cold_start_event(node_id).set()
        logger.info("Cold start finished: node=%s model=%s", node_id, model_name)

    async def signal_cold_start_ready(self, node_id: str) -> None:
        self._get_node_cold_start_event(node_id).set()
        logger.info("Cold start ready signal: node=%s", node_id)

    async def get_cold_start_status(
        self, node_id: str, model_name: str
    ) -> Tuple[bool, Optional[str]]:
        async with self.cold_start_state_lock:
            status = self.cold_start_status.get((node_id, model_name))
            if status and status.get("in_progress"):
                return True, status.get("method")
            return False, None

    async def wait_cold_start_ready(self, node_id: str) -> None:
        await self._get_node_cold_start_event(node_id).wait()

    async def notify_first_token(self, node_id: str, model_name: str) -> None:
        async with self.cold_start_state_lock:
            status = self.cold_start_status.get((node_id, model_name))
            if status and status.get("in_progress") and status.get("method") == "layerwise":
                self._get_node_cold_start_event(node_id).set()
                logger.info(
                    "Layerwise cold start: first token → node ready (node=%s model=%s)",
                    node_id, model_name,
                )

    async def notify_first_token_by_instance(
        self, instance_id: str, model_name: str
    ) -> None:
        node_id = None
        async with self.metadata_lock:
            alloc = self.instance_allocations.get(instance_id)
            if alloc:
                node_id = alloc.get("node_id")
        if node_id:
            await self.notify_first_token(node_id, model_name)
        else:
            logger.warning(
                "notify_first_token_by_instance: no allocation for instance %s", instance_id
            )
