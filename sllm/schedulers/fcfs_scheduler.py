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
from typing import Deque, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from sllm.logger import init_logger
from sllm.utils import get_worker_nodes

from .numa_utils import load_gpu_numa_affinity
from .scheduler_utils import SllmScheduler

logger = init_logger(__name__)


class FcfsScheduler(SllmScheduler):
    def __init__(self, scheduler_config: Optional[Mapping] = None):
        super().__init__()
        self.scheduler_config = scheduler_config

        self.queue_lock = asyncio.Lock()
        self.model_loading_queues = {}

        self.metadata_lock = asyncio.Lock()
        self.worker_nodes = {}
        self.model_instance = {}
        
        # Round-robin counters for GPU assignment per node
        self.node_gpu_counters = {}

        # Global GPU lock table: {node_id: {gpu_id: locked_bool}}
        self.gpu_locks = {}
        self.gpu_locks_lock = asyncio.Lock()

        # Per-node cold start coordination
        # Shared CPU compute resources and CPU→GPU bandwidth are per-node.
        self.cold_start_state_lock = asyncio.Lock()
        # (node_id, model_name) → {"in_progress": bool, "method": str}
        self.cold_start_status: Dict[Tuple[str, str], Dict] = {}
        # node_id → asyncio.Event (set = node ready for a new cold start)
        self.node_cold_start_ready_events: Dict[str, asyncio.Event] = {}

        self.loop = asyncio.get_running_loop()

        self.running_lock = asyncio.Lock()
        self.running = False
        self.load_window_seconds = int(
            self.scheduler_config.get("load_window_seconds", 120)
        ) if self.scheduler_config else 120
        self.forecast_horizon_seconds = int(
            self.scheduler_config.get("forecast_horizon_seconds", 20)
        ) if self.scheduler_config else 20
        self.model_load_history: Dict[str, Deque[Tuple[float, float]]] = defaultdict(deque)
        self.instance_allocations: Dict[str, Dict] = {}
        self.gpu_numa_affinity = load_gpu_numa_affinity(
            config=dict(self.scheduler_config) if self.scheduler_config else None
        )

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
            return {
                "current_load": current_load,
                "predicted_load": current_load,
                "trend": 0.0,
            }
        start_t, start_v = recent[0]
        end_t, end_v = recent[-1]
        dt = max(end_t - start_t, 1e-6)
        trend = (end_v - start_v) / dt
        predicted = max(0.0, end_v + trend * self.forecast_horizon_seconds)
        return {
            "current_load": current_load,
            "predicted_load": predicted,
            "trend": trend,
        }

    async def get_model_load_stats(self, model_name: str) -> Dict[str, float]:
        return self._predict_model_load(model_name)

    def _get_allocated_gpu_ids(self, node_id: str) -> Set[int]:
        """Return GPU IDs already allocated to instances on *node_id*."""
        allocated: Set[int] = set()
        for _iid, info in self.instance_allocations.items():
            if info.get("node_id") == node_id:
                for gid in info.get("gpu_ids", []):
                    allocated.add(int(gid))
        return allocated

    def _get_free_gpu_ids(
        self,
        node_id: str,
        total_gpus: int,
        reserved_gpu_ids: Optional[Sequence[int]] = None,
    ) -> List[int]:
        """Return GPU IDs that are free on *node_id*.

        A GPU is considered occupied if it is:
          - locked via ``gpu_locks`` (active weight loading), OR
          - allocated to an existing instance (``instance_allocations``), OR
          - in the per-batch *reserved_gpu_ids* set.
        """
        reserved_set = set(reserved_gpu_ids or [])
        node_locks = self.gpu_locks.get(node_id, {})
        allocated_set = self._get_allocated_gpu_ids(node_id)
        free_gpu_ids = []
        for gpu_id in range(total_gpus):
            if node_locks.get(gpu_id, False):
                continue
            if gpu_id in reserved_set:
                continue
            if gpu_id in allocated_set:
                continue
            free_gpu_ids.append(gpu_id)
        return free_gpu_ids

    def _pick_roundrobin_gpu_ids(
        self,
        node_id: str,
        free_gpu_ids: List[int],
        gpu_group_size: int,
    ) -> Optional[List[int]]:
        if len(free_gpu_ids) < gpu_group_size:
            return None
        if node_id not in self.node_gpu_counters:
            self.node_gpu_counters[node_id] = 0
        free_gpu_ids_sorted = sorted(free_gpu_ids)
        start_idx = self.node_gpu_counters[node_id] % len(free_gpu_ids_sorted)
        assigned = []
        for i in range(gpu_group_size):
            assigned.append(free_gpu_ids_sorted[(start_idx + i) % len(free_gpu_ids_sorted)])
        self.node_gpu_counters[node_id] = (start_idx + gpu_group_size) % len(free_gpu_ids_sorted)
        return assigned

    def _pick_balanced_numa_gpu_ids(
        self,
        node_id: str,
        free_gpu_ids: List[int],
        gpu_group_size: int,
    ) -> Optional[List[int]]:
        if len(free_gpu_ids) < gpu_group_size:
            return None
        numa_map = self.gpu_numa_affinity.get(node_id, {})
        if not numa_map or gpu_group_size <= 1:
            return self._pick_roundrobin_gpu_ids(node_id, free_gpu_ids, gpu_group_size)
        numa_to_gpu: Dict[int, List[int]] = defaultdict(list)
        for gpu_id in free_gpu_ids:
            numa_to_gpu[numa_map.get(gpu_id, -1)].append(gpu_id)
        for gpu_list in numa_to_gpu.values():
            gpu_list.sort()
        selected: List[int] = []
        selected_numa_counts: Dict[int, int] = defaultdict(int)
        while len(selected) < gpu_group_size:
            candidates = [numa for numa, gpus in numa_to_gpu.items() if gpus]
            if not candidates:
                break
            candidates.sort(
                key=lambda numa: (
                    selected_numa_counts[numa],
                    -len(numa_to_gpu[numa]),
                    numa,
                )
            )
            chosen_numa = candidates[0]
            selected.append(numa_to_gpu[chosen_numa].pop(0))
            selected_numa_counts[chosen_numa] += 1
        if len(selected) != gpu_group_size:
            return None
        return selected

    def _select_gpu_ids(
        self,
        node_id: str,
        node_info: Mapping,
        gpu_group_size: int,
        preferred_gpu_ids: Optional[Sequence[int]] = None,
        reserved_gpu_ids: Optional[Sequence[int]] = None,
    ) -> Optional[List[int]]:
        total_gpus = int(node_info.get("total_gpu", 0))
        if gpu_group_size <= 0:
            return []
        free_gpu_ids = self._get_free_gpu_ids(node_id, total_gpus, reserved_gpu_ids)
        if preferred_gpu_ids:
            preferred = [int(gid) for gid in preferred_gpu_ids]
            if all(gid in free_gpu_ids for gid in preferred):
                return preferred
            return None
        return self._pick_balanced_numa_gpu_ids(node_id, free_gpu_ids, int(gpu_group_size))

    async def acquire_gpu_lock(self, node_id: str, gpu_ids: List[int]) -> bool:
        async with self.gpu_locks_lock:
            if node_id not in self.gpu_locks:
                self.gpu_locks[node_id] = {}
            
            # Check if any requested GPU is locked
            for gpu_id in gpu_ids:
                if self.gpu_locks[node_id].get(gpu_id, False):
                    return False
            
            # Lock all requested GPUs
            for gpu_id in gpu_ids:
                self.gpu_locks[node_id][gpu_id] = True
            
            logger.info(f"Acquired GPU lock for node {node_id} GPUs {gpu_ids}")
            return True

    async def release_gpu_lock(self, node_id: str, gpu_ids: List[int]) -> None:
        async with self.gpu_locks_lock:
            if node_id in self.gpu_locks:
                for gpu_id in gpu_ids:
                    self.gpu_locks[node_id][gpu_id] = False
                logger.info(f"Released GPU lock for node {node_id} GPUs {gpu_ids}")

    async def allocate_resource(
        self, model_name: str, instance_id: str, resources: Mapping
    ) -> int:
        logger.info(f"Model {model_name} requested")
        num_gpus = resources.get("num_gpus", 0)
        gpu_group_size = resources.get("tp_size", 0)
        empty_instance = bool(resources.get("empty_instance", False))
        preferred_gpu_ids = resources.get("preferred_gpu_ids", None)
        async with self.queue_lock:
            if model_name not in self.model_loading_queues:
                self.model_loading_queues[model_name] = []
            allocation_result = self.loop.create_future()
            self.model_loading_queues[model_name].append(
                (
                    time.time(),
                    num_gpus,
                    gpu_group_size,
                    empty_instance,
                    preferred_gpu_ids,
                    allocation_result,
                )
            )
        logger.info(f"Model {model_name} added to the loading queue")
        allocation_info = await allocation_result  # 在control_loop中通过set_result设置
        
        if isinstance(allocation_info, dict):
            node_id = allocation_info.get("node_id")
        else:
            node_id = allocation_info
            
        async with self.metadata_lock:
            if model_name not in self.model_instance:
                self.model_instance[model_name] = {}
            self.model_instance[model_name][instance_id] = node_id
            gpu_ids = []
            if isinstance(allocation_info, dict):
                gpu_ids = list(allocation_info.get("gpu_ids", []))
            self.instance_allocations[instance_id] = {
                "model_name": model_name,
                "node_id": node_id,
                "gpu_ids": gpu_ids,
                "empty_instance": empty_instance,
            }
        return allocation_info

    async def deallocate_resource(
        self, model_name: str, instance_id: str, resources: Mapping
    ):
        logger.info(f"Deallocating model {model_name} instance {instance_id}")
        num_gpus = resources.get("num_gpus", 0)
        gpu_ids_to_release: List[int] = []
        release_node_id: Optional[str] = None

        async with self.metadata_lock:
            if model_name not in self.model_instance:
                logger.error(f"Model {model_name} not found")
                return
            if instance_id not in self.model_instance[model_name]:
                logger.error(f"Instance {instance_id} not found")
                return
            node_id = self.model_instance[model_name].pop(instance_id)
            alloc_info = self.instance_allocations.pop(instance_id, None)
            if alloc_info:
                gpu_ids_to_release = [int(g) for g in alloc_info.get("gpu_ids", [])]
                release_node_id = alloc_info.get("node_id", node_id)
            logger.info(f"Node {node_id} deallocated {num_gpus} GPUs")
            # Keep worker_nodes.free_gpu roughly up-to-date between control
            # loop iterations (_get_worker_nodes recalculates each time).
            if node_id in self.worker_nodes:
                self.worker_nodes[node_id]["free_gpu"] += num_gpus

        # Safety: release any GPU locks held by this instance.  The Router
        # normally releases locks *before* calling deallocate_resource, but
        # if it crashes we still need to clean up.
        if gpu_ids_to_release and release_node_id:
            await self.release_gpu_lock(release_node_id, gpu_ids_to_release)

        logger.info(f"Model {model_name} instance {instance_id} deallocated")

    async def _control_loop(self):
        logger.info("Starting control loop")
        while self.running:
            loading_requests = []
            reserved_gpu_ids_by_node: Dict[str, set] = defaultdict(set)
            async with self.queue_lock:
                for (
                    model_name,
                    loading_queue,
                ) in self.model_loading_queues.items():
                    for idx, (
                        request_time,
                        num_gpus,
                        gpu_group_size,
                        empty_instance,
                        preferred_gpu_ids,
                        allocation_result,
                    ) in enumerate(loading_queue):
                        loading_requests.append(
                            (
                                model_name,
                                idx,
                                request_time,
                                num_gpus,
                                gpu_group_size,
                                empty_instance,
                                preferred_gpu_ids,
                                allocation_result,
                            )
                        )
            if len(loading_requests) > 0:
                worker_nodes = await self._get_worker_nodes()
                logger.info(f"Worker nodes: {worker_nodes}")
                loading_requests.sort(key=lambda x: x[2])
                for (
                    model_name,
                    idx,
                    request_time,
                    num_gpus,
                    gpu_group_size,
                    empty_instance,
                    preferred_gpu_ids,
                    allocation_result,
                ) in loading_requests:
                    allocated = False
                    for node_id, node_info in worker_nodes.items():
                        if node_info["free_gpu"] < num_gpus:
                            continue
                        candidate_gpu_ids = self._select_gpu_ids(
                            node_id=node_id,
                            node_info=node_info,
                            gpu_group_size=int(gpu_group_size),
                            preferred_gpu_ids=preferred_gpu_ids,
                            reserved_gpu_ids=list(reserved_gpu_ids_by_node[node_id]),
                        )
                        if gpu_group_size > 0 and not candidate_gpu_ids:
                            continue
                        if candidate_gpu_ids:
                            reserved_gpu_ids_by_node[node_id].update(candidate_gpu_ids)
                        async with self.queue_lock:
                            if allocation_result.done():
                                allocated = True
                                break
                            try:
                                self.model_loading_queues[
                                    model_name
                                ].remove(
                                    (
                                        request_time,
                                        num_gpus,
                                        gpu_group_size,
                                        empty_instance,
                                        preferred_gpu_ids,
                                        allocation_result,
                                    )
                                )

                                allocation_info = {"node_id": node_id}
                                if candidate_gpu_ids:
                                    allocation_info["gpu_ids"] = candidate_gpu_ids
                                allocation_result.set_result(allocation_info)
                            except ValueError:
                                break
                        allocated = True
                        logger.info(
                            f"Allocated node {node_id} for model {model_name}"
                        )
                        node_info["free_gpu"] -= num_gpus
                        break
                    if not allocated:
                        logger.info(f"No available node for model {model_name}")
                await self._update_worker_nodes(worker_nodes)

            await asyncio.sleep(1)

    async def _get_worker_nodes(self):
        worker_nodes = get_worker_nodes()
        async with self.metadata_lock:
            updated_worker_nodes = copy.deepcopy(self.worker_nodes)

        async with self.gpu_locks_lock:
            for node_id, node_info in worker_nodes.items():
                if node_id == "0":
                    continue
                total_gpus = int(node_info.get("total_gpu", 0))
                # Occupied = locked (weight-loading) ∪ allocated (instance bound)
                occupied: set = set()
                if node_id in self.gpu_locks:
                    for gpu_id in range(total_gpus):
                        if self.gpu_locks[node_id].get(gpu_id, False):
                            occupied.add(gpu_id)
                occupied |= self._get_allocated_gpu_ids(node_id)
                node_info["free_gpu"] = total_gpus - len(occupied)

        for node_id, node_info in worker_nodes.items():
            if node_id not in updated_worker_nodes:
                updated_worker_nodes[node_id] = copy.deepcopy(node_info)
            else:
                updated_worker_nodes[node_id]["free_gpu"] = node_info["free_gpu"]
                updated_worker_nodes[node_id]["total_gpu"] = node_info["total_gpu"]

        async with self.metadata_lock:
            self.worker_nodes = updated_worker_nodes

        return updated_worker_nodes

    async def _update_worker_nodes(self, worker_nodes) -> None:
        async with self.metadata_lock:
            updated_worker_nodes = copy.deepcopy(self.worker_nodes)
        for node_id, node_info in worker_nodes.items():
            if node_id not in updated_worker_nodes:
                logger.error(f"Node {node_id} not found")
                continue
            updated_worker_nodes[node_id] = copy.deepcopy(node_info)
        async with self.metadata_lock:
            self.worker_nodes = updated_worker_nodes
        logger.info(f"Worker nodes updated: {updated_worker_nodes}")

    async def suggest_instance_migration(
        self,
        model_name: str,
        tp_size: int,
    ) -> Dict:
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
            free_gpu_ids = self._get_free_gpu_ids(node_id, total_gpus)
            if len(free_gpu_ids) < tp_size:
                continue
            free_numa_map: Dict[int, List[int]] = defaultdict(list)
            for gpu_id in free_gpu_ids:
                free_numa_map[numa_map.get(gpu_id, -1)].append(gpu_id)
            if len([k for k, v in free_numa_map.items() if len(v) > 0]) >= tp_size:
                continue
            dominant_numa = max(
                free_numa_map.items(), key=lambda item: len(item[1])
            )[0]
            for instance_id, info in allocations.items():
                if info.get("model_name") != model_name:
                    continue
                if info.get("node_id") != node_id:
                    continue
                if not info.get("empty_instance", False):
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
                    gid
                    for gid in free_numa_map.get(dominant_numa, [])
                    if gid not in gpu_ids
                ]
                if len(target_candidates) < len(gpu_ids):
                    continue
                return {
                    "node_id": node_id,
                    "instance_id": instance_id,
                    "source_gpu_ids": gpu_ids,
                    "target_gpu_ids": target_candidates[: len(gpu_ids)],
                }
        return {}

    # ------------------------------------------------------------------ #
    #  Per-node cold start coordination                                   #
    # ------------------------------------------------------------------ #

    def _get_node_cold_start_event(self, node_id: str) -> asyncio.Event:
        """Get or lazily create the cold-start ready event for *node_id*."""
        if node_id not in self.node_cold_start_ready_events:
            event = asyncio.Event()
            event.set()  # Initially ready (no prior cold start)
            self.node_cold_start_ready_events[node_id] = event
        return self.node_cold_start_ready_events[node_id]

    async def get_node_for_model(self, model_name: str) -> Optional[str]:
        """Return the node_id hosting the first GPU instance of *model_name*."""
        async with self.metadata_lock:
            instances = self.model_instance.get(model_name, {})
            for _instance_id, node_id in instances.items():
                return node_id
        return None

    async def start_cold_start(
        self, node_id: str, model_name: str, method: str
    ) -> None:
        """Mark a cold start as in-progress on *node_id* for *model_name*.

        Clears the node's ready event so that subsequent cold starts on
        the same node will wait.
        """
        async with self.cold_start_state_lock:
            key = (node_id, model_name)
            self.cold_start_status[key] = {
                "in_progress": True,
                "method": method,
            }
            self._get_node_cold_start_event(node_id).clear()
        logger.info(
            f"Cold start started: node={node_id}, model={model_name}, "
            f"method={method}"
        )

    async def finish_cold_start(self, node_id: str, model_name: str) -> None:
        """Mark a cold start as fully completed.

        If no other cold starts are active on *node_id*, the node's ready
        event is set as a safety net.
        """
        async with self.cold_start_state_lock:
            key = (node_id, model_name)
            self.cold_start_status.pop(key, None)
            # Set the ready event only when no other cold start is active
            any_active = any(
                k[0] == node_id and v.get("in_progress")
                for k, v in self.cold_start_status.items()
            )
            if not any_active:
                self._get_node_cold_start_event(node_id).set()
        logger.info(
            f"Cold start finished: node={node_id}, model={model_name}"
        )

    async def signal_cold_start_ready(self, node_id: str) -> None:
        """Explicitly signal that shared resources on *node_id* are free.

        Used by the controller for *tokenwise* cold starts when CPU
        computation completes.
        """
        self._get_node_cold_start_event(node_id).set()
        logger.info(f"Cold start ready signal: node={node_id}")

    async def get_cold_start_status(
        self, node_id: str, model_name: str
    ) -> Tuple[bool, Optional[str]]:
        """Return ``(is_in_progress, method)`` for *model_name* on *node_id*."""
        async with self.cold_start_state_lock:
            key = (node_id, model_name)
            status = self.cold_start_status.get(key)
            if status and status.get("in_progress"):
                return True, status.get("method")
            return False, None

    async def wait_cold_start_ready(self, node_id: str) -> None:
        """Block until the node's shared cold-start resources are free."""
        event = self._get_node_cold_start_event(node_id)
        await event.wait()

    async def notify_first_token(
        self, node_id: str, model_name: str
    ) -> None:
        """Called when the first token is generated.

        For *layerwise* cold starts this signals that the next cold start
        on the same node may begin.
        """
        async with self.cold_start_state_lock:
            key = (node_id, model_name)
            status = self.cold_start_status.get(key)
            if (
                status
                and status.get("in_progress")
                and status.get("method") == "layerwise"
            ):
                self._get_node_cold_start_event(node_id).set()
                logger.info(
                    f"Layerwise cold start: first token generated, "
                    f"ready for next cold start "
                    f"(node={node_id}, model={model_name})"
                )

    async def notify_first_token_by_instance(
        self, instance_id: str, model_name: str
    ) -> None:
        """Called directly by vllm_backend when the first token is generated.

        Resolves *node_id* from the instance allocation table and delegates
        to :meth:`notify_first_token`.
        """
        node_id = None
        async with self.metadata_lock:
            alloc = self.instance_allocations.get(instance_id)
            if alloc:
                node_id = alloc.get("node_id")
        if node_id:
            await self.notify_first_token(node_id, model_name)
        else:
            logger.warning(
                f"notify_first_token_by_instance: no allocation found "
                f"for instance {instance_id}"
            )
