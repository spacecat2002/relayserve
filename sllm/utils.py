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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import ray

# Ray custom resource keys for worker placement (must match ``ray start --resources``).
_CPU_WORKER_HEAD = "cpu_worker_"
_GPU_WORKER_HEAD = "gpu_worker_"


def _logical_worker_id_from_ray_resource(resource_key: str) -> Optional[str]:
    """Map ``gpu_worker_<id>`` to the scheduler *node_id* suffix (same string as *id*)."""
    if resource_key.startswith(_GPU_WORKER_HEAD):
        return resource_key.split("_")[-1]
    return None


def get_worker_nodes():
    ray_nodes = ray.nodes()
    worker_node_info = {}
    for node in ray_nodes:
        ray_node_id = node.get("NodeID", None)
        if ray_node_id is None:
            continue
        resources = node.get("Resources", {})
        if not resources:
            continue
        node_address = node.get("NodeManagerAddress", None)
        if node_address is None or node_address == "":
            continue
        if resources.get("control_node", 0) > 0:
            continue  # Skip the control node

        total_gpu = float(resources.get("GPU", 0) or 0)
        for key in resources.keys():
            logical_id = _logical_worker_id_from_ray_resource(key)
            if logical_id is None:
                continue
            if total_gpu <= 0:
                continue
            worker_node_info[logical_id] = {
                "ray_node_id": ray_node_id,
                "address": node_address,
                "remaining_instance_slots": int(resources.get("GPU", 0)),
                "total_gpu": int(resources.get("GPU", 0)),
            }

    return worker_node_info


def ray_gpu_actor_placement_resources(
    node_id: Optional[str],
    backend_config: Dict[str, Any],
    worker_resource_fraction: Optional[float] = None,
) -> Dict[str, float]:
    """Ray ``resources=`` for placing a GPU model actor on a logical worker *node_id*."""
    frac = float(
        worker_resource_fraction
        if worker_resource_fraction is not None
        else backend_config.get("ray_worker_resource_fraction", 0.001)
    )
    out: Dict[str, float] = {}
    if bool(backend_config.get("ray_placement_include_worker_node", True)):
        out["worker_node"] = frac
    if node_id is None:
        return out
    out[f"{_GPU_WORKER_HEAD}{node_id}"] = frac
    return out


def discover_cpu_worker_placement_keys() -> List[str]:
    """Return sorted Ray resource keys ``cpu_worker_<id>`` on non-control nodes.

    Used when no explicit CPU placement is set: the controller can round-robin models across
    these slots. Requires ``ray.init()`` and workers to declare ``cpu_worker_<id>`` resources.
    """
    if not ray.is_initialized():
        return []
    head = _CPU_WORKER_HEAD
    found: set = set()
    for node in ray.nodes():
        resources = node.get("Resources") or {}
        if resources.get("control_node", 0) > 0:
            continue
        for key, raw in resources.items():
            if not isinstance(key, str) or not key.startswith(head):
                continue
            suffix = key[len(head) :]
            if not suffix:
                continue
            try:
                amount = float(raw)
            except (TypeError, ValueError):
                continue
            if amount <= 0:
                continue
            found.add(key)
    return sorted(found)


def ray_cpu_actor_placement_resources(
    backend_config: Dict[str, Any],
) -> Tuple[Dict[str, float], Optional[str]]:
    """Ray ``resources=`` for the CPU probe actor and optional logical *cpu_placement_node_id*."""
    frac = float(backend_config.get("ray_worker_resource_fraction", 0.001))
    custom = backend_config.get("cpu_placement_resources")
    if isinstance(custom, dict) and custom:
        out = {str(k): float(v) for k, v in custom.items()}
        nid = backend_config.get("cpu_placement_node_id")
        return out, (str(nid) if nid is not None else None)

    nid = backend_config.get("cpu_placement_node_id")
    if nid is not None and str(nid).strip() != "":
        logical = str(nid).strip()
        return {f"{_CPU_WORKER_HEAD}{logical}": frac}, logical

    return {"worker_node": frac}, None


@dataclass
class InstanceStatus:
    instance_id: str
    node_id: str
    concurrency: int

    model_name: Optional[str] = None
    num_current_tokens: Optional[int] = None
    resuming_latency: Optional[float] = None


@dataclass
class InstanceHandle:
    instance_id: str
    max_queue_length: int

    node_id: Optional[str] = None
    backend_instance: Optional[ray.actor.ActorHandle] = None
    ready: bool = False
    concurrency: int = 0
    num_gpu: int = 0
    empty_instance: bool = False
    load_method: Optional[str] = None
    gpu_group: Optional[List[int]] = None
    # NUMA domains touched by this instance's GPUs (from scheduler allocation PP0 stage).
    gpu_numa_nodes: Optional[Tuple[int, ...]] = None
    preferred_gpu_ids: Optional[List[int]] = None
    preferred_pp0_node_id: Optional[str] = None
    # After NUMA migration: replay prefills on the new GPU engine (ShmConnector consumer).
    kv_resume_after_init: Optional[List[List[int]]] = None
    # Migration replacement: load full weights at init (no CPU/AMX lazy_load_weights path).
    force_eager_weight_load: bool = False
    gpu_locked: bool = False
    allocation_info: Optional[Dict[str, Any]] = None
    status: bool = True
    # Ray placement group used to pin PP stage workers to designated nodes.
    # Created by the router for pp_size > 1; must be removed on teardown.
    placement_group: Optional[Any] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def add_requests(self, num_requests: int = 1):
        async with self.lock:
            if not self.ready:
                return False
            if (
                self.concurrency + num_requests > self.max_queue_length
                or self.concurrency + num_requests < 0
            ):
                return False
            self.concurrency += num_requests
            return True

    async def check_request_queue(self):
        async with self.lock:
            return self.concurrency + 1 <= self.max_queue_length

    async def get_status(self):
        async with self.lock:
            return InstanceStatus(
                self.instance_id,
                self.node_id,
                self.concurrency,
            )


from transformers import AutoTokenizer
class TokenizerWrapper:
    def __init__(
        self,
        tokenizer_name: str,
        trust_remote_code: bool = False,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=trust_remote_code,
        )

    def get_prompt_len(self, text: str) -> int:
        if not isinstance(text, str) or text == "":
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))
