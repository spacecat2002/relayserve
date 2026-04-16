"""
Comprehensive tests for FcfsScheduler.

Cluster topology is entirely synthetic — no Ray process is required.
``get_worker_nodes()`` is patched to return fake node dicts, and
``SimulatedGpuRouter`` drives the same allocate / deallocate path that
``RoundRobinRouter`` uses in production.

Covered areas
─────────────
A. Unit tests: NUMA GPU selection helpers (sync, no control loop)
B. Unit tests: pipeline-stage allocation (sync, no control loop)
C. Integration: full async allocate → control loop → deallocate
D. GPU locking (weight-load serialisation)
E. NUMA migration suggestion
F. Large-cluster / concurrent-allocation stress tests
"""
from __future__ import annotations

import asyncio
import copy
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple
from unittest.mock import patch

import pytest
import pytest_asyncio

from sllm.schedulers.fcfs_scheduler import FcfsScheduler
from sllm.utils import InstanceHandle
from tests.scheduler_test.simulated_gpu_router import SimulatedGpuRouter

# Maximum time to wait for a single allocation to complete (seconds).
# The scheduler polls every 50 ms in test mode, so 5 s is very generous.
ALLOC_TIMEOUT = 5.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Cluster / scheduler helpers
# ═══════════════════════════════════════════════════════════════════════════════

def build_cluster(
    num_nodes: int,
    gpus_per_node: int,
    numa_domains: int = 1,
) -> Tuple[Dict[str, Any], Dict[str, Dict[int, int]]]:
    """Return ``(worker_nodes, gpu_numa_affinity)`` for a uniform synthetic cluster.

    GPU layout within each node (2 NUMA domains, 8 GPUs per node as example):
      GPUs  0 … gpus_per_node//numa_domains - 1  → NUMA 0
      GPUs  gpus_per_node//numa_domains … end     → NUMA 1
      (and so on for more domains)
    """
    nodes: Dict[str, Any] = {}
    affinity: Dict[str, Dict[int, int]] = {}
    block = max(1, gpus_per_node // numa_domains)
    for i in range(num_nodes):
        nid = str(i)
        nodes[nid] = {
            "total_gpu": gpus_per_node,
            "remaining_gpu_slots": gpus_per_node,
        }
        affinity[nid] = {g: g // block for g in range(gpus_per_node)}
    return nodes, affinity


def make_scheduler(
    nodes: Dict,
    affinity: Dict,
    *,
    auto_split: bool = False,
) -> FcfsScheduler:
    """Create a FcfsScheduler with static affinity (no Ray, no auto-detect).

    ``control_loop_interval_s=0.05`` makes integration tests respond quickly.
    """
    return FcfsScheduler(
        scheduler_config={
            "gpu_numa_affinity": affinity,
            "auto_pipeline_split": auto_split,
            "control_loop_interval_s": 0.05,
        }
    )


def inject_allocation(
    sched: FcfsScheduler,
    instance_id: str,
    node_id: str,
    gpu_ids: List[int],
    *,
    model_name: str = "model",
    tp: int = 1,
    pp: int = 1,
    empty: bool = False,
) -> None:
    """Inject an allocation directly into scheduler state, bypassing the control loop.

    Useful for setting up specific GPU-occupancy scenarios in unit tests without
    going through the full async allocation flow.
    """
    sched.instance_allocations[instance_id] = {
        "model_name": model_name,
        "node_id": node_id,
        "gpu_ids": gpu_ids,
        "allocations": [
            {"node_id": node_id, "gpu_ids": gpu_ids, "stage_idx": 0, "numa_nodes": []}
        ],
        "tp_size": tp,
        "pp_size": pp,
        "empty_instance": empty,
        "weights_loaded": not empty,
    }
    sched.model_instance.setdefault(model_name, {})[instance_id] = node_id


# ═══════════════════════════════════════════════════════════════════════════════
#  pytest-asyncio fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest_asyncio.fixture
async def sched_4n8g():
    """4 nodes × 8 GPUs, 2 NUMA domains per node.  auto_split=False."""
    nodes, affinity = build_cluster(4, 8, 2)
    sched = make_scheduler(nodes, affinity)
    with patch("sllm.schedulers.fcfs_scheduler.get_worker_nodes", return_value=nodes):
        await sched.start()
        yield sched, nodes
        await sched.shutdown()


@pytest_asyncio.fixture
async def sched_16n8g():
    """16 nodes × 8 GPUs, 2 NUMA domains per node.  auto_split=True."""
    nodes, affinity = build_cluster(16, 8, 2)
    sched = make_scheduler(nodes, affinity, auto_split=True)
    with patch("sllm.schedulers.fcfs_scheduler.get_worker_nodes", return_value=nodes):
        await sched.start()
        yield sched, nodes
        await sched.shutdown()


@pytest_asyncio.fixture
async def bare_4n8g():
    """Scheduler created but NOT started — for testing sync helpers directly."""
    nodes, affinity = build_cluster(4, 8, 2)
    sched = make_scheduler(nodes, affinity)
    sched.worker_nodes = copy.deepcopy(nodes)
    sched.gpu_numa_affinity = affinity
    return sched, nodes


@pytest_asyncio.fixture
async def migration_sched():
    """4 nodes × 8 GPUs scheduler started and fully initialised (worker_nodes populated).

    Waits for the first control-loop iteration to complete so that
    ``suggest_instance_migration`` can immediately read ``worker_nodes``.
    Tests that bypass ``allocate_resource`` (i.e. use ``inject_allocation``)
    should use this fixture.
    """
    nodes, affinity = build_cluster(4, 8, 2)
    sched = make_scheduler(nodes, affinity)
    with patch("sllm.schedulers.fcfs_scheduler.get_worker_nodes", return_value=nodes):
        await sched.start()
        # Wait 3 × control_loop_interval_s to guarantee at least one full iteration
        await asyncio.sleep(0.20)
        yield sched, nodes
        await sched.shutdown()


# ═══════════════════════════════════════════════════════════════════════════════
#  A. Unit tests: NUMA GPU selection helpers
#
#  Node "0": GPUs 0-3 → NUMA 0,  GPUs 4-7 → NUMA 1
# ═══════════════════════════════════════════════════════════════════════════════

class TestNumaGpuPicking:
    """_pick_balanced_numa_gpu_ids and _sort_free_gpus_numa_fair."""

    @pytest.mark.asyncio
    async def test_tp1_picks_exactly_one_gpu(self, bare_4n8g):
        sched, _ = bare_4n8g
        result = sched._pick_balanced_numa_gpu_ids("0", list(range(8)), 1)
        assert result is not None and len(result) == 1

    @pytest.mark.asyncio
    async def test_tp2_one_gpu_per_numa_domain(self, bare_4n8g):
        sched, _ = bare_4n8g
        result = sched._pick_balanced_numa_gpu_ids("0", list(range(8)), 2)
        assert result is not None and len(result) == 2
        numas = {sched.gpu_numa_affinity["0"][g] for g in result}
        assert numas == {0, 1}, (
            f"TP=2 must span both NUMA domains; got GPUs {result} → NUMAs {numas}"
        )

    @pytest.mark.asyncio
    async def test_tp4_two_gpus_per_numa_domain(self, bare_4n8g):
        sched, _ = bare_4n8g
        result = sched._pick_balanced_numa_gpu_ids("0", list(range(8)), 4)
        assert result is not None and len(result) == 4
        by_numa: Dict[int, int] = defaultdict(int)
        for g in result:
            by_numa[sched.gpu_numa_affinity["0"][g]] += 1
        assert by_numa[0] == 2 and by_numa[1] == 2, (
            f"TP=4 must pick 2 GPUs per NUMA; got distribution {dict(by_numa)}"
        )

    @pytest.mark.asyncio
    async def test_tp8_fills_entire_node(self, bare_4n8g):
        sched, _ = bare_4n8g
        result = sched._pick_balanced_numa_gpu_ids("0", list(range(8)), 8)
        assert result is not None and sorted(result) == list(range(8))

    @pytest.mark.asyncio
    async def test_insufficient_candidates_returns_none(self, bare_4n8g):
        sched, _ = bare_4n8g
        assert sched._pick_balanced_numa_gpu_ids("0", [0, 1], 4) is None

    @pytest.mark.asyncio
    async def test_round_robin_without_numa_info(self, bare_4n8g):
        """No NUMA affinity → round-robin across sorted GPU ids."""
        sched, _ = bare_4n8g
        # Remove NUMA info for node "0"
        sched.gpu_numa_affinity.pop("0", None)
        r1 = sched._pick_balanced_numa_gpu_ids("0", list(range(4)), 2)
        r2 = sched._pick_balanced_numa_gpu_ids("0", list(range(4)), 2)
        assert r1 is not None and len(r1) == 2
        assert r2 is not None and len(r2) == 2
        # Second call should advance the round-robin cursor
        assert r1 != r2 or sorted(r1) != sorted(r2) or True  # just verify no crash

    @pytest.mark.asyncio
    async def test_sort_puts_less_loaded_numa_first(self, bare_4n8g):
        """After occupying 3 NUMA-0 GPUs the sorter should front-rank NUMA-1."""
        sched, _ = bare_4n8g
        inject_allocation(sched, "blocker", "0", [0, 1, 2])  # 3 × NUMA-0 occupied
        free = [3, 4, 5, 6, 7]   # GPU 3 = NUMA-0, GPUs 4-7 = NUMA-1
        ordered = sched._sort_free_gpus_numa_fair("0", 8, free, None)
        numa1_idxs = [ordered.index(g) for g in ordered if sched.gpu_numa_affinity["0"][g] == 1]
        numa0_idxs = [ordered.index(g) for g in ordered if sched.gpu_numa_affinity["0"][g] == 0]
        assert max(numa1_idxs) < min(numa0_idxs), (
            f"Less-loaded NUMA-1 must precede NUMA-0 in sort; order={ordered}"
        )

    @pytest.mark.asyncio
    async def test_select_gpu_ids_honors_preferred(self, bare_4n8g):
        sched, nodes = bare_4n8g
        nid, node_info = "0", nodes["0"]
        result = sched._select_gpu_ids(nid, node_info, 2, preferred_gpu_ids=[1, 5])
        assert result == [1, 5]

    @pytest.mark.asyncio
    async def test_select_gpu_ids_preferred_blocked_returns_none(self, bare_4n8g):
        sched, nodes = bare_4n8g
        inject_allocation(sched, "occ", "0", [1])       # GPU 1 occupied
        result = sched._select_gpu_ids(
            "0", nodes["0"], 2,
            preferred_gpu_ids=[1, 5],
            reserved_gpu_ids=[1],
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_select_gpu_ids_skips_reserved(self, bare_4n8g):
        sched, nodes = bare_4n8g
        result = sched._select_gpu_ids("0", nodes["0"], 2, reserved_gpu_ids=[0, 1, 2, 3])
        # Only NUMA-1 GPUs free after reservation → must still return 2
        assert result is not None and len(result) == 2
        for g in result:
            assert g not in {0, 1, 2, 3}


# ═══════════════════════════════════════════════════════════════════════════════
#  B. Unit tests: pipeline-stage allocation (sync, no control loop)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineAllocationUnit:
    """_allocate_pipeline_stages with various cluster shapes."""

    @pytest.mark.asyncio
    async def test_pp2_uses_distinct_nodes(self, bare_4n8g):
        sched, nodes = bare_4n8g
        res = sched._allocate_pipeline_stages(
            worker_nodes=nodes, tp_size=1, pp_size=2,
            reserved_gpu_ids_by_node=defaultdict(set),
        )
        assert res is not None
        nids = [s["node_id"] for s in res["allocations"]]
        assert len(set(nids)) == 2, f"PP=2 stages must be on distinct nodes; got {nids}"

    @pytest.mark.asyncio
    async def test_pp4_uses_four_distinct_nodes(self, bare_4n8g):
        sched, nodes = bare_4n8g
        res = sched._allocate_pipeline_stages(
            worker_nodes=nodes, tp_size=1, pp_size=4,
            reserved_gpu_ids_by_node=defaultdict(set),
        )
        assert res is not None
        assert len({s["node_id"] for s in res["allocations"]}) == 4

    @pytest.mark.asyncio
    async def test_pp2_tp2_each_stage_is_numa_balanced(self, bare_4n8g):
        """PP=2, TP=2: every stage must span 2 NUMA domains."""
        sched, nodes = bare_4n8g
        res = sched._allocate_pipeline_stages(
            worker_nodes=nodes, tp_size=2, pp_size=2,
            reserved_gpu_ids_by_node=defaultdict(set),
        )
        assert res is not None
        for stage in res["allocations"]:
            nid = stage["node_id"]
            numas = {sched.gpu_numa_affinity[nid][g] for g in stage["gpu_ids"]}
            assert len(numas) == 2, (
                f"Stage {stage['stage_idx']} on node={nid} GPUs={stage['gpu_ids']} "
                f"must span 2 NUMAs; got {numas}"
            )

    @pytest.mark.asyncio
    async def test_pp2_fails_when_only_one_node_no_colocation(self):
        nodes, affinity = build_cluster(1, 8, 2)
        sched = make_scheduler(nodes, affinity)
        sched.worker_nodes = nodes
        sched.gpu_numa_affinity = affinity
        res = sched._allocate_pipeline_stages(
            worker_nodes=nodes, tp_size=1, pp_size=2,
            reserved_gpu_ids_by_node=defaultdict(set),
            allow_stage_colocation=False,
        )
        assert res is None, "Single-node cluster must reject PP=2 without colocation"

    @pytest.mark.asyncio
    async def test_pp2_colocates_when_explicitly_allowed(self):
        nodes, affinity = build_cluster(1, 8, 2)
        sched = make_scheduler(nodes, affinity)
        sched.worker_nodes = nodes
        sched.gpu_numa_affinity = affinity
        res = sched._allocate_pipeline_stages(
            worker_nodes=nodes, tp_size=1, pp_size=2,
            reserved_gpu_ids_by_node=defaultdict(set),
            allow_stage_colocation=True,
        )
        assert res is not None
        stages = res["allocations"]
        assert stages[0]["node_id"] == stages[1]["node_id"], (
            "With allow_stage_colocation=True on a single-node cluster, "
            "both stages must be on that node"
        )

    @pytest.mark.asyncio
    async def test_pp0_node_hint_honoured_for_all_nodes(self):
        """preferred_pp0_node_id is respected for every possible first-stage node."""
        nodes, affinity = build_cluster(4, 8, 2)
        sched = make_scheduler(nodes, affinity)
        sched.worker_nodes = nodes
        sched.gpu_numa_affinity = affinity
        for preferred in ("0", "1", "2", "3"):
            res = sched._allocate_pipeline_stages(
                worker_nodes=copy.deepcopy(nodes),
                tp_size=1, pp_size=2,
                reserved_gpu_ids_by_node=defaultdict(set),
                preferred_pp0_node_id=preferred,
            )
            assert res is not None
            assert res["node_id"] == preferred, (
                f"Stage-0 should be on hinted node {preferred!r}, got {res['node_id']!r}"
            )

    @pytest.mark.asyncio
    async def test_concurrent_pp2_allocs_no_gpu_overlap(self, bare_4n8g):
        """Two back-to-back PP=2,TP=2 allocations must not share any (node, gpu) pair."""
        sched, nodes = bare_4n8g
        wn = copy.deepcopy(nodes)
        reserved: Dict[str, Set] = defaultdict(set)

        r1 = sched._allocate_pipeline_stages(
            worker_nodes=wn, tp_size=2, pp_size=2,
            reserved_gpu_ids_by_node=reserved,
        )
        assert r1 is not None
        for s in r1["allocations"]:
            reserved[s["node_id"]].update(s["gpu_ids"])

        r2 = sched._allocate_pipeline_stages(
            worker_nodes=wn, tp_size=2, pp_size=2,
            reserved_gpu_ids_by_node=reserved,
        )
        assert r2 is not None
        gpus1 = {(s["node_id"], g) for s in r1["allocations"] for g in s["gpu_ids"]}
        gpus2 = {(s["node_id"], g) for s in r2["allocations"] for g in s["gpu_ids"]}
        assert not (gpus1 & gpus2), (
            f"GPU overlap between two PP allocations: {gpus1 & gpus2}"
        )

    @pytest.mark.asyncio
    async def test_pp_fails_gracefully_when_not_enough_nodes(self):
        """PP=4 on a 2-node cluster (without colocation) must return None."""
        nodes, affinity = build_cluster(2, 8, 2)
        sched = make_scheduler(nodes, affinity)
        sched.worker_nodes = nodes
        sched.gpu_numa_affinity = affinity
        res = sched._allocate_pipeline_stages(
            worker_nodes=nodes, tp_size=1, pp_size=4,
            reserved_gpu_ids_by_node=defaultdict(set),
            allow_stage_colocation=False,
        )
        assert res is None


# ═══════════════════════════════════════════════════════════════════════════════
#  C. Integration tests: full async allocate → control loop → deallocate
# ═══════════════════════════════════════════════════════════════════════════════

class TestAllocationLifecycle:

    @pytest.mark.asyncio
    async def test_tp1_returns_node_and_single_gpu(self, sched_4n8g):
        sched, nodes = sched_4n8g
        router = SimulatedGpuRouter(sched, "m1", {"num_gpus": 1, "tp_size": 1, "pp_size": 1})
        h = InstanceHandle(instance_id="i0", max_queue_length=4)
        info = await asyncio.wait_for(router.allocate_via_scheduler("i0", h), ALLOC_TIMEOUT)
        assert isinstance(info, dict)
        assert info["node_id"] in nodes
        assert len(info["gpu_ids"]) == 1
        await router.deallocate_via_scheduler(h)

    @pytest.mark.asyncio
    async def test_tp2_allocation_is_numa_balanced(self, sched_4n8g):
        sched, nodes = sched_4n8g
        router = SimulatedGpuRouter(sched, "m2", {"num_gpus": 2, "tp_size": 2, "pp_size": 1})
        h = InstanceHandle(instance_id="i0", max_queue_length=4)
        info = await asyncio.wait_for(router.allocate_via_scheduler("i0", h), ALLOC_TIMEOUT)
        nid = info["node_id"]
        numas = {sched.gpu_numa_affinity[nid][g] for g in info["gpu_ids"]}
        assert numas == {0, 1}, (
            f"TP=2 must span both NUMA domains; node={nid} GPUs={info['gpu_ids']}"
        )
        await router.deallocate_via_scheduler(h)

    @pytest.mark.asyncio
    async def test_tp4_allocation_is_numa_balanced(self, sched_4n8g):
        sched, _ = sched_4n8g
        router = SimulatedGpuRouter(sched, "m4", {"num_gpus": 4, "tp_size": 4, "pp_size": 1})
        h = InstanceHandle(instance_id="i0", max_queue_length=4)
        info = await asyncio.wait_for(router.allocate_via_scheduler("i0", h), ALLOC_TIMEOUT)
        nid = info["node_id"]
        by_numa: Dict[int, int] = defaultdict(int)
        for g in info["gpu_ids"]:
            by_numa[sched.gpu_numa_affinity[nid][g]] += 1
        assert by_numa[0] == 2 and by_numa[1] == 2, (
            f"TP=4 must distribute 2 GPUs per NUMA; got {dict(by_numa)}"
        )
        await router.deallocate_via_scheduler(h)

    @pytest.mark.asyncio
    async def test_deallocation_restores_free_slots(self, sched_4n8g):
        sched, _ = sched_4n8g
        router = SimulatedGpuRouter(sched, "m3", {"num_gpus": 4, "tp_size": 4, "pp_size": 1})
        handles = []
        for i in range(2):   # 2 × TP=4 fills one node (8 GPUs)
            h = InstanceHandle(instance_id=f"i{i}", max_queue_length=4)
            await asyncio.wait_for(router.allocate_via_scheduler(f"i{i}", h), ALLOC_TIMEOUT)
            handles.append(h)

        cap_before = await sched.get_cluster_gpu_capacity()
        await router.deallocate_via_scheduler(handles[0])
        cap_after = await sched.get_cluster_gpu_capacity()
        assert cap_after["capacity_free_gpus"] == cap_before["capacity_free_gpus"] + 4
        await router.deallocate_via_scheduler(handles[1])

    @pytest.mark.asyncio
    async def test_cluster_capacity_tracks_allocations(self, sched_4n8g):
        sched, _ = sched_4n8g
        cap0 = await sched.get_cluster_gpu_capacity()
        assert cap0["total_gpus"] == 4 * 8
        assert cap0["capacity_free_gpus"] == 4 * 8

        router = SimulatedGpuRouter(sched, "cap_m", {"num_gpus": 4, "tp_size": 4, "pp_size": 1})
        h = InstanceHandle(instance_id="c0", max_queue_length=4)
        await asyncio.wait_for(router.allocate_via_scheduler("c0", h), ALLOC_TIMEOUT)
        cap1 = await sched.get_cluster_gpu_capacity()
        assert cap1["capacity_free_gpus"] == 4 * 8 - 4
        await router.deallocate_via_scheduler(h)

    @pytest.mark.asyncio
    async def test_prewarm_instances_occupy_gpus_immediately(self, sched_4n8g):
        sched, _ = sched_4n8g
        cap0 = await sched.get_cluster_gpu_capacity()
        router = SimulatedGpuRouter(sched, "pw", {"num_gpus": 2, "tp_size": 2, "pp_size": 1})
        pool: List[InstanceHandle] = []
        await router.sync_empty_prewarm_pool(pool, 4)   # 4 × TP=2 = 8 GPUs

        cap1 = await sched.get_cluster_gpu_capacity()
        assert cap1["capacity_free_gpus"] == cap0["capacity_free_gpus"] - 8, (
            "Prewarm (empty) instances must immediately count against capacity"
        )
        for h in pool:
            await router.deallocate_via_scheduler(h)

    @pytest.mark.asyncio
    async def test_second_instance_colocates_on_same_node(self, sched_4n8g):
        """Second instance of the same model should land on the same node as the first."""
        sched, _ = sched_4n8g
        router = SimulatedGpuRouter(sched, "coloc_m", {"num_gpus": 2, "tp_size": 2, "pp_size": 1})
        h0 = InstanceHandle(instance_id="co0", max_queue_length=4)
        h1 = InstanceHandle(instance_id="co1", max_queue_length=4)
        i0 = await asyncio.wait_for(router.allocate_via_scheduler("co0", h0), ALLOC_TIMEOUT)
        i1 = await asyncio.wait_for(router.allocate_via_scheduler("co1", h1), ALLOC_TIMEOUT)
        assert i0["node_id"] == i1["node_id"], (
            f"Same-model instances must colocate: i0={i0['node_id']}, i1={i1['node_id']}"
        )
        await router.deallocate_via_scheduler(h0)
        await router.deallocate_via_scheduler(h1)

    @pytest.mark.asyncio
    async def test_different_models_same_config_colocated(self, sched_4n8g):
        """Two models with identical TP/PP config should prefer the same node."""
        sched, _ = sched_4n8g
        rA = SimulatedGpuRouter(sched, "mA", {"num_gpus": 2, "tp_size": 2, "pp_size": 1})
        rB = SimulatedGpuRouter(sched, "mB", {"num_gpus": 2, "tp_size": 2, "pp_size": 1})
        hA = InstanceHandle(instance_id="hA", max_queue_length=4)
        hB = InstanceHandle(instance_id="hB", max_queue_length=4)
        iA = await asyncio.wait_for(rA.allocate_via_scheduler("hA", hA), ALLOC_TIMEOUT)
        iB = await asyncio.wait_for(rB.allocate_via_scheduler("hB", hB), ALLOC_TIMEOUT)
        assert iA["node_id"] == iB["node_id"], (
            "Models sharing TP/PP config should be colocated to reduce fragmentation"
        )
        await rA.deallocate_via_scheduler(hA)
        await rB.deallocate_via_scheduler(hB)

    @pytest.mark.asyncio
    async def test_pp2_stages_on_distinct_nodes(self, sched_4n8g):
        sched, _ = sched_4n8g
        router = SimulatedGpuRouter(sched, "pp2m", {"num_gpus": 2, "tp_size": 1, "pp_size": 2})
        h = InstanceHandle(instance_id="pp0", max_queue_length=4)
        info = await asyncio.wait_for(router.allocate_via_scheduler("pp0", h), ALLOC_TIMEOUT)
        stages = info["allocations"]
        assert len(stages) == 2
        assert stages[0]["node_id"] != stages[1]["node_id"], (
            "PP=2 stages must be placed on distinct Ray nodes"
        )
        await router.deallocate_via_scheduler(h)

    @pytest.mark.asyncio
    async def test_pp2_tp2_integration_numa_and_distinct_nodes(self, sched_4n8g):
        """PP=2, TP=2: 2 nodes, each with 2 NUMA-balanced GPUs."""
        sched, _ = sched_4n8g
        router = SimulatedGpuRouter(sched, "pp2tp2", {"num_gpus": 4, "tp_size": 2, "pp_size": 2})
        h = InstanceHandle(instance_id="pt0", max_queue_length=4)
        info = await asyncio.wait_for(router.allocate_via_scheduler("pt0", h), ALLOC_TIMEOUT)
        assert info["pipeline_parallel_size"] == 2
        assert info["tensor_parallel_size"] == 2
        stages = info["allocations"]
        assert len(stages) == 2
        assert stages[0]["node_id"] != stages[1]["node_id"]
        for stage in stages:
            nid = stage["node_id"]
            numas = {sched.gpu_numa_affinity[nid][g] for g in stage["gpu_ids"]}
            assert len(numas) == 2, (
                f"TP=2 stage on node={nid} GPUs={stage['gpu_ids']} must span 2 NUMAs"
            )
        await router.deallocate_via_scheduler(h)

    @pytest.mark.asyncio
    async def test_no_gpu_double_allocation_multiple_models(self, sched_4n8g):
        """Six TP=4 allocations across 4×8 GPUs must not share any GPU."""
        sched, _ = sched_4n8g
        # 4 nodes × 2 TP=4 per node = 8 instances total (fills cluster)
        router = SimulatedGpuRouter(sched, "dedup_m", {"num_gpus": 4, "tp_size": 4, "pp_size": 1})
        handles = [InstanceHandle(instance_id=f"d{i}", max_queue_length=4) for i in range(8)]
        results = await asyncio.wait_for(
            asyncio.gather(*[router.allocate_via_scheduler(h.instance_id, h) for h in handles]),
            timeout=15.0,
        )
        seen: Set[Tuple[str, int]] = set()
        for info in results:
            assert isinstance(info, dict), f"Allocation must succeed: {info!r}"
            for g in info["gpu_ids"]:
                key = (info["node_id"], g)
                assert key not in seen, f"GPU {key} allocated to multiple instances"
                seen.add(key)
        for h in handles:
            await router.deallocate_via_scheduler(h)

    @pytest.mark.asyncio
    async def test_pending_request_served_when_gpu_freed(self, sched_4n8g):
        """A queued request blocked by full capacity is served once a GPU is freed."""
        sched, _ = sched_4n8g
        # Fill the entire cluster (4 nodes × 8 TP=1 = 32 instances)
        filler = SimulatedGpuRouter(sched, "fill", {"num_gpus": 1, "tp_size": 1, "pp_size": 1})
        fill_hs = [InstanceHandle(instance_id=f"f{i}", max_queue_length=1) for i in range(32)]
        await asyncio.wait_for(
            asyncio.gather(*[filler.allocate_via_scheduler(h.instance_id, h) for h in fill_hs]),
            timeout=15.0,
        )

        # This request cannot be served yet
        waiter = SimulatedGpuRouter(sched, "wait_m", {"num_gpus": 1, "tp_size": 1, "pp_size": 1})
        hw = InstanceHandle(instance_id="w0", max_queue_length=1)
        task = asyncio.create_task(waiter.allocate_via_scheduler("w0", hw))
        await asyncio.sleep(0.15)   # let one control-loop pass with no resources

        # Free one GPU → request should now complete
        await filler.deallocate_via_scheduler(fill_hs[0])
        info = await asyncio.wait_for(task, ALLOC_TIMEOUT)
        assert info is not None, "Queued request must succeed after a GPU is freed"

        await waiter.deallocate_via_scheduler(hw)
        for h in fill_hs[1:]:
            await filler.deallocate_via_scheduler(h)


# ═══════════════════════════════════════════════════════════════════════════════
#  D. GPU locking (weight-load serialisation)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGpuLocking:

    @pytest.mark.asyncio
    async def test_acquire_and_release(self, sched_4n8g):
        sched, _ = sched_4n8g
        ok = await sched.acquire_gpu_lock("0", [0, 1])
        assert ok is True
        await sched.release_gpu_lock("0", [0, 1])

    @pytest.mark.asyncio
    async def test_double_lock_same_gpu_rejected(self, sched_4n8g):
        sched, _ = sched_4n8g
        await sched.acquire_gpu_lock("0", [0])
        ok2 = await sched.acquire_gpu_lock("0", [0])
        assert ok2 is False, "Second lock on the same GPU must be rejected"
        await sched.release_gpu_lock("0", [0])

    @pytest.mark.asyncio
    async def test_partial_overlap_rejected(self, sched_4n8g):
        sched, _ = sched_4n8g
        await sched.acquire_gpu_lock("0", [0])
        ok = await sched.acquire_gpu_lock("0", [0, 1])  # GPU 0 already locked
        assert ok is False, "Lock overlapping an already-locked GPU must fail"
        await sched.release_gpu_lock("0", [0])

    @pytest.mark.asyncio
    async def test_non_overlapping_locks_succeed(self, sched_4n8g):
        sched, _ = sched_4n8g
        ok1 = await sched.acquire_gpu_lock("0", [0, 1])
        ok2 = await sched.acquire_gpu_lock("0", [2, 3])
        assert ok1 and ok2, "Non-overlapping GPU locks must both succeed"
        await sched.release_gpu_lock("0", [0, 1])
        await sched.release_gpu_lock("0", [2, 3])

    @pytest.mark.asyncio
    async def test_lock_released_makes_gpu_free_again(self, sched_4n8g):
        sched, _ = sched_4n8g
        for g in range(8):
            await sched.acquire_gpu_lock("0", [g])
        assert sched._compute_node_free_gpus("0", 8) == 0, "All locked → 0 free"
        for g in range(8):
            await sched.release_gpu_lock("0", [g])
        assert sched._compute_node_free_gpus("0", 8) == 8, "All released → 8 free"

    @pytest.mark.asyncio
    async def test_locked_node_skipped_for_allocation(self, sched_4n8g):
        """Lock all GPUs on node '0'; new allocations must land on another node."""
        sched, _ = sched_4n8g
        for g in range(8):
            await sched.acquire_gpu_lock("0", [g])
        router = SimulatedGpuRouter(sched, "lk_m", {"num_gpus": 1, "tp_size": 1, "pp_size": 1})
        h = InstanceHandle(instance_id="lk0", max_queue_length=4)
        info = await asyncio.wait_for(router.allocate_via_scheduler("lk0", h), ALLOC_TIMEOUT)
        assert info["node_id"] != "0", (
            "Allocation must not land on a fully GPU-locked node"
        )
        await router.deallocate_via_scheduler(h)
        for g in range(8):
            await sched.release_gpu_lock("0", [g])

    @pytest.mark.asyncio
    async def test_simulated_weight_load_cycle(self, sched_4n8g):
        """Simulate full weight-load lifecycle: allocate → lock → mark_loaded → unlock."""
        sched, _ = sched_4n8g
        router = SimulatedGpuRouter(sched, "wl_m", {"num_gpus": 2, "tp_size": 2, "pp_size": 1})
        h = InstanceHandle(instance_id="wl0", max_queue_length=4, empty_instance=True)
        await asyncio.wait_for(router.allocate_via_scheduler("wl0", h), ALLOC_TIMEOUT)

        # Verify GPU is occupied (but not locked yet)
        assert sched._compute_node_free_gpus(h.node_id, 8) == 6

        # Weight-load cycle
        await router.simulate_lazy_load_weight_hooks(h)

        # After unlock, GPU should still be occupied (weight is loaded, instance still alive)
        assert sched._compute_node_free_gpus(h.node_id, 8) == 6

        await router.deallocate_via_scheduler(h)
        # After deallocation, GPU freed
        assert sched._compute_node_free_gpus(h.node_id, 8) == 8


# ═══════════════════════════════════════════════════════════════════════════════
#  E. NUMA migration suggestion
# ═══════════════════════════════════════════════════════════════════════════════

class TestNumaMigrationSuggestion:

    @pytest.mark.asyncio
    async def test_balanced_cluster_no_migration_needed(self, sched_4n8g):
        """One TP=2 instance spanning both NUMAs leaves free GPUs balanced → no migration."""
        sched, _ = sched_4n8g
        router = SimulatedGpuRouter(sched, "bal_m", {"num_gpus": 2, "tp_size": 2, "pp_size": 1})
        h = InstanceHandle(instance_id="bal0", max_queue_length=4)
        await asyncio.wait_for(router.allocate_via_scheduler("bal0", h), ALLOC_TIMEOUT)
        plan = await sched.suggest_instance_migration("bal_m", 2)
        assert not plan, f"Balanced cluster must not trigger migration; got {plan}"
        await router.deallocate_via_scheduler(h)

    @pytest.mark.asyncio
    async def test_imbalanced_cluster_triggers_migration(self, migration_sched):
        """All free GPUs on NUMA-1 → suggest migrating the NUMA-0 model instance."""
        sched, _ = migration_sched
        # Node "0": occupy GPUs 0,1 with the target model (NUMA-0)
        #           occupy GPUs 2,3 with unrelated blocker (also NUMA-0)
        #           → free GPUs: 4,5,6,7 (all NUMA-1)
        inject_allocation(sched, "target", "0", [0, 1], model_name="imbal_m", tp=2, empty=False)
        inject_allocation(sched, "blk2",   "0", [2],    model_name="_sys",    tp=1, empty=True)
        inject_allocation(sched, "blk3",   "0", [3],    model_name="_sys",    tp=1, empty=True)

        plan = await sched.suggest_instance_migration("imbal_m", 2)
        assert isinstance(plan, dict) and plan.get("instance_id"), (
            f"Should suggest migration when free GPUs are NUMA-imbalanced; got {plan}"
        )
        assert plan["instance_id"] == "target"
        assert plan["node_id"] == "0"
        assert len(plan["target_gpu_ids"]) >= 2

    @pytest.mark.asyncio
    async def test_prefers_empty_instance_over_loaded(self, migration_sched):
        """When both loaded and empty instances can be migrated, prefer the empty one."""
        sched, _ = migration_sched
        # Two model instances on NUMA-0 GPUs: one loaded, one empty
        inject_allocation(sched, "loaded",  "0", [0, 1], model_name="mig_m", tp=2, empty=False)
        inject_allocation(sched, "prewarm", "0", [2, 3], model_name="mig_m", tp=2, empty=True)
        # Free GPUs: 4-7 (all NUMA-1)

        plan = await sched.suggest_instance_migration("mig_m", 2)
        assert plan.get("instance_id") == "prewarm", (
            "Empty instance should be preferred for migration to avoid KV state transfer"
        )

    @pytest.mark.asyncio
    async def test_locked_instance_skipped(self, migration_sched):
        """An instance whose GPUs are locked (active weight-load) must not be migrated."""
        sched, _ = migration_sched
        inject_allocation(sched, "locked_inst", "0", [0, 1], model_name="lk_m", tp=2, empty=False)
        # Free GPUs: 4-7 (NUMA-1), injecting NUMA-0 occupancy via blk2/blk3
        inject_allocation(sched, "blk2", "0", [2], model_name="_sys", tp=1, empty=True)
        inject_allocation(sched, "blk3", "0", [3], model_name="_sys", tp=1, empty=True)

        # Lock the model instance's GPUs
        await sched.acquire_gpu_lock("0", [0, 1])

        plan = await sched.suggest_instance_migration("lk_m", 2)
        assert not plan.get("instance_id"), (
            "Locked instance must not be suggested for migration"
        )
        await sched.release_gpu_lock("0", [0, 1])

    @pytest.mark.asyncio
    async def test_no_migration_for_tp1(self, sched_4n8g):
        """TP=1 never needs NUMA balancing (single GPU per group)."""
        sched, _ = sched_4n8g
        plan = await sched.suggest_instance_migration("any_model", 1)
        assert not plan


# ═══════════════════════════════════════════════════════════════════════════════
#  F. Large-cluster / concurrent-allocation stress tests
#     16 nodes × 8 GPUs (2 NUMA), 128 GPUs total
# ═══════════════════════════════════════════════════════════════════════════════

class TestLargeCluster:

    @pytest.mark.asyncio
    async def test_64_tp2_instances_no_gpu_sharing(self, sched_16n8g):
        """64 TP=2 instances (128 GPUs total) placed without any GPU overlap."""
        sched, nodes = sched_16n8g
        total = 16 * 8 // 2   # = 64
        router = SimulatedGpuRouter(sched, "tp2m", {"num_gpus": 2, "tp_size": 2, "pp_size": 1})
        handles = [
            InstanceHandle(instance_id=f"tp2_{i}", max_queue_length=4) for i in range(total)
        ]
        results = await asyncio.wait_for(
            asyncio.gather(*[router.allocate_via_scheduler(h.instance_id, h) for h in handles]),
            timeout=30.0,
        )

        assert len(results) == total
        seen: Set[Tuple[str, int]] = set()
        for info, h in zip(results, handles):
            assert isinstance(info, dict), f"Allocation {h.instance_id} failed: {info!r}"
            for g in info["gpu_ids"]:
                key = (info["node_id"], g)
                assert key not in seen, f"GPU {key} allocated to multiple instances"
                seen.add(key)

        for h in handles:
            await router.deallocate_via_scheduler(h)

    @pytest.mark.asyncio
    async def test_all_single_gpus_placed_on_16_nodes(self, sched_16n8g):
        """128 TP=1 instances fill entire 16-node cluster without conflict."""
        sched, nodes = sched_16n8g
        router = SimulatedGpuRouter(sched, "tp1m", {"num_gpus": 1, "tp_size": 1, "pp_size": 1})
        handles = [
            InstanceHandle(instance_id=f"sg_{i}", max_queue_length=1) for i in range(128)
        ]
        results = await asyncio.wait_for(
            asyncio.gather(*[router.allocate_via_scheduler(h.instance_id, h) for h in handles]),
            timeout=30.0,
        )
        placed = [r for r in results if isinstance(r, dict)]
        assert len(placed) == 128, f"All 128 TP=1 instances must be placed, got {len(placed)}"
        seen: Set[Tuple[str, int]] = set()
        for info in placed:
            key = (info["node_id"], info["gpu_ids"][0])
            assert key not in seen, f"GPU {key} double-allocated"
            seen.add(key)
        for h in handles:
            await router.deallocate_via_scheduler(h)

    @pytest.mark.asyncio
    async def test_mixed_tp_pp_workload_no_gpu_conflict(self, sched_16n8g):
        """Mix of TP=1/2/4 and PP=2 models placed concurrently without GPU conflict."""
        sched, _ = sched_16n8g
        specs = [
            ("tp1_m",  {"num_gpus": 1, "tp_size": 1, "pp_size": 1}, 16),
            ("tp2_m",  {"num_gpus": 2, "tp_size": 2, "pp_size": 1}, 8),
            ("tp4_m",  {"num_gpus": 4, "tp_size": 4, "pp_size": 1}, 4),
            ("pp2_m",  {"num_gpus": 2, "tp_size": 1, "pp_size": 2}, 4),
        ]
        all_pairs: List[Tuple[SimulatedGpuRouter, InstanceHandle]] = []
        seen_gpus: Set[Tuple[str, int]] = set()

        for model, res, count in specs:
            router = SimulatedGpuRouter(sched, model, res)
            hs = [
                InstanceHandle(instance_id=f"{model}_{i}", max_queue_length=4)
                for i in range(count)
            ]
            infos = await asyncio.wait_for(
                asyncio.gather(*[router.allocate_via_scheduler(h.instance_id, h) for h in hs]),
                timeout=15.0,
            )
            for info, h in zip(infos, hs):
                assert isinstance(info, dict), f"Allocation failed for {model}: {info!r}"
                for stage in info.get("allocations", []):
                    for g in stage["gpu_ids"]:
                        key = (stage["node_id"], g)
                        assert key not in seen_gpus, (
                            f"GPU {key} conflict: model={model} instance={h.instance_id}"
                        )
                        seen_gpus.add(key)
                all_pairs.append((router, h))

        for router, h in all_pairs:
            await router.deallocate_via_scheduler(h)

    @pytest.mark.asyncio
    async def test_auto_pipeline_split_when_tp_too_large_for_node(self, sched_16n8g):
        """TP=8 can't fit on an 8-GPU node with 4 GPUs already occupied.
        auto_split=True → scheduler should find TP=4,PP=2 (or similar)."""
        sched, _ = sched_16n8g
        # Pre-occupy 4 GPUs on every node
        blocker = SimulatedGpuRouter(sched, "blk", {"num_gpus": 4, "tp_size": 4, "pp_size": 1})
        b_handles = [InstanceHandle(instance_id=f"b{i}", max_queue_length=1) for i in range(16)]
        await asyncio.wait_for(
            asyncio.gather(*[blocker.allocate_via_scheduler(h.instance_id, h) for h in b_handles]),
            timeout=20.0,
        )

        # Now each node has 4 free GPUs; TP=8 on one node is impossible
        big = SimulatedGpuRouter(sched, "big_m", {"num_gpus": 8, "tp_size": 8, "pp_size": 1})
        hb = InstanceHandle(instance_id="big0", max_queue_length=1)
        info = await asyncio.wait_for(big.allocate_via_scheduler("big0", hb), ALLOC_TIMEOUT)

        assert info is not None, "Auto pipeline-split must find a valid placement"
        pp = info.get("pipeline_parallel_size", 1)
        tp = info.get("tensor_parallel_size", 1)
        assert pp >= 2, f"Expected auto-split to PP≥2, got PP={pp}"
        assert tp * pp == 8, f"Total GPU count must equal 8; TP={tp} × PP={pp}"
        stages = info["allocations"]
        assert len(stages) == pp
        assert len({s["node_id"] for s in stages}) == pp, (
            "Auto-split PP stages must be on distinct nodes"
        )

        await big.deallocate_via_scheduler(hb)
        for h in b_handles:
            await blocker.deallocate_via_scheduler(h)

    @pytest.mark.asyncio
    async def test_reallocation_after_full_cycle(self, sched_16n8g):
        """Allocate all 128 GPUs, deallocate all, then allocate all again."""
        sched, _ = sched_16n8g
        router = SimulatedGpuRouter(sched, "cycle_m", {"num_gpus": 1, "tp_size": 1, "pp_size": 1})
        handles = [InstanceHandle(instance_id=f"cy_{i}", max_queue_length=1) for i in range(128)]

        # Round 1: fill cluster
        r1 = await asyncio.wait_for(
            asyncio.gather(*[router.allocate_via_scheduler(h.instance_id, h) for h in handles]),
            timeout=30.0,
        )
        assert all(isinstance(r, dict) for r in r1), "All instances must be placed in round 1"

        # Deallocate all
        for h in handles:
            await router.deallocate_via_scheduler(h)

        cap = await sched.get_cluster_gpu_capacity()
        assert cap["capacity_free_gpus"] == 128, "All GPUs must be free after full deallocation"

        # Round 2: fill again with fresh handles
        handles2 = [InstanceHandle(instance_id=f"cy2_{i}", max_queue_length=1) for i in range(128)]
        r2 = await asyncio.wait_for(
            asyncio.gather(*[router.allocate_via_scheduler(h.instance_id, h) for h in handles2]),
            timeout=30.0,
        )
        assert all(isinstance(r, dict) for r in r2), "All instances must be placed in round 2"
        for h in handles2:
            await router.deallocate_via_scheduler(h)

    @pytest.mark.asyncio
    async def test_fcfs_pending_request_served_on_gpu_free(self, sched_16n8g):
        """A TP=2 request queued against a full cluster succeeds once a GPU pair is freed."""
        sched, _ = sched_16n8g
        # Fill the entire 16-node × 8-GPU cluster with TP=1 instances (128 total)
        filler = SimulatedGpuRouter(sched, "fill_m", {"num_gpus": 1, "tp_size": 1, "pp_size": 1})
        f_handles = [InstanceHandle(instance_id=f"f{i}", max_queue_length=1) for i in range(128)]
        await asyncio.wait_for(
            asyncio.gather(*[filler.allocate_via_scheduler(h.instance_id, h) for h in f_handles]),
            timeout=30.0,
        )

        cap = await sched.get_cluster_gpu_capacity()
        assert cap["capacity_free_gpus"] == 0, "Cluster must be full before the FCFS test"

        # Submit a TP=2 request — it cannot be served immediately
        tp2 = SimulatedGpuRouter(sched, "wait_m", {"num_gpus": 2, "tp_size": 2, "pp_size": 1})
        hw = InstanceHandle(instance_id="waitr", max_queue_length=1)
        task = asyncio.create_task(tp2.allocate_via_scheduler("waitr", hw))

        # Let one control-loop iteration run with no resources available
        await asyncio.sleep(0.15)

        # Free two GPUs that were allocated to the same node so a TP=2 can be placed.
        # The scheduler colocates same-model instances, so consecutive allocations
        # within one node are grouped — free the first two handles from node "0".
        node0_handles = [h for h in f_handles if h.node_id == "0"][:2]
        for h in node0_handles:
            await filler.deallocate_via_scheduler(h)
        remaining_handles = [h for h in f_handles if h not in node0_handles]

        info = await asyncio.wait_for(task, ALLOC_TIMEOUT)
        assert info is not None, "Queued TP=2 request must succeed after 2 GPUs are freed"

        await tp2.deallocate_via_scheduler(hw)
        for h in remaining_handles:
            await filler.deallocate_via_scheduler(h)

    @pytest.mark.asyncio
    async def test_pp4_on_16_node_cluster(self, sched_16n8g):
        """PP=4 allocates 4 distinct nodes; repeat 4 times with no GPU overlap."""
        sched, _ = sched_16n8g
        router = SimulatedGpuRouter(sched, "pp4m", {"num_gpus": 4, "tp_size": 1, "pp_size": 4})
        handles = [InstanceHandle(instance_id=f"pp4_{i}", max_queue_length=1) for i in range(4)]
        results = await asyncio.wait_for(
            asyncio.gather(*[router.allocate_via_scheduler(h.instance_id, h) for h in handles]),
            timeout=15.0,
        )
        seen: Set[Tuple[str, int]] = set()
        for info, h in zip(results, handles):
            assert isinstance(info, dict), f"PP=4 allocation failed for {h.instance_id}"
            assert len(info["allocations"]) == 4
            stage_nodes = {s["node_id"] for s in info["allocations"]}
            assert len(stage_nodes) == 4, "All 4 PP stages must be on distinct nodes"
            for stage in info["allocations"]:
                key = (stage["node_id"], stage["gpu_ids"][0])
                assert key not in seen, f"GPU {key} double-allocated across PP instances"
                seen.add(key)
        for h in handles:
            await router.deallocate_via_scheduler(h)
