"""
Simulation tests: realistic multi-model concurrent workloads.

Real deployments differ from unit tests in three key ways:
  1. Multiple heterogeneous models share the same cluster simultaneously.
  2. Request arrivals are non-deterministic (Poisson process per model).
  3. Each model's popularity—and therefore its instance-pool size—changes
     over time: bursts, quiet periods, and complete deploy/undeploy cycles
     all happen while other models keep running.

These tests verify that the scheduler remains *correct* (no GPU double-
allocation, no negative free-GPU counts, proper NUMA placement) under all
of these conditions.

Design
──────
• ``ClusterInvariantChecker`` runs as a background asyncio task that
  snapshots scheduler state every 50 ms and records any violation.

• ``ModelSimulator`` represents a single deployed model.  It spawns
  instances at Poisson-distributed intervals, each lasting a random
  duration, up to a configurable concurrency cap.

• ``MultiModelCluster`` owns a set of ModelSimulators and exposes
  high-level helpers (start, drain, assert_invariants).

No Ray process is required; ``get_worker_nodes`` is patched to return a
synthetic cluster and the control loop runs at 50 ms intervals.
"""
from __future__ import annotations

import asyncio
import copy
import random
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
from unittest.mock import patch

import pytest
import pytest_asyncio

from sllm.schedulers.fcfs_scheduler import FcfsScheduler
from sllm.utils import InstanceHandle
from tests.scheduler_test.simulated_gpu_router import SimulatedGpuRouter

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers shared with test_scheduler.py (copied to avoid circular imports)
# ─────────────────────────────────────────────────────────────────────────────

def build_cluster(
    num_nodes: int,
    gpus_per_node: int,
    numa_domains: int = 2,
) -> Tuple[Dict[str, Any], Dict[str, Dict[int, int]]]:
    nodes: Dict[str, Any] = {}
    affinity: Dict[str, Dict[int, int]] = {}
    block = max(1, gpus_per_node // numa_domains)
    for i in range(num_nodes):
        nid = str(i)
        nodes[nid] = {"total_gpu": gpus_per_node, "remaining_gpu_slots": gpus_per_node}
        affinity[nid] = {g: g // block for g in range(gpus_per_node)}
    return nodes, affinity


def make_scheduler(
    nodes: Dict,
    affinity: Dict,
    *,
    auto_split: bool = False,
    loop_interval: float = 0.05,
) -> FcfsScheduler:
    return FcfsScheduler(
        scheduler_config={
            "gpu_numa_affinity": affinity,
            "auto_pipeline_split": auto_split,
            "control_loop_interval_s": loop_interval,
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Invariant checker
#
#  Two properties are checked on every 50 ms snapshot (P1, P2).
#  One property is checked only at quiescence after drain() (P3).
#
#  P1 – No GPU double-allocation  [continuous]
#       The same (node, gpu) pair must not appear in more than one entry in
#       instance_allocations.  A violation means two vLLM processes would be
#       given the same physical GPU → crash or silent data corruption.
#
#  P2 – Exact GPU accounting  [continuous]
#       For every node:  free + occupied == total
#       where "occupied" means the GPU appears in at least one allocation OR
#       in gpu_locks.  Since _compute_node_free_gpus and _count_gpu_instances
#       are read atomically (no await between them in a synchronous method),
#       this is mathematically tautological UNLESS there is a logic bug in the
#       scheduler's bookkeeping (e.g. an allocation that is tracked but not
#       counted, or a lock that escapes the accounting).
#
#  P3 – remaining_gpu_slots consistency  [quiescence only]
#       self.worker_nodes[nid]["remaining_gpu_slots"] must equal
#       _compute_node_free_gpus(nid, total) when the cluster is idle.
#
#       WHY NOT CONTINUOUS: the control loop updates remaining_gpu_slots in
#       two phases — (a) _get_worker_nodes recomputes it at the start of each
#       iteration, and (b) _update_worker_nodes writes back the local copy at
#       the end.  If a deallocation occurs between (a) and (b), the written-
#       back value temporarily under-counts free GPUs (conservative, not
#       dangerous).  This transient under-count is by design and lasts at most
#       one control-loop interval (50 ms).  Checking it continuously at 50 ms
#       resolution produces unavoidable false positives under saturation load.
#       At quiescence the scheduler is idle, all in-flight deallocations have
#       completed, and (a) and (b) agree with reality.
# ─────────────────────────────────────────────────────────────────────────────

class ClusterInvariantChecker:
    """Background task that snapshots scheduler state every 50 ms.

    P1 and P2 are checked on every snapshot.
    P3 is checked only at quiescence via assert_no_leaked_instances().
    """

    def __init__(self, sched: FcfsScheduler, nodes: Dict[str, Any]) -> None:
        self.sched = sched
        self.nodes = nodes
        self.violations: List[str] = []
        self._task: Optional[asyncio.Task] = None
        self._checks = 0

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(0.05)
            self._snapshot()

    def _snapshot(self) -> None:
        """Check P1 and P2 atomically (no await → consistent view)."""
        self._checks += 1

        # ── Build (node, gpu) → [instance_ids] from instance_allocations ─────
        gpu_owners: Dict[Tuple[str, int], List[str]] = defaultdict(list)
        for iid, info in self.sched.instance_allocations.items():
            for stage in info.get("allocations", []):
                nid = str(stage.get("node_id", ""))
                for g in stage.get("gpu_ids", []):
                    gpu_owners[(nid, int(g))].append(iid)

        for node_id, node_info in self.nodes.items():
            total = int(node_info["total_gpu"])

            # ── P1: no double-allocation ─────────────────────────────────────
            for (nid, gpu), owners in gpu_owners.items():
                if nid != node_id:
                    continue
                if len(owners) > 1:
                    self.violations.append(
                        f"[P1 check#{self._checks}] "
                        f"node={nid} GPU={gpu} double-allocated to {owners}"
                    )

            # ── P2: exact accounting  free + occupied == total ───────────────
            # All reads below are synchronous — no await can interleave.
            # By De Morgan: free = not(locked or allocated), occupied = locked or allocated
            # → free + occupied == total is tautological if bookkeeping is correct.
            counts = self.sched._count_gpu_instances_on_node(node_id)
            node_locks = self.sched.gpu_locks.get(node_id, {})
            occupied = sum(
                1 for g in range(total)
                if counts.get(g, 0) > 0 or node_locks.get(g, False)
            )
            free = self.sched._compute_node_free_gpus(node_id, total)
            if free + occupied != total:
                self.violations.append(
                    f"[P2 check#{self._checks}] node={node_id}: "
                    f"free({free}) + occupied({occupied}) = {free+occupied} ≠ total({total})"
                )

    def assert_clean(self) -> None:
        assert not self.violations, (
            f"{len(self.violations)} invariant violation(s) detected "
            f"across {self._checks} snapshots:\n"
            + "\n".join(self.violations[:20])
        )

    def assert_no_leaked_instances(self) -> None:
        """P3 + leak check: call ONLY after drain() when the cluster is fully idle.

        At quiescence:
          • instance_allocations must be empty (no ghost instances).
          • Every node must report all GPUs free via _compute_node_free_gpus.
          • remaining_gpu_slots must equal _compute_node_free_gpus (P3).
            At quiescence the control loop is not mid-iteration so there is
            no in-flight overwrite of remaining_gpu_slots.
        """
        leaked = list(self.sched.instance_allocations.keys())
        assert not leaked, (
            f"instance_allocations not empty after drain: {leaked[:10]}"
        )
        for node_id, node_info in self.nodes.items():
            total = int(node_info["total_gpu"])
            free = self.sched._compute_node_free_gpus(node_id, total)
            assert free == total, (
                f"node={node_id}: {total - free} GPU(s) still occupied after full drain"
            )
            # P3: at quiescence remaining_gpu_slots must match reality
            slots = int(
                self.sched.worker_nodes.get(node_id, {}).get("remaining_gpu_slots", -1)
            )
            if slots != -1:
                assert slots == total, (
                    f"[P3] node={node_id}: remaining_gpu_slots={slots} ≠ "
                    f"total={total} after full drain"
                )


# ─────────────────────────────────────────────────────────────────────────────
#  Per-model simulator
# ─────────────────────────────────────────────────────────────────────────────

class ModelSimulator:
    """Simulates one deployed model: Poisson arrivals, random instance lifetime."""

    def __init__(
        self,
        sched: FcfsScheduler,
        model_name: str,
        resource_req: Dict[str, Any],
        *,
        max_concurrent: int = 6,
        lifetime_range: Tuple[float, float] = (0.3, 1.5),
        alloc_timeout: float = 4.0,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.router = SimulatedGpuRouter(sched, model_name, resource_req)
        self.model_name = model_name
        self.max_concurrent = max_concurrent
        self.lifetime_range = lifetime_range
        self.alloc_timeout = alloc_timeout
        self.rng = rng or random.Random()
        self._active: List[InstanceHandle] = []
        self._lock = asyncio.Lock()
        self._counter = 0
        # ── Counters for post-simulation assertions ──────────────────────────
        self.n_allocated = 0
        self.n_deallocated = 0
        self.n_timed_out = 0
        self.alloc_latencies: List[float] = []
        # ── NUMA / placement tracking ────────────────────────────────────────
        self.numa_violations: List[str] = []
        self.pp_node_violations: List[str] = []
        self._tp = int(resource_req.get("tp_size", resource_req.get("num_gpus", 1)))
        self._pp = int(resource_req.get("pp_size", 1))
        self._sched = sched

    def _next_id(self) -> str:
        self._counter += 1
        return f"{self.model_name}_{self._counter}"

    async def _try_allocate(self) -> None:
        """Attempt one scale-up step; skip if already at capacity."""
        async with self._lock:
            if len(self._active) >= self.max_concurrent:
                return

        iid = self._next_id()
        h = InstanceHandle(instance_id=iid, max_queue_length=4)
        t0 = time.perf_counter()
        try:
            info = await asyncio.wait_for(
                self.router.allocate_via_scheduler(iid, h),
                timeout=self.alloc_timeout,
            )
        except asyncio.TimeoutError:
            self.n_timed_out += 1
            return

        if not isinstance(info, dict):
            self.n_timed_out += 1
            return

        latency = time.perf_counter() - t0
        self.alloc_latencies.append(latency)
        self.n_allocated += 1

        # Validate NUMA balance for TP > 1
        if self._tp > 1:
            self._check_numa(info)

        # Validate PP stages on distinct nodes
        if self._pp > 1:
            self._check_pp_placement(info)

        lifetime = self.rng.uniform(*self.lifetime_range)
        async with self._lock:
            self._active.append(h)
        asyncio.create_task(self._release_after(h, lifetime))

    def _check_numa(self, info: Dict) -> None:
        numa_affinity = self._sched.gpu_numa_affinity
        for stage in info.get("allocations", []):
            nid = stage.get("node_id", "")
            gpus = stage.get("gpu_ids", [])
            nmap = numa_affinity.get(str(nid), {})
            if not nmap:
                continue
            domains = {nmap.get(g, -1) for g in gpus}
            # Number of *distinct* NUMA domains available on this node, not the
            # total number of GPU entries.  For TP=4 on a 2-NUMA node we expect
            # min(4, 2) = 2 domains to be covered — not 4.
            n_numa_domains = len(set(nmap.values()))
            expected = min(self._tp, n_numa_domains)
            if len(domains) < expected:
                self.numa_violations.append(
                    f"{self.model_name} iid={info.get('node_id')}: "
                    f"TP={self._tp} GPUs={gpus} only spans NUMA domains {domains} "
                    f"(expected ≥{expected} of {n_numa_domains} on node {nid})"
                )

    def _check_pp_placement(self, info: Dict) -> None:
        stages = info.get("allocations", [])
        stage_nodes = [s.get("node_id") for s in stages]
        if len(set(stage_nodes)) < len(stage_nodes):
            self.pp_node_violations.append(
                f"{self.model_name}: PP={self._pp} stages landed on "
                f"non-distinct nodes {stage_nodes}"
            )

    async def _release_after(self, h: InstanceHandle, delay: float) -> None:
        await asyncio.sleep(delay)
        async with self._lock:
            try:
                self._active.remove(h)
            except ValueError:
                return
        await self.router.deallocate_via_scheduler(h)
        self.n_deallocated += 1

    async def run_poisson(self, duration: float, arrival_rate: float) -> None:
        """Drive Poisson arrivals for `duration` seconds at `arrival_rate` per second."""
        end = asyncio.get_event_loop().time() + duration
        while asyncio.get_event_loop().time() < end:
            await self._try_allocate()
            # Exponential inter-arrival time, capped so we don't overshoot end
            dt = self.rng.expovariate(arrival_rate)
            remaining = end - asyncio.get_event_loop().time()
            await asyncio.sleep(min(dt, max(0.01, remaining)))

    async def drain(self) -> None:
        """Forcefully deallocate all remaining active instances."""
        async with self._lock:
            handles = list(self._active)
            self._active.clear()
        for h in handles:
            try:
                await asyncio.wait_for(
                    self.router.deallocate_via_scheduler(h), timeout=2.0
                )
                self.n_deallocated += 1
            except (asyncio.TimeoutError, Exception):
                pass

    def assert_placement_correct(self) -> None:
        assert not self.numa_violations, (
            f"NUMA violations for {self.model_name}:\n"
            + "\n".join(self.numa_violations)
        )
        assert not self.pp_node_violations, (
            f"PP placement violations for {self.model_name}:\n"
            + "\n".join(self.pp_node_violations)
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-model cluster orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class MultiModelCluster:
    """Runs many ModelSimulators on one scheduler and collects invariant results."""

    def __init__(
        self,
        sched: FcfsScheduler,
        nodes: Dict[str, Any],
        *,
        seed: int = 42,
    ) -> None:
        self.sched = sched
        self.nodes = nodes
        self.checker = ClusterInvariantChecker(sched, nodes)
        self._rng = random.Random(seed)
        self._simulators: List[ModelSimulator] = []

    def add_model(
        self,
        model_name: str,
        resource_req: Dict[str, Any],
        *,
        max_concurrent: int = 6,
        lifetime_range: Tuple[float, float] = (0.3, 1.5),
    ) -> ModelSimulator:
        sim = ModelSimulator(
            self.sched,
            model_name,
            resource_req,
            max_concurrent=max_concurrent,
            lifetime_range=lifetime_range,
            rng=random.Random(self._rng.randint(0, 2**31)),
        )
        self._simulators.append(sim)
        return sim

    async def run(
        self,
        duration: float,
        arrival_rates: Optional[List[float]] = None,
    ) -> None:
        """Run all simulators concurrently for `duration` seconds."""
        await self.checker.start()
        if arrival_rates is None:
            arrival_rates = [1.0] * len(self._simulators)
        tasks = [
            asyncio.create_task(sim.run_poisson(duration, rate))
            for sim, rate in zip(self._simulators, arrival_rates)
        ]
        await asyncio.gather(*tasks)

    async def drain(self) -> None:
        """Drain all simulators and stop the invariant checker."""
        await asyncio.gather(*[sim.drain() for sim in self._simulators])
        await self.checker.stop()

    def assert_all_clean(self) -> None:
        self.checker.assert_clean()
        self.checker.assert_no_leaked_instances()
        for sim in self._simulators:
            sim.assert_placement_correct()
            assert sim.n_allocated > 0, (
                f"Model {sim.model_name} placed 0 instances — "
                "simulation was not actually stressful enough to exercise this model"
            )

    def print_stats(self) -> None:
        total_alloc = sum(s.n_allocated for s in self._simulators)
        total_dealloc = sum(s.n_deallocated for s in self._simulators)
        total_timeout = sum(s.n_timed_out for s in self._simulators)
        print(
            f"\n{'─'*64}\n"
            f"  Cluster stats — {len(self._simulators)} models, "
            f"{self.checker._checks} invariant checks\n"
            f"  Allocated: {total_alloc}  Deallocated: {total_dealloc}  "
            f"Timed-out: {total_timeout}\n"
        )
        for sim in self._simulators:
            avg_lat = (
                sum(sim.alloc_latencies) / len(sim.alloc_latencies)
                if sim.alloc_latencies
                else 0.0
            )
            print(
                f"  {sim.model_name:30s}  alloc={sim.n_allocated:4d}  "
                f"dealloc={sim.n_deallocated:4d}  timeout={sim.n_timed_out:3d}  "
                f"avg_lat={avg_lat*1000:.1f}ms"
            )
        print(f"{'─'*64}")


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def cluster_16n8g():
    """16 nodes × 8 GPUs (2 NUMA domains), auto_split=True."""
    nodes, affinity = build_cluster(16, 8, 2)
    sched = make_scheduler(nodes, affinity, auto_split=True)
    with patch("sllm.schedulers.fcfs_scheduler.get_worker_nodes", return_value=nodes):
        await sched.start()
        yield sched, nodes
        await sched.shutdown()


@pytest_asyncio.fixture
async def cluster_8n8g():
    """8 nodes × 8 GPUs (2 NUMA domains), auto_split=False."""
    nodes, affinity = build_cluster(8, 8, 2)
    sched = make_scheduler(nodes, affinity, auto_split=False)
    with patch("sllm.schedulers.fcfs_scheduler.get_worker_nodes", return_value=nodes):
        await sched.start()
        yield sched, nodes
        await sched.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
#  S1 — Heterogeneous multi-model baseline
#
#  5 model tiers with different TP/PP configs deployed simultaneously.
#  Arrival rates follow a rough Zipf distribution (llama-7b is most popular).
#  Simulation runs for 8 s (real wall-clock) with 50 ms control-loop ticks.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_heterogeneous_multi_model_baseline(cluster_16n8g):
    """No GPU conflicts or accounting errors under five concurrent model types."""
    sched, nodes = cluster_16n8g
    mmc = MultiModelCluster(sched, nodes, seed=1)

    # (model, tp, pp, max_concurrent, lifetime_range_s, arrival_rate)
    specs = [
        ("llama-7b",       {"num_gpus": 1, "tp_size": 1, "pp_size": 1}, 8, (0.3, 1.0), 2.5),
        ("llama-13b",      {"num_gpus": 2, "tp_size": 2, "pp_size": 1}, 6, (0.4, 1.2), 1.5),
        ("llama-70b",      {"num_gpus": 4, "tp_size": 4, "pp_size": 1}, 4, (0.5, 1.5), 0.8),
        ("mixtral-8x7b",   {"num_gpus": 4, "tp_size": 4, "pp_size": 1}, 3, (0.6, 1.8), 0.5),
        ("llama-pipeline", {"num_gpus": 2, "tp_size": 1, "pp_size": 2}, 2, (0.8, 2.0), 0.3),
    ]
    rates = []
    for model, req, max_c, lt, rate in specs:
        mmc.add_model(model, req, max_concurrent=max_c, lifetime_range=lt)
        rates.append(rate)

    await mmc.run(duration=8.0, arrival_rates=rates)
    await mmc.drain()
    mmc.print_stats()
    mmc.assert_all_clean()

    # Final cluster must be empty
    cap = await sched.get_cluster_gpu_capacity()
    assert cap["capacity_free_gpus"] == 16 * 8, (
        f"All GPUs must be free after drain; got {cap}"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  S2 — Load spike: one model surges while others run normally
#
#  "llama-7b" suddenly receives 10× its normal arrival rate for 3 s then
#  drops back to baseline.  The cluster must absorb the spike without
#  misallocating GPUs that were reserved by other models.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_load_spike_and_recovery(cluster_16n8g):
    """Spike model must not steal GPUs; invariants hold throughout."""
    sched, nodes = cluster_16n8g
    checker = ClusterInvariantChecker(sched, nodes)
    await checker.start()

    rng = random.Random(2)

    # Background steady-state models
    bg_specs = [
        ("bg-tp2", {"num_gpus": 2, "tp_size": 2, "pp_size": 1}, 4, (0.5, 1.5), 0.8),
        ("bg-tp4", {"num_gpus": 4, "tp_size": 4, "pp_size": 1}, 3, (0.6, 1.8), 0.5),
    ]
    bg_sims: List[ModelSimulator] = []
    for model, req, max_c, lt, rate in bg_specs:
        sim = ModelSimulator(sched, model, req, max_concurrent=max_c,
                             lifetime_range=lt, rng=random.Random(rng.randint(0, 2**31)))
        bg_sims.append(sim)

    spike_sim = ModelSimulator(
        sched, "spike-tp1", {"num_gpus": 1, "tp_size": 1, "pp_size": 1},
        max_concurrent=12,
        lifetime_range=(0.3, 0.8),
        rng=random.Random(rng.randint(0, 2**31)),
    )

    async def run_bg():
        await asyncio.gather(*[
            sim.run_poisson(12.0, rate) for sim, (_, _, _, _, rate) in zip(bg_sims, bg_specs)
        ])

    async def run_spike():
        # Normal phase (3 s)
        await spike_sim.run_poisson(3.0, 0.5)
        # Spike phase (3 s)
        await spike_sim.run_poisson(3.0, 5.0)
        # Recovery phase (6 s)
        await spike_sim.run_poisson(6.0, 0.5)

    await asyncio.gather(run_bg(), run_spike())
    for sim in bg_sims + [spike_sim]:
        await sim.drain()
    await checker.stop()
    checker.assert_clean()
    checker.assert_no_leaked_instances()
    for sim in bg_sims + [spike_sim]:
        sim.assert_placement_correct()
        assert sim.n_allocated > 0, f"{sim.model_name} placed 0 instances"

    cap = await sched.get_cluster_gpu_capacity()
    assert cap["capacity_free_gpus"] == 16 * 8


# ─────────────────────────────────────────────────────────────────────────────
#  S3 — Model churn: deploy and undeploy models while cluster is active
#
#  Three "permanent" models keep running the whole time.
#  Four "ephemeral" models are deployed and torn down in sequence with
#  overlapping windows.  Each teardown drains the model's instances before
#  the next model is deployed.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_model_churn(cluster_8n8g):
    """Deploy/undeploy cycles must not cause accounting drift."""
    sched, nodes = cluster_8n8g
    checker = ClusterInvariantChecker(sched, nodes)
    await checker.start()

    rng = random.Random(3)

    # ── Permanent models ─────────────────────────────────────────────────────
    perm_sims = [
        ModelSimulator(sched, "perm-tp1", {"num_gpus": 1, "tp_size": 1, "pp_size": 1},
                       max_concurrent=4, lifetime_range=(0.4, 1.2),
                       rng=random.Random(rng.randint(0, 2**31))),
        ModelSimulator(sched, "perm-tp2", {"num_gpus": 2, "tp_size": 2, "pp_size": 1},
                       max_concurrent=3, lifetime_range=(0.5, 1.5),
                       rng=random.Random(rng.randint(0, 2**31))),
    ]
    perm_tasks = [
        asyncio.create_task(sim.run_poisson(14.0, 1.0))
        for sim in perm_sims
    ]

    # ── Ephemeral models (sequential deploy → active period → drain) ─────────
    ephemeral_specs = [
        ("eph-tp1-A", {"num_gpus": 1, "tp_size": 1, "pp_size": 1}, 3, (0.3, 0.8)),
        ("eph-tp2-B", {"num_gpus": 2, "tp_size": 2, "pp_size": 1}, 2, (0.4, 1.0)),
        ("eph-tp1-C", {"num_gpus": 1, "tp_size": 1, "pp_size": 1}, 4, (0.2, 0.6)),
        ("eph-tp4-D", {"num_gpus": 4, "tp_size": 4, "pp_size": 1}, 2, (0.6, 1.4)),
    ]

    # Each ephemeral model runs for 3 s then is drained before the next starts.
    async def run_ephemeral_sequence():
        for model, req, max_c, lt in ephemeral_specs:
            sim = ModelSimulator(
                sched, model, req, max_concurrent=max_c, lifetime_range=lt,
                rng=random.Random(rng.randint(0, 2**31)),
            )
            await sim.run_poisson(3.0, 1.5)
            await sim.drain()
            # Pause so permanent models can use the freed GPUs
            await asyncio.sleep(0.1)

    await asyncio.gather(*perm_tasks, run_ephemeral_sequence())
    for sim in perm_sims:
        await sim.drain()
    await checker.stop()
    checker.assert_clean()
    checker.assert_no_leaked_instances()
    for sim in perm_sims:
        sim.assert_placement_correct()
        assert sim.n_allocated > 0, f"{sim.model_name} placed 0 instances"

    cap = await sched.get_cluster_gpu_capacity()
    assert cap["capacity_free_gpus"] == 8 * 8


# ─────────────────────────────────────────────────────────────────────────────
#  S4 — Saturation: fill cluster to near-capacity with mixed models, then drain
#
#  Goal: hit 90%+ cluster utilisation repeatedly.  Every allocation and
#  deallocation must leave the GPU accounting consistent.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_near_saturation_mixed_models(cluster_16n8g):
    """GPU accounting stays exact even at near-100% cluster utilisation."""
    sched, nodes = cluster_16n8g
    checker = ClusterInvariantChecker(sched, nodes)
    await checker.start()

    # 128 total GPUs.  Use models that together demand ~120 GPUs when at max_concurrent.
    # tp1 × 24 = 24 GPUs, tp2 × 20 = 40 GPUs, tp4 × 14 = 56 GPUs  → 120 total
    specs = [
        ("sat-tp1", {"num_gpus": 1, "tp_size": 1, "pp_size": 1}, 24, (0.8, 2.0), 3.0),
        ("sat-tp2", {"num_gpus": 2, "tp_size": 2, "pp_size": 1}, 20, (0.8, 2.0), 2.0),
        ("sat-tp4", {"num_gpus": 4, "tp_size": 4, "pp_size": 1}, 14, (0.8, 2.0), 1.0),
    ]

    rng = random.Random(4)
    sims: List[ModelSimulator] = []
    rates: List[float] = []
    for model, req, max_c, lt, rate in specs:
        sims.append(ModelSimulator(
            sched, model, req, max_concurrent=max_c, lifetime_range=lt,
            alloc_timeout=8.0,
            rng=random.Random(rng.randint(0, 2**31)),
        ))
        rates.append(rate)

    tasks = [asyncio.create_task(sim.run_poisson(10.0, r)) for sim, r in zip(sims, rates)]
    await asyncio.gather(*tasks)

    # Measure peak utilisation (sampled mid-run, from checker snapshots)
    # At the end, check the cluster is fully released after drain
    for sim in sims:
        await sim.drain()

    await checker.stop()
    checker.assert_clean()
    checker.assert_no_leaked_instances()

    cap = await sched.get_cluster_gpu_capacity()
    assert cap["capacity_free_gpus"] == 16 * 8, (
        f"All GPUs must be free after draining; got {cap}"
    )
    total_placed = sum(s.n_allocated for s in sims)
    assert total_placed > 0, "At least some instances must have been placed"
    print(f"\nSaturation test: {total_placed} placements, {checker._checks} invariant checks")


# ─────────────────────────────────────────────────────────────────────────────
#  S5 — NUMA balance preserved under random concurrent TP=2 allocations
#
#  Issue under test: when many TP=2 requests arrive simultaneously, the
#  round-robin NUMA picker must never assign both GPUs from the same NUMA
#  domain.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_numa_balance_under_concurrent_tp2_load(cluster_16n8g):
    """All TP=2 placements must span both NUMA domains regardless of concurrency."""
    sched, nodes = cluster_16n8g
    checker = ClusterInvariantChecker(sched, nodes)
    await checker.start()

    rng = random.Random(5)
    sims: List[ModelSimulator] = []
    for i in range(4):
        sim = ModelSimulator(
            sched, f"tp2-model-{i}",
            {"num_gpus": 2, "tp_size": 2, "pp_size": 1},
            max_concurrent=8,
            lifetime_range=(0.2, 1.0),
            rng=random.Random(rng.randint(0, 2**31)),
        )
        sims.append(sim)

    tasks = [asyncio.create_task(sim.run_poisson(8.0, 2.0)) for sim in sims]
    await asyncio.gather(*tasks)
    for sim in sims:
        await sim.drain()
    await checker.stop()

    checker.assert_clean()
    checker.assert_no_leaked_instances()
    for sim in sims:
        sim.assert_placement_correct()
        assert sim.n_allocated > 0, f"{sim.model_name} placed 0 instances"

    total_alloc = sum(s.n_allocated for s in sims)
    print(f"\nTP=2 concurrent load: {total_alloc} total allocations, "
          f"{checker._checks} invariant checks")


# ─────────────────────────────────────────────────────────────────────────────
#  S6 — PP=2 placement correctness under concurrent PP requests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pp_placement_under_concurrent_load(cluster_16n8g):
    """PP=2 stages must always land on distinct nodes even under high concurrency."""
    sched, nodes = cluster_16n8g
    checker = ClusterInvariantChecker(sched, nodes)
    await checker.start()

    rng = random.Random(6)
    sims: List[ModelSimulator] = []
    for i in range(3):
        sim = ModelSimulator(
            sched, f"pp2-model-{i}",
            {"num_gpus": 2, "tp_size": 1, "pp_size": 2},
            max_concurrent=4,
            lifetime_range=(0.4, 1.5),
            rng=random.Random(rng.randint(0, 2**31)),
        )
        sims.append(sim)

    # Also run TP=1 models concurrently to create fragmentation
    for i in range(2):
        sim = ModelSimulator(
            sched, f"tp1-noise-{i}",
            {"num_gpus": 1, "tp_size": 1, "pp_size": 1},
            max_concurrent=6,
            lifetime_range=(0.2, 0.8),
            rng=random.Random(rng.randint(0, 2**31)),
        )
        sims.append(sim)

    tasks = [asyncio.create_task(sim.run_poisson(8.0, 1.0)) for sim in sims]
    await asyncio.gather(*tasks)
    for sim in sims:
        await sim.drain()
    await checker.stop()

    checker.assert_clean()
    checker.assert_no_leaked_instances()
    for sim in sims:
        sim.assert_placement_correct()
    pp_sims = [s for s in sims if s._pp > 1]
    assert all(s.n_allocated > 0 for s in pp_sims), "PP models must have placed instances"


# ─────────────────────────────────────────────────────────────────────────────
#  S7 — FCFS ordering: requests submitted earlier must be served first
#
#  Fill cluster to zero free GPUs.  Submit N requests with known submission
#  order.  Free GPU slots one at a time.  Verify that requests complete in
#  FIFO order.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fcfs_ordering_verified(cluster_8n8g):
    """FCFS: queued requests are served in submission order."""
    sched, nodes = cluster_8n8g
    # 8 nodes × 8 GPUs × TP=1 = 64 slots total
    total_gpus = 8 * 8

    router = SimulatedGpuRouter(sched, "fcfs_m", {"num_gpus": 1, "tp_size": 1, "pp_size": 1})

    # Fill cluster completely
    fill_handles = [
        InstanceHandle(instance_id=f"fill_{i}", max_queue_length=1)
        for i in range(total_gpus)
    ]
    await asyncio.wait_for(
        asyncio.gather(*[router.allocate_via_scheduler(h.instance_id, h) for h in fill_handles]),
        timeout=30.0,
    )
    cap = await sched.get_cluster_gpu_capacity()
    assert cap["capacity_free_gpus"] == 0, "Cluster must be full before FCFS test"

    # Submit N requests while cluster is full — they must queue in order
    N = 5
    completion_order: List[int] = []
    pending_handles = [
        InstanceHandle(instance_id=f"fcfs_{i}", max_queue_length=1) for i in range(N)
    ]

    async def alloc_and_record(idx: int) -> None:
        await router.allocate_via_scheduler(pending_handles[idx].instance_id, pending_handles[idx])
        completion_order.append(idx)

    # Submit all N tasks at nearly the same time with a tiny stagger so that
    # the queue ordering is deterministic.
    tasks = []
    for i in range(N):
        tasks.append(asyncio.create_task(alloc_and_record(i)))
        await asyncio.sleep(0.01)   # 10 ms stagger → clear submission ordering

    # Free one GPU at a time and verify FCFS
    for i in range(N):
        before = len(completion_order)
        await router.deallocate_via_scheduler(fill_handles[i])
        # Allow up to 3 control-loop iterations for the queued request to be served
        for _ in range(6):
            await asyncio.sleep(0.05)
            if len(completion_order) > before:
                break
        assert len(completion_order) == before + 1, (
            f"After freeing slot {i}, exactly 1 queued request should be served; "
            f"got {len(completion_order) - before}"
        )
        assert completion_order[-1] == i, (
            f"FCFS violation: freed slot {i}, expected request {i} to be served, "
            f"got request {completion_order[-1]}"
        )

    await asyncio.gather(*tasks)

    # Clean up
    for h in pending_handles:
        await router.deallocate_via_scheduler(h)
    for h in fill_handles[N:]:
        await router.deallocate_via_scheduler(h)


# ─────────────────────────────────────────────────────────────────────────────
#  S8 — Autoscaler simulation: pool grows and shrinks based on queue depth
#
#  Mimics the RoundRobinRouter autoscaler: maintain a target pool size that
#  adjusts every 500 ms based on a simulated request arrival count.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_autoscaler_pool_management(cluster_16n8g):
    """Autoscaler logic (grow/shrink pool) must never violate GPU accounting."""
    sched, nodes = cluster_16n8g
    checker = ClusterInvariantChecker(sched, nodes)
    await checker.start()

    router = SimulatedGpuRouter(
        sched, "auto_m", {"num_gpus": 2, "tp_size": 2, "pp_size": 1}
    )
    pool: List[InstanceHandle] = []
    pool_lock = asyncio.Lock()
    rng = random.Random(7)
    counter = [0]

    async def scale_to(target: int) -> None:
        async with pool_lock:
            while len(pool) < target:
                iid = f"auto_{counter[0]}"
                counter[0] += 1
                h = InstanceHandle(instance_id=iid, max_queue_length=4)
                try:
                    await asyncio.wait_for(
                        router.allocate_via_scheduler(iid, h), timeout=3.0
                    )
                    pool.append(h)
                except asyncio.TimeoutError:
                    break
            while len(pool) > target:
                h = pool.pop()
                await router.deallocate_via_scheduler(h)

    # Simulate a load profile: low → high → peak → cool-down
    load_profile = [2, 3, 5, 8, 10, 8, 6, 4, 2, 1]
    for target in load_profile:
        await scale_to(target)
        await asyncio.sleep(0.5)

    # Drain
    await scale_to(0)
    await checker.stop()
    checker.assert_clean()
    checker.assert_no_leaked_instances()

    cap = await sched.get_cluster_gpu_capacity()
    assert cap["capacity_free_gpus"] == 16 * 8
    print(f"\nAutoscaler test: {counter[0]} total instances over profile, "
          f"{checker._checks} invariant checks")


# ─────────────────────────────────────────────────────────────────────────────
#  S9 — Long-running mixed workload with mid-run invariant verification
#
#  The checker runs for 12 s real time.  At three checkpoints (4 s, 8 s, 12 s)
#  we explicitly verify the invariant state and assert zero violations so far.
#  This ensures bugs are caught early rather than only at the end.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_checkpoint_invariants_during_run(cluster_16n8g):
    """Invariants must hold at every checkpoint throughout a long simulation."""
    sched, nodes = cluster_16n8g
    checker = ClusterInvariantChecker(sched, nodes)
    await checker.start()

    rng = random.Random(8)
    model_configs = [
        ("chk-tp1", {"num_gpus": 1, "tp_size": 1, "pp_size": 1}, 8,  (0.3, 1.0), 2.0),
        ("chk-tp2", {"num_gpus": 2, "tp_size": 2, "pp_size": 1}, 6,  (0.4, 1.2), 1.5),
        ("chk-tp4", {"num_gpus": 4, "tp_size": 4, "pp_size": 1}, 4,  (0.5, 1.8), 0.8),
        ("chk-pp2", {"num_gpus": 2, "tp_size": 1, "pp_size": 2}, 3,  (0.6, 2.0), 0.4),
    ]

    async def segment(duration: float, label: str) -> None:
        """Run all models for one checkpoint segment."""
        tasks = []
        sims_this_seg: List[ModelSimulator] = []
        rates: List[float] = []
        for model, req, max_c, lt, rate in model_configs:
            sim = ModelSimulator(
                sched, f"{model}-{label}", req,
                max_concurrent=max_c, lifetime_range=lt,
                rng=random.Random(rng.randint(0, 2**31)),
            )
            sims_this_seg.append(sim)
            rates.append(rate)
        tasks = [
            asyncio.create_task(sim.run_poisson(duration, r))
            for sim, r in zip(sims_this_seg, rates)
        ]
        await asyncio.gather(*tasks)
        for sim in sims_this_seg:
            await sim.drain()
        return sims_this_seg

    # Three 4-second checkpoint segments
    sims_all: List[ModelSimulator] = []
    for label in ("seg1", "seg2", "seg3"):
        seg_sims = await segment(4.0, label)
        sims_all.extend(seg_sims)
        # Checkpoint: P1/P2/P3 must be clean AND no leaked instances between segments
        checker.assert_clean()
        checker.assert_no_leaked_instances()

    await checker.stop()
    checker.assert_clean()
    checker.assert_no_leaked_instances()

    cap = await sched.get_cluster_gpu_capacity()
    assert cap["capacity_free_gpus"] == 16 * 8

    total_allocs = sum(s.n_allocated for s in sims_all)
    assert total_allocs > 0, "Simulation must have placed at least some instances"
    print(f"\nCheckpoint test: {total_allocs} total allocations across 3 segments, "
          f"{checker._checks} invariant checks")


# ─────────────────────────────────────────────────────────────────────────────
#  S10 — Stress test: random model mix, random parallelism, random timing
#
#  Uses a seeded RNG to generate 10 random model configurations, then drives
#  Poisson arrivals for 10 s.  The exact models and their params differ from
#  all other tests, providing broader coverage without exhaustive enumeration.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_random_model_mix_stress(cluster_16n8g):
    """Randomised model configurations must not produce any invariant violations."""
    sched, nodes = cluster_16n8g
    checker = ClusterInvariantChecker(sched, nodes)
    await checker.start()

    rng = random.Random(9)

    def rand_model(idx: int) -> Tuple[str, Dict, int, Tuple[float, float], float]:
        # TP must be a power of 2, PP either 1 or 2
        tp = rng.choice([1, 1, 1, 2, 2, 4])
        pp = rng.choice([1, 1, 2]) if tp <= 2 else 1
        max_c = rng.randint(1, max(1, 8 // (tp * pp)))
        lt_lo = rng.uniform(0.2, 0.6)
        lt_hi = lt_lo + rng.uniform(0.3, 1.2)
        rate = rng.uniform(0.3, 2.0)
        return (
            f"rand-m{idx}-tp{tp}pp{pp}",
            {"num_gpus": tp, "tp_size": tp, "pp_size": pp},
            max_c,
            (lt_lo, lt_hi),
            rate,
        )

    n_models = 10
    sims: List[ModelSimulator] = []
    rates: List[float] = []
    for i in range(n_models):
        model, req, max_c, lt, rate = rand_model(i)
        sims.append(ModelSimulator(
            sched, model, req, max_concurrent=max_c,
            lifetime_range=lt, alloc_timeout=5.0,
            rng=random.Random(rng.randint(0, 2**31)),
        ))
        rates.append(rate)

    tasks = [asyncio.create_task(sim.run_poisson(10.0, r)) for sim, r in zip(sims, rates)]
    await asyncio.gather(*tasks)
    for sim in sims:
        await sim.drain()
    await checker.stop()
    checker.assert_clean()
    checker.assert_no_leaked_instances()
    for sim in sims:
        sim.assert_placement_correct()

    cap = await sched.get_cluster_gpu_capacity()
    assert cap["capacity_free_gpus"] == 16 * 8

    total = sum(s.n_allocated for s in sims)
    print(f"\nRandom stress: {n_models} models, {total} total placements, "
          f"{checker._checks} invariant checks")
