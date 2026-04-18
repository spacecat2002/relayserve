# ---------------------------------------------------------------------------- #
#  Broad automated coverage for sllm (unit + light integration, mocked I/O).   #
#                                                                              #
#  vLLM is **not** required: stub ``vllm`` / ``vllm.inputs`` before any        #
#  ``sllm.backends.*`` import. If those modules were already loaded (full      #
#  suite), they are reloaded so bindings use the stub.                         #
#                                                                              #
#  This file is **not** a substitute for:                                    #
#  - Real Ray cluster + GPU e2e (see tests/router_test, inference_test)        #
#  - Real vLLM engines / weight loading                                      #
#                                                                              #
#  Run:  pytest tests/test_comprehensive_sllm_suite.py -v                      #
# ---------------------------------------------------------------------------- #
from __future__ import annotations

import importlib
import sys
import types
from dataclasses import field, make_dataclass
from typing import Any, Dict, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_MARK = "_SLLM_VLLM_TEST_STUB"

# Names ``build_core_filtered_engine_config`` / ``GPUBackend`` may pass to
# ``AsyncEngineArgs(**filtered)`` (must be a real dataclass for ``fields()``).
_ASYNC_ENGINE_ARG_NAMES = (
    "model",
    "tokenizer",
    "tokenizer_mode",
    "trust_remote_code",
    "dtype",
    "load_format",
    "enforce_eager",
    "enable_prefix_caching",
    "task",
    "load_method",
    "max_model_len",
    "max_seq_len",
    "pipeline_parallel_size",
    "tensor_parallel_size",
    "worker_use_ray",
    "seed",
    "shm_tp_size",
    "shm_kv_cache_size",
    "shm_num_blocks",
    "shm_block_len",
    "kv_transfer_config",
)


def _install_vllm_stub() -> Tuple[types.ModuleType, types.ModuleType]:
    """Minimal vLLM surface so ``backend_utils`` / ``gpu_backend`` import."""
    vllm = types.ModuleType("vllm")
    setattr(vllm, _MARK, True)

    AsyncEngineArgs = make_dataclass(
        "AsyncEngineArgs",
        [(n, Any, field(default=None)) for n in _ASYNC_ENGINE_ARG_NAMES],
    )

    class AsyncLLMEngine:
        @staticmethod
        def from_engine_args(*_a: Any, **_k: Any) -> Any:
            raise RuntimeError("stub AsyncLLMEngine.from_engine_args (patch in test)")

    class RequestOutput:
        pass

    class SamplingParams:
        def __init__(self, **kw: Any) -> None:
            self.kw = kw

    vllm.AsyncEngineArgs = AsyncEngineArgs
    vllm.AsyncLLMEngine = AsyncLLMEngine
    vllm.RequestOutput = RequestOutput
    vllm.SamplingParams = SamplingParams

    vllm_inputs = types.ModuleType("vllm.inputs")

    class TokensPrompt:
        def __init__(self, prompt_token_ids: Any = None, **kw: Any) -> None:
            self.prompt_token_ids = prompt_token_ids
            self.extra = kw

    vllm_inputs.TokensPrompt = TokensPrompt

    sys.modules["vllm"] = vllm
    sys.modules["vllm.inputs"] = vllm_inputs
    return vllm, vllm_inputs


_install_vllm_stub()
for _mod in ("sllm.backends.backend_utils", "sllm.backends.gpu_backend"):
    if _mod in sys.modules:
        importlib.reload(sys.modules[_mod])

from sllm.controller import SllmController
from sllm.loading_perf_profile import load_loading_perf_profile, solve_lazy_load_method
from sllm.routers.migration_router import MigrationRouter
from sllm.routers.roundrobin_router import RoundRobinRouter, auto_scaler
from sllm.schedulers.fcfs_scheduler import FcfsScheduler
from sllm.utils import InstanceHandle, ray_cpu_actor_placement_resources, ray_gpu_actor_placement_resources
from sllm.backends.backend_utils import parse_vllm_generate_request
from sllm.backends.gpu_backend import GPUBackend

from tests.controller_test.test_generate_stream_mocked import (
    FixedTokenizer,
    build_cpu_backend_mock,
    build_cpu_router_mock,
    build_gpu_router_mock,
    build_mock_scheduler,
)


def test_vllm_import_is_stub_not_package() -> None:
    """Guarantee this suite runs without the real ``vllm`` package."""
    vllm_mod = sys.modules.get("vllm")
    assert vllm_mod is not None
    assert getattr(vllm_mod, _MARK, False) is True


# --------------------------------------------------------------------------- #
#  Utils & placement                                                          #
# --------------------------------------------------------------------------- #


def test_ray_gpu_actor_placement_resources():
    bc: Dict[str, Any] = {"ray_worker_resource_fraction": 0.01}
    r = ray_gpu_actor_placement_resources("node7", bc)
    assert r["worker_node"] == pytest.approx(0.01)
    assert r["gpu_worker_node7"] == pytest.approx(0.01)

    r2 = ray_gpu_actor_placement_resources(
        "n1", {**bc, "ray_placement_include_worker_node": False}
    )
    assert "worker_node" not in r2
    assert "gpu_worker_n1" in r2


def test_ray_cpu_actor_placement_resources_custom_and_default():
    custom = {"cpu_worker_a": 1.0}
    res, nid = ray_cpu_actor_placement_resources(
        {"cpu_placement_resources": custom, "cpu_placement_node_id": "x"}
    )
    assert res == {"cpu_worker_a": 1.0}
    assert nid == "x"

    res2, nid2 = ray_cpu_actor_placement_resources(
        {"ray_worker_resource_fraction": 0.02, "cpu_placement_node_id": "3"}
    )
    assert res2 == {"cpu_worker_3": 0.02}
    assert nid2 == "3"

    res3, nid3 = ray_cpu_actor_placement_resources({})
    assert res3 == {"worker_node": 0.001}
    assert nid3 is None


@pytest.mark.asyncio
async def test_instance_handle_queue_semantics():
    h = InstanceHandle(instance_id="i1", max_queue_length=2, ready=True)
    assert await h.add_requests(1) is True
    assert h.concurrency == 1
    assert await h.check_request_queue() is True
    assert await h.add_requests(2) is False
    h.ready = False
    assert await h.add_requests(-1) is False


# --------------------------------------------------------------------------- #
#  Round-robin router & auto scaler                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_round_robin_router_init_and_predictive_fields():
    bc = {
        "load_window_seconds": 30,
        "forecast_horizon_seconds": 5,
        "predictive_prewarm_threshold": 0.8,
    }
    r = RoundRobinRouter("m1", {"num_gpus": 2}, "vllm", bc, "gpu")
    assert r.load_window_seconds == 30
    assert r.forecast_horizon_seconds == 5
    assert r.predictive_prewarm_threshold == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_auto_scaler_clamps_to_min_max():
    n = await auto_scaler(
        {"request_count": 0},
        {"min_instances": 3, "max_instances": 10, "target": 10},
    )
    assert n == 3
    n2 = await auto_scaler(
        {"request_count": 1000},
        {"min_instances": 0, "max_instances": 4, "target": 10},
    )
    assert n2 == 4


# --------------------------------------------------------------------------- #
#  PP placement group (bundles only; Ray PG mocked)                           #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_build_pp_placement_group_bundle_shape():
    from sllm.routers.roundrobin_router import _build_pp_placement_group

    allocations = [
        {"address": "10.0.0.1", "stage_idx": 0},
        {"address": "10.0.0.2", "stage_idx": 1},
    ]
    mock_pg = MagicMock()
    mock_pg.ready = AsyncMock()

    with patch(
        "ray.util.placement_group.placement_group", return_value=mock_pg
    ) as pg_fn:
        out = await _build_pp_placement_group(allocations, tp_size=2)
    assert out is mock_pg
    bundles = pg_fn.call_args[0][0]
    assert len(bundles) == 1 + 2 * 2
    assert bundles[0] == {"CPU": 1, "node:10.0.0.1": 0.001}
    assert bundles[1] == {"GPU": 1.0, "node:10.0.0.1": 0.001}


@pytest.mark.asyncio
async def test_build_pp_placement_group_empty_returns_none():
    from sllm.routers.roundrobin_router import _build_pp_placement_group

    assert await _build_pp_placement_group([], tp_size=1) is None


# --------------------------------------------------------------------------- #
#  Placement group removal                                                      #
# --------------------------------------------------------------------------- #


def test_remove_placement_group_noop_and_calls_remove():
    from sllm.routers.roundrobin_router import _remove_placement_group

    h = InstanceHandle(instance_id="x", max_queue_length=1, placement_group=None)
    _remove_placement_group(h, "x")
    assert h.placement_group is None

    pg = MagicMock()
    h2 = InstanceHandle(instance_id="y", max_queue_length=1, placement_group=pg)
    with patch(
        "ray.util.placement_group.remove_placement_group"
    ) as rm:
        _remove_placement_group(h2, "y")
    rm.assert_called_once_with(pg)
    assert h2.placement_group is None


# --------------------------------------------------------------------------- #
#  Migration router                                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_migration_router_skips_numa_rebalance_when_pp_gt_one():
    r = MigrationRouter(
        "m",
        {"num_gpus": 4, "pp_size": 2},
        "vllm",
        {},
        "gpu",
    )
    r.model_loading_scheduler = MagicMock()
    await r._try_rebalance_for_tp()
    r.model_loading_scheduler.suggest_instance_migration.remote.assert_not_called()


# --------------------------------------------------------------------------- #
#  FCFS scheduler                                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_fcfs_normalize_parallel_request():
    s = FcfsScheduler({})
    d = s._normalize_parallel_request({"num_gpus": 8, "pp_size": 2})
    assert d["pp_size"] == 2
    assert d["tp_size"] == 4


@pytest.mark.asyncio
async def test_fcfs_report_and_predict_load():
    s = FcfsScheduler(
        {"load_window_seconds": 3600, "forecast_horizon_seconds": 10}
    )
    await s.report_model_load("mm", 2.0)
    stats = await s.get_model_load_stats("mm")
    assert stats["current_load"] == pytest.approx(2.0)
    assert "predicted_load" in stats


# --------------------------------------------------------------------------- #
#  Loading perf profile & solver                                                #
# --------------------------------------------------------------------------- #


def test_load_loading_perf_profile_has_models():
    p = load_loading_perf_profile()
    assert "models" in p
    assert "hardware" in p
    assert "Qwen3-8B" in p["models"]


def test_solve_lazy_load_method_default_profile_qwen():
    p = load_loading_perf_profile()
    out = solve_lazy_load_method(p, "Qwen3-8B", 512, {}, {}, debug=False)
    assert out is not None
    assert out[0] in ("tokenwise", "layerwise")
    assert isinstance(out[1], int)
    assert isinstance(out[2], list)


# --------------------------------------------------------------------------- #
#  Backend utils                                                                #
# --------------------------------------------------------------------------- #


def test_parse_vllm_generate_request_minimal():
    rd = {
        "prompt": "hi",
        "request_id": "r1",
        "sampling_params": {"max_tokens": 5},
    }
    model, _inputs, rid, remainder = parse_vllm_generate_request(rd, "default-model")
    assert model == "default-model"
    assert rid == "r1"
    assert remainder.get("sampling_params", {}).get("max_tokens") == 5


# --------------------------------------------------------------------------- #
#  GPU backend (lazy vs non-lazy; engine mocked)                                #
# --------------------------------------------------------------------------- #


def _gpu_backend_config(*, lazy: bool) -> Dict[str, Any]:
    return {
        "pretrained_model_name_or_path": "dummy-model",
        "load_format": "sharded_state",
        "lazy_load": lazy,
        "trace_debug": False,
    }


@pytest.mark.asyncio
async def test_gpu_backend_non_lazy_sets_flag_and_inits_full_engine():
    with patch("sllm.backends.gpu_backend.ray.get_actor", return_value=MagicMock()):
        be = GPUBackend(
            "inst",
            "dummy-model",
            "gpu",
            _gpu_backend_config(lazy=False),
            None,
        )
    assert be.lazy_load is False

    mock_engine = MagicMock()
    with patch(
        "sllm.backends.gpu_backend.ray.get_actor", return_value=MagicMock()
    ), patch(
        "sllm.backends.gpu_backend.AsyncLLMEngine.from_engine_args",
        return_value=mock_engine,
    ):
        be2 = GPUBackend(
            "inst2",
            "dummy-model",
            "gpu",
            _gpu_backend_config(lazy=False),
            None,
        )
        await be2.init_backend()
    assert be2.engine is mock_engine
    mock_engine.lazy_init.assert_not_called()


@pytest.mark.asyncio
async def test_gpu_backend_lazy_load_weights_calls_lazy_init():
    with patch("sllm.backends.gpu_backend.ray.get_actor", return_value=MagicMock()):
        be = GPUBackend(
            "inst",
            "dummy-model",
            "gpu",
            _gpu_backend_config(lazy=True),
            None,
        )
    be.engine = MagicMock()
    be.engine.lazy_init = AsyncMock()
    await be.lazy_load_weights([[0, 1]], request_id="rq")
    be.engine.lazy_init.assert_awaited_once()
    assert be.weights_loaded is True


# --------------------------------------------------------------------------- #
#  Controller (extends mocked cold / hot paths)                               #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_controller_exists_get_models():
    c = SllmController()
    c.registered_models["m"] = {"model": "m"}
    assert await c.exists("m") is True
    assert await c.exists("missing") is False
    assert "m" in await c.get_models()


def test_controller_control_node_resources():
    c = SllmController({"control_node_fraction": 0.25})
    assert c._control_node_resources() == {"control_node": 0.25}


@pytest.mark.asyncio
async def test_controller_cold_start_lazy_load_failure_propagates():
    ctrl = SllmController()
    model = "m1"
    cfg = {
        "model": model,
        "backend": "vllm",
        "num_gpus": 1,
        "backend_config": {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "load_method": "tokenwise",
            "layerwise_end_layer": 2,
            "load_method_policy": {
                "tokenwise_max_prompt_len": 1024,
                "layerwise_min_prompt_len": 2048,
            },
        },
    }
    ctrl.registered_models[model] = cfg
    ctrl.tokenizers[model] = FixedTokenizer(100)
    gpu = build_gpu_router_mock(
        pool_status={
            "loaded_ready": 0,
            "loaded_available": 0,
            "empty_ready": 0,
            "empty_starting": 0,
        },
        lazy_load_ok=False,
    )
    cpu = build_cpu_router_mock(build_cpu_backend_mock())
    ctrl.gpu_request_routers[model] = gpu
    ctrl.cpu_request_routers[model] = cpu
    ctrl.scheduler = build_mock_scheduler()

    # Avoid loading_perf profile needing a ``models[m1]`` entry.
    with patch.object(
        SllmController,
        "_generate_lazy_load_method",
        return_value=["tokenwise", 8, [0, 1]],
    ), pytest.raises(RuntimeError, match="lazy_load_weights failed"):
        await ctrl.generate_stream(model, {"prompt": "z", "request_id": "r-fail"})


@pytest.mark.asyncio
async def test_controller_hot_path_no_lazy_call_when_pool_loaded():
    """Mirrors a non-lazy / already-loaded GPU instance: direct inference only."""
    ctrl = SllmController()
    model = "m1"
    cfg = {
        "model": model,
        "backend": "vllm",
        "num_gpus": 1,
        "backend_config": {"tensor_parallel_size": 1, "pipeline_parallel_size": 1},
    }
    ctrl.registered_models[model] = cfg
    ctrl.tokenizers[model] = FixedTokenizer(50)
    gpu = build_gpu_router_mock(
        pool_status={
            "loaded_ready": 1,
            "loaded_available": 1,
            "empty_ready": 0,
            "empty_starting": 0,
        }
    )
    cpu = build_cpu_router_mock(build_cpu_backend_mock())
    ctrl.gpu_request_routers[model] = gpu
    ctrl.cpu_request_routers[model] = cpu
    ctrl.scheduler = build_mock_scheduler()

    await ctrl.generate_stream(model, {"prompt": "hi", "request_id": "r-hot"})
    assert gpu.lazy_load_weights.remote.await_count == 0
    gpu.inference.remote.assert_awaited_once()

