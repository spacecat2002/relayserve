# ---------------------------------------------------------------------------- #
#  Mocked async tests for SllmController.generate_stream (no Ray / no vLLM).  #
# ---------------------------------------------------------------------------- #
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest

from sllm.controller import SllmController


def build_mock_scheduler(
    *,
    node_id: Optional[str] = "node-1",
    cold_status: Tuple[bool, str] = (False, "tokenwise"),
) -> MagicMock:
    sched = MagicMock()

    async def _get_node(mn: str) -> Optional[str]:  # noqa: ARG001
        return node_id

    async def _cold(nid: str, mn: str) -> Tuple[bool, str]:  # noqa: ARG001
        return cold_status

    async def _noop(*_a: Any, **_k: Any) -> None:
        return None

    for name, fn in (
        ("get_node_for_model", _get_node),
        ("get_cold_start_status", _cold),
        ("wait_cold_start_ready", _noop),
        ("start_cold_start", _noop),
        ("finish_cold_start", _noop),
        ("signal_cold_start_ready", _noop),
    ):
        m = MagicMock()
        m.remote = AsyncMock(side_effect=fn)
        setattr(sched, name, m)
    return sched


def build_gpu_router_mock(
    *,
    pool_status: Dict[str, int],
    inference_result: Any = None,
    lazy_load_ok: bool = True,
) -> MagicMock:
    gpu = MagicMock()

    async def _pool() -> Dict[str, int]:
        return dict(pool_status)

    gps = MagicMock()
    gps.remote = AsyncMock(side_effect=_pool)
    gpu.get_instance_pool_status = gps

    async def _infer(**kwargs: Any) -> Any:
        del kwargs
        return inference_result if inference_result is not None else _gpu_inference_result()

    inf = MagicMock()
    inf.remote = AsyncMock(side_effect=_infer)
    gpu.inference = inf

    async def _lazy(**kwargs: Any) -> None:
        del kwargs
        if not lazy_load_ok:
            raise RuntimeError("lazy_load_weights failed")

    ll = MagicMock()
    ll.remote = AsyncMock(side_effect=_lazy)
    gpu.lazy_load_weights = ll

    ee = MagicMock()
    ee.remote = AsyncMock(return_value=False)
    gpu.ensure_empty_instance = ee
    return gpu


def _gpu_inference_result() -> Tuple[Any, Dict[str, Any]]:
    now = time.perf_counter()
    return (
        {"choices": [{"text": "ok"}]},
        {
            "e2e": 0.05,
            "ttft": 0.01,
            "tpot": 0.02,
            "output_length": 3,
            "first_token_time": now,
            "itls": [],
        },
    )


def _cpu_tokenwise_pair_for_gpu() -> Tuple[Any, Dict[str, Any]]:
    return (
        {
            "choices": [{"text": " cpu_prefill"}],
            "kv_transfer_params": {"remote_block_ids": [7, 8]},
        },
        {"ttft": 0.02, "output_length": 1, "itls": [0.01]},
    )


def build_cpu_backend_mock(
    *,
    tokenwise_generate: Optional[Tuple[Any, Any]] = None,
    generate_remote_fire_and_forget: bool = False,
) -> MagicMock:
    be = MagicMock()
    ucl = MagicMock()
    ucl.remote = AsyncMock(return_value=None)
    be.update_computing_layers = ucl

    pair = tokenwise_generate or _cpu_tokenwise_pair_for_gpu()

    gen = MagicMock()
    if generate_remote_fire_and_forget:
        gen.remote = MagicMock(return_value=None)
    else:
        async def _gen(**kwargs: Any) -> Any:
            del kwargs
            return pair

        gen.remote = AsyncMock(side_effect=_gen)
    be.generate = gen
    return be


def build_cpu_router_mock(
    cpu_backend: MagicMock,
    *,
    inference_remote_fire_and_forget: bool = False,
) -> MagicMock:
    cpu = MagicMock()
    inst = MagicMock()
    inst.backend_instance = cpu_backend
    gi = MagicMock()
    gi.remote = AsyncMock(return_value=inst)
    cpu.get_instance = gi

    inf = MagicMock()
    if inference_remote_fire_and_forget:
        inf.remote = MagicMock(return_value=None)
    else:
        async def _infer(**kwargs: Any) -> Any:
            del kwargs
            return {"ok": True, "path": "cpu_only"}

        inf.remote = AsyncMock(side_effect=_infer)
    cpu.inference = inf
    return cpu


class FixedTokenizer:
    def __init__(self, length: int) -> None:
        self._length = length

    def get_prompt_len(self, prompt: str) -> int:  # noqa: ARG002
        return self._length


@pytest.fixture
def base_model_config() -> Dict[str, Any]:
    return {
        "model": "m1",
        "backend": "mock",
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


@pytest.mark.asyncio
async def test_direct_gpu_when_loaded_capacity_available(base_model_config: Dict[str, Any]):
    ctrl = SllmController()
    model = "m1"
    ctrl.registered_models[model] = base_model_config
    ctrl.tokenizers[model] = FixedTokenizer(100)
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

    out = await ctrl.generate_stream(model, {"prompt": "hi", "request_id": "r1"})
    assert isinstance(out, tuple)
    gpu.inference.remote.assert_called_once()
    cpu.inference.remote.assert_not_called()


@pytest.mark.asyncio
async def test_cpu_only_then_ensure_empty_when_loaded_but_saturated_no_prewarm(
    base_model_config: Dict[str, Any],
):
    ctrl = SllmController()
    model = "m1"
    ctrl.registered_models[model] = base_model_config
    ctrl.tokenizers[model] = FixedTokenizer(100)

    gpu = build_gpu_router_mock(
        pool_status={
            "loaded_ready": 1,
            "loaded_available": 0,
            "empty_ready": 0,
            "empty_starting": 0,
        }
    )
    cpu = build_cpu_router_mock(build_cpu_backend_mock())
    ctrl.gpu_request_routers[model] = gpu
    ctrl.cpu_request_routers[model] = cpu
    ctrl.scheduler = build_mock_scheduler()

    out = await ctrl.generate_stream(model, {"prompt": "x", "request_id": "r2"})
    assert out == {"ok": True, "path": "cpu_only"}
    cpu.inference.remote.assert_called_once()
    gpu.ensure_empty_instance.remote.assert_called_once()


@pytest.mark.asyncio
async def test_amx_forced_tokenwise_when_loaded_saturated_and_empty_prewarm(
    base_model_config: Dict[str, Any],
):
    ctrl = SllmController()
    model = "m1"
    ctrl.registered_models[model] = base_model_config
    ctrl.tokenizers[model] = FixedTokenizer(100)

    gpu = build_gpu_router_mock(
        pool_status={
            "loaded_ready": 1,
            "loaded_available": 0,
            "empty_ready": 1,
            "empty_starting": 0,
        }
    )
    be = build_cpu_backend_mock()
    cpu = build_cpu_router_mock(be)
    ctrl.gpu_request_routers[model] = gpu
    ctrl.cpu_request_routers[model] = cpu
    ctrl.scheduler = build_mock_scheduler()

    await ctrl.generate_stream(model, {"prompt": "y", "request_id": "r3"})

    calls = gpu.lazy_load_weights.remote.call_args_list
    assert len(calls) == 1
    assert calls[0].kwargs.get("load_method") == "tokenwise"


@pytest.mark.asyncio
async def test_cold_start_tokenwise_full_path(base_model_config: Dict[str, Any]):
    ctrl = SllmController()
    model = "m1"
    ctrl.registered_models[model] = base_model_config
    ctrl.tokenizers[model] = FixedTokenizer(100)

    gpu = build_gpu_router_mock(
        pool_status={
            "loaded_ready": 0,
            "loaded_available": 0,
            "empty_ready": 0,
            "empty_starting": 0,
        }
    )
    cpu = build_cpu_router_mock(build_cpu_backend_mock())
    ctrl.gpu_request_routers[model] = gpu
    ctrl.cpu_request_routers[model] = cpu
    ctrl.scheduler = build_mock_scheduler()

    await ctrl.generate_stream(model, {"prompt": "z", "request_id": "r4"})
    gpu.lazy_load_weights.remote.assert_called_once()
    assert gpu.lazy_load_weights.remote.call_args.kwargs.get("load_method") == "tokenwise"
    assert gpu.inference.remote.call_count >= 1


@pytest.mark.asyncio
async def test_cold_start_layerwise_long_prompt(base_model_config: Dict[str, Any]):
    ctrl = SllmController()
    model = "m1"
    cfg = dict(base_model_config)
    ctrl.registered_models[model] = cfg
    ctrl.tokenizers[model] = FixedTokenizer(3000)

    gpu = build_gpu_router_mock(
        pool_status={
            "loaded_ready": 0,
            "loaded_available": 0,
            "empty_ready": 0,
            "empty_starting": 0,
        }
    )
    cpu = build_cpu_router_mock(
        build_cpu_backend_mock(generate_remote_fire_and_forget=True),
    )
    ctrl.gpu_request_routers[model] = gpu
    ctrl.cpu_request_routers[model] = cpu
    ctrl.scheduler = build_mock_scheduler()

    await ctrl.generate_stream(model, {"prompt": "long", "request_id": "r5"})
    assert gpu.lazy_load_weights.remote.call_args.kwargs.get("load_method") == "layerwise"


@pytest.mark.asyncio
async def test_piggyback_tokenwise_only_gpu(base_model_config: Dict[str, Any]):
    ctrl = SllmController()
    model = "m1"
    ctrl.registered_models[model] = base_model_config
    ctrl.tokenizers[model] = FixedTokenizer(100)

    gpu = build_gpu_router_mock(
        pool_status={
            "loaded_ready": 0,
            "loaded_available": 0,
            "empty_ready": 0,
            "empty_starting": 0,
        }
    )
    cpu = build_cpu_router_mock(build_cpu_backend_mock())
    ctrl.gpu_request_routers[model] = gpu
    ctrl.cpu_request_routers[model] = cpu
    ctrl.scheduler = build_mock_scheduler(node_id="n1", cold_status=(True, "tokenwise"))

    await ctrl.generate_stream(model, {"prompt": "p", "request_id": "r6"})
    gpu.inference.remote.assert_called_once()
    cpu.inference.remote.assert_not_called()


@pytest.mark.asyncio
async def test_piggyback_pp_layerwise_only_gpu(base_model_config: Dict[str, Any]):
    ctrl = SllmController()
    model = "m1"
    cfg = dict(base_model_config)
    cfg["pipeline_parallel_size"] = 2
    cfg["backend_config"] = dict(cfg["backend_config"])
    cfg["backend_config"]["pipeline_parallel_size"] = 2
    ctrl.registered_models[model] = cfg
    ctrl.tokenizers[model] = FixedTokenizer(100)

    gpu = build_gpu_router_mock(
        pool_status={
            "loaded_ready": 0,
            "loaded_available": 0,
            "empty_ready": 0,
            "empty_starting": 0,
        }
    )
    cpu = build_cpu_router_mock(build_cpu_backend_mock())
    ctrl.gpu_request_routers[model] = gpu
    ctrl.cpu_request_routers[model] = cpu
    ctrl.scheduler = build_mock_scheduler(node_id="n1", cold_status=(True, "layerwise"))

    await ctrl.generate_stream(model, {"prompt": "pp", "request_id": "r7"})
    gpu.inference.remote.assert_called_once()
    cpu.inference.remote.assert_not_called()


@pytest.mark.asyncio
async def test_piggyback_non_pp_layerwise_cpu_and_gpu(base_model_config: Dict[str, Any]):
    ctrl = SllmController()
    model = "m1"
    ctrl.registered_models[model] = base_model_config
    ctrl.tokenizers[model] = FixedTokenizer(100)

    gpu = build_gpu_router_mock(
        pool_status={
            "loaded_ready": 0,
            "loaded_available": 0,
            "empty_ready": 0,
            "empty_starting": 0,
        }
    )
    cpu = build_cpu_router_mock(
        build_cpu_backend_mock(),
        inference_remote_fire_and_forget=True,
    )
    ctrl.gpu_request_routers[model] = gpu
    ctrl.cpu_request_routers[model] = cpu
    ctrl.scheduler = build_mock_scheduler(node_id="n1", cold_status=(True, "layerwise"))

    await ctrl.generate_stream(model, {"prompt": "lw", "request_id": "r8"})
    assert cpu.inference.remote.called
    gpu.inference.remote.assert_called_once()


@pytest.mark.asyncio
async def test_no_gpu_node_returns_error(base_model_config: Dict[str, Any]):
    ctrl = SllmController()
    model = "m1"
    ctrl.registered_models[model] = base_model_config
    ctrl.tokenizers[model] = FixedTokenizer(100)

    gpu = build_gpu_router_mock(
        pool_status={
            "loaded_ready": 0,
            "loaded_available": 0,
            "empty_ready": 0,
            "empty_starting": 0,
        }
    )
    cpu = build_cpu_router_mock(build_cpu_backend_mock())
    ctrl.gpu_request_routers[model] = gpu
    ctrl.cpu_request_routers[model] = cpu
    ctrl.scheduler = build_mock_scheduler(node_id=None)

    out = await ctrl.generate_stream(model, {"prompt": "e", "request_id": "r9"})
    assert out == {"error": "No GPU node allocated for this model"}
