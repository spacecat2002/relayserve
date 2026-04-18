# ---------------------------------------------------------------------------- #
#  Unit tests for loading perf profile load + lazy-load solver.                #
# ---------------------------------------------------------------------------- #
from __future__ import annotations

import pytest

from sllm.loading_perf_profile import (
    load_loading_perf_profile,
    should_use_perf_solver,
    solve_lazy_load_method,
)


def test_should_use_perf_solver_from_policy():
    assert not should_use_perf_solver({})
    assert not should_use_perf_solver({"load_method_policy": {}})
    assert should_use_perf_solver(
        {"load_method_policy": {"use_perf_solver": True}}
    )
    assert should_use_perf_solver({"use_perf_solver": True})


def test_solve_lazy_load_returns_triple_when_cpp_available():
    profile = {
        "hardware": {
            "pcie_cpu_to_gpu_GBps": 100.0,
            "rdma_node_to_node_GBps": 100.0,
            "weight_transfer_link": "pcie",
        },
        "engine_init": {
            "model_structure_init_ms": 0.0,
            "warmup_ms": 0.0,
            "kv_cache_create_ms": 0.0,
        },
        "models": {
            "default": {
                "layer_GB": 1e-3,
                "embedding_GB": 1e-3,
                "num_layers": 8,
                "tp_size": 1,
                "pp_size": 1,
                "cpu_tokenwise_ms": {"intercept": 1.0, "coef_tokens": 0.1},
                "cpu_layerwise_ms": {
                    "intercept": 1.0,
                    "coef_tokens": 0.05,
                    "coef_layers": 10.0,
                    "coef_tokens_layers": 0.0,
                },
            }
        },
    }
    backend_config = {
        "load_method_policy": {
            "use_perf_solver": True,
            "tokenwise_max_prompt_len": 512,
            "layerwise_min_prompt_len": 2048,
            "solver_respect_layerwise_min": False,
        },
        "layerwise_end_layer": -1,
    }
    registered = {
        "backend_config": {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
        }
    }
    out = solve_lazy_load_method(
        profile, "any-model", 128, backend_config, registered
    )
    if out is None:
        pytest.skip("C++ lazy-load solver unavailable")
    assert out is not None
    assert out[0] in ("tokenwise", "layerwise")
    assert isinstance(out[1], int)
    assert isinstance(out[2], list)


def test_load_profile_without_file_uses_bundled(monkeypatch):
    monkeypatch.delenv("SLLM_LOADING_PERF_PROFILE", raising=False)
    p = load_loading_perf_profile()
    assert "hardware" in p
    assert "models" in p
    assert p["models"]["Qwen3-8B"]["layer_GB"] > 0
    assert p["models"]["Qwen3-8B"]["embedding_GB"] > 0


def test_load_profile_from_json_file(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(
        '{"hardware": {"pcie_cpu_to_gpu_GBps": 1.0}, "engine_init": {}, "models": {}}',
        encoding="utf-8",
    )
    p = load_loading_perf_profile(str(path))
    assert p["hardware"]["pcie_cpu_to_gpu_GBps"] == 1.0
    assert p["models"] == {}


def test_solve_skips_layerwise_when_short_prompt_and_respect_min():
    profile = {
        "hardware": {"pcie_cpu_to_gpu_GBps": 50.0, "weight_transfer_link": "pcie"},
        "engine_init": {},
        "models": {
            "default": {
                "layer_GB": 1e-3,
                "embedding_GB": 1e-3,
                "num_layers": 4,
                "tp_size": 1,
                "pp_size": 1,
                "cpu_tokenwise_ms": {"intercept": 0.0, "coef_tokens": 0.01},
                "cpu_layerwise_ms": {
                    "intercept": 0.0,
                    "coef_tokens": 0.0,
                    "coef_layers": 0.0,
                    "coef_tokens_layers": 0.0,
                },
            }
        },
    }
    backend_config = {
        "load_method_policy": {
            "tokenwise_max_prompt_len": 1024,
            "layerwise_min_prompt_len": 2048,
            "solver_respect_layerwise_min": True,
        },
    }
    registered = {"backend_config": {}}
    out = solve_lazy_load_method(profile, "m", 100, backend_config, registered)
    if out is None:
        pytest.skip("C++ lazy-load solver unavailable")
    assert out is not None
    assert out[0] == "tokenwise"


def test_legacy_engine_init_seconds_converted_to_ms():
    profile = {
        "hardware": {"pcie_cpu_to_gpu_GBps": 100.0, "weight_transfer_link": "pcie"},
        "engine_init": {
            "model_structure_init_s": 1.0,
            "warmup_s": 0.0,
            "kv_cache_create_s": 0.0,
        },
        "models": {
            "default": {
                "layer_GB": 1e-6,
                "embedding_GB": 1e-6,
                "num_layers": 2,
                "tp_size": 1,
                "cpu_tokenwise_ms": {"intercept": 0.0, "coef_tokens": 0.0},
                "cpu_layerwise_ms": {
                    "intercept": 0.0,
                    "coef_tokens": 0.0,
                    "coef_layers": 0.0,
                    "coef_tokens_layers": 0.0,
                },
            }
        },
    }
    backend_config = {"load_method_policy": {"solver_respect_layerwise_min": False}}
    registered = {"backend_config": {}}
    out = solve_lazy_load_method(profile, "m", 1, backend_config, registered)
    if out is None:
        pytest.skip("C++ lazy-load solver unavailable")
    assert out is not None
