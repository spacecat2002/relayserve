# ---------------------------------------------------------------------------- #
#  Load external loading perf profiles and solve tokenwise vs layerwise cold    #
#  start parameters from bandwidth + analytic CPU time models.                 #
#  Solver: C++ only. Profile/solver use ms for time and GB for weight sizes.   #
# ---------------------------------------------------------------------------- #
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional


from sllm.logger import init_logger

logger = init_logger(__name__)

import sllm._loading_perf_profile_solver as solver

_BYTES_PER_GB = 1e9


def load_loading_perf_profile() -> Dict[str, Any]:
    from sllm.data.loading_perf_profile_default import LOADING_PERF_PROFILE
    return dict(LOADING_PERF_PROFILE)

def _prepare_solver_inputs(
    profile: Mapping[str, Any],
    model_name: str,
    input_length: int,
) -> Optional[Dict[str, Any]]:
    models = profile.get("models", {})
    m = dict(models[model_name])
    try:
        layer_GB = float(m["layer_GB"])
        embedding_GB = float(m["embedding_GB"])
        num_layers = int(m["num_layers"])
        tp_size = int(m.get("tp_size", 1) or 1)
        pp_size = int(m.get("pp_size", 1) or 1)
        hidden_size = int(m.get("hidden_size", 4096))
        head_dim = int(m.get("head_dim", 128))
        num_kv_heads = int(m.get("num_kv_heads", 8))
    except (KeyError, TypeError, ValueError):
        return None

    cpu_tw = m.get("cpu_tokenwise_ms", {})
    cpu_lw = m.get("cpu_layerwise_ms", {})
    gpu_tw = m.get("gpu_tokenwise_ms", {})
    gpu_lw = m.get("gpu_layerwise_ms", {})
    cpu_dec = m.get("cpu_decode_ms", {})
    hareware_info = profile.get("hardware", {})

    return {
        "fixed_ms": float(m.get("engine_init_ms", 0.0)),
        "pcie_GBps": hareware_info.get("pcie_cpu_to_gpu_GBps", 0.0),
        "rdma_GBps": hareware_info.get("rdma_node_to_node_GBps", 0.0),
        "numa_nodes": hareware_info.get("numa_nodes", 2),
        "embedding_GB": embedding_GB,
        "layer_GB": layer_GB,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "hidden_size": hidden_size,
        "tp_size": tp_size,
        "pp_size": pp_size,
        "input_length": int(input_length),
        "cpu_tw": cpu_tw,
        "cpu_lw": cpu_lw,
        "gpu_tw": gpu_tw,
        "gpu_lw": gpu_lw,
        "cpu_dec": cpu_dec,
    }


def _solve_lazy_load_method_cpp(
    inputs: Mapping[str, Any], *, debug: bool = False
) -> Optional[List[Any]]:
    solved = solver.solve_lazy_load_method_cpp(
        float(inputs["fixed_ms"]),
        float(inputs["pcie_GBps"]),
        float(inputs["rdma_GBps"]),
        int(inputs["numa_nodes"]),
        float(inputs["embedding_GB"]),
        float(inputs["layer_GB"]),
        int(inputs["num_layers"]),
        int(inputs["num_kv_heads"]),
        int(inputs["head_dim"]),
        int(inputs["hidden_size"]),
        int(inputs["tp_size"]),
        int(inputs["pp_size"]),
        int(inputs["input_length"]),
        float(inputs["cpu_tw"].get("intercept", 0.0)),
        float(inputs["cpu_tw"].get("coef_tokens", 0.0)),
        float(inputs["gpu_tw"].get("intercept", 0.0)),
        float(inputs["gpu_tw"].get("coef_tokens", 0.0)),
        float(inputs["cpu_lw"].get("intercept", 0.0)),
        float(inputs["cpu_lw"].get("coef_tokens", 0.0)),
        float(inputs["gpu_lw"].get("intercept", 0.0)),
        float(inputs["gpu_lw"].get("coef_tokens", 0.0)),
        float(inputs["cpu_dec"].get("intercept", 0.0)),
        float(inputs["cpu_dec"].get("coef_kv_tokens", 0.0)),
        debug,
    )
    if solved is None:
        return None
    if len(solved) == 4:
        method, second, layer_idxes, _ranking_ttft_ms = solved
    else:
        method, second, layer_idxes = solved
    return [str(method), int(second), list(layer_idxes)]


def _solve_lazy_load_method_cpp_with_ttft(
    inputs: Mapping[str, Any], *, debug: bool = False
) -> Optional[List[Any]]:
    """Like _solve_lazy_load_method_cpp, but also returns solver ranking TTFT."""
    solved = solver.solve_lazy_load_method_cpp(
        float(inputs["fixed_ms"]),
        float(inputs["pcie_GBps"]),
        float(inputs["rdma_GBps"]),
        int(inputs["numa_nodes"]),
        float(inputs["embedding_GB"]),
        float(inputs["layer_GB"]),
        int(inputs["num_layers"]),
        int(inputs["num_kv_heads"]),
        int(inputs["head_dim"]),
        int(inputs["hidden_size"]),
        int(inputs["tp_size"]),
        int(inputs["pp_size"]),
        int(inputs["input_length"]),
        float(inputs["cpu_tw"].get("intercept", 0.0)),
        float(inputs["cpu_tw"].get("coef_tokens", 0.0)),
        float(inputs["gpu_tw"].get("intercept", 0.0)),
        float(inputs["gpu_tw"].get("coef_tokens", 0.0)),
        float(inputs["cpu_lw"].get("intercept", 0.0)),
        float(inputs["cpu_lw"].get("coef_tokens", 0.0)),
        float(inputs["gpu_lw"].get("intercept", 0.0)),
        float(inputs["gpu_lw"].get("coef_tokens", 0.0)),
        float(inputs["cpu_dec"].get("intercept", 0.0)),
        float(inputs["cpu_dec"].get("coef_kv_tokens", 0.0)),
        debug,
    )
    if solved is None:
        return None
    if len(solved) == 4:
        method, second, layer_idxes, ranking_ttft_ms = solved
        return [str(method), int(second), list(layer_idxes), float(ranking_ttft_ms)]
    method, second, layer_idxes = solved
    return [str(method), int(second), list(layer_idxes), None]


def solve_lazy_load_method(
    profile: Mapping[str, Any],
    model_name: str,
    input_length: int,
    *_unused_backend_args: Any,
    debug: bool = False,
) -> Optional[List[Any]]:
    """
    Returns the same triple as ``SllmController._generate_lazy_load_method`` heuristic,
    or None to signal fallback.

    Implemented only by the bundled C++ solver via an installed pybind11 extension.
    Profile values use ms for time and GB for weight sizes.

    Extra positional arguments (e.g. backend_config, model_config) are accepted for
    call-site compatibility and ignored by the solver.

    If ``debug`` is True, writes a phase-by-phase timing breakdown to stderr.

    Model profile may include optional ``cpu_decode_ms`` (``intercept``, ``coef_kv_tokens``):
    decode step time is ``intercept + coef_kv_tokens * input_length``. When full-input CPU
    prefill is below ``engine_init_ms`` and the decode profile is non-zero, the solver may
    pick an **engine-overlap** tokenwise plan: ranking TTFT is CPU full-input prefill time,
    ``cpu_compute_length`` can extend to prefill plus decode steps that fit in the init
    window, and GPU preload is the largest layer prefix whose first weight DMA fits in
    ``engine_init_ms``. Otherwise tokenwise ranking uses the classic GPU-handoff TTFT
    (phase1 + KV transfer + phase2). Omit or set both coeffs to 0 to disable overlap.
    """
    inputs = _prepare_solver_inputs(
        profile,
        model_name,
        input_length,
    )
    return _solve_lazy_load_method_cpp(inputs, debug=debug)
