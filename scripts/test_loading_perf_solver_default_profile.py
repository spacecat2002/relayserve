#!/usr/bin/env python3
# ---------------------------------------------------------------------------- #
#  Smoke-test the C++ lazy-load solver against bundled default profile data.   #
#  Run from repo root (with package on PYTHONPATH or editable install):        #
#    python scripts/test_loading_perf_solver_default_profile.py                #
#                                                                              #
#  Printed TTFT columns (analytic, same formulas as loading_perf_profile_       #
#  solver.cpp):                                                                #
#    ttft_solved_ms   — Ranking TTFT for the returned plan (matches C++):      #
#      classic tokenwise: gpu handoff (phase1+KV+phase2);                      #
#      engine-overlap: CPU full-input prefill time only.                      #
#    ttft_baseline_ms — reference without lazy-load:                           #
#      pp=1: full weight DMA + engine_init + full GPU tokenwise prefill.       #
#      pp>1: pipeline with nc=0 and uniform per-stage layer counts.            #
# ---------------------------------------------------------------------------- #
from __future__ import annotations

import copy
import math
import sys
from typing import Any, List, Mapping, Optional

from sllm.data.loading_perf_profile_default import LOADING_PERF_PROFILE
from sllm.loading_perf_profile import (
    _prepare_solver_inputs,
    _solve_lazy_load_method_cpp_with_ttft,
)


# --- Analytic TTFT (mirrors sllm/data/loading_perf_profile_solver.cpp) --------

def _tokenwise_ms(intercept: float, coef_tokens: float, tokens: float) -> float:
    return intercept + coef_tokens * tokens


def _layerwise_ms(intercept: float, coef_tokens: float, tokens: float, layers: float) -> float:
    return layers * (intercept + coef_tokens * tokens)


def _weights_GB(
    embedding_GB: float,
    layer_GB: float,
    layers_to_load: int,
    numa_nodes: int,
    tp_size: int,
) -> float:
    return (embedding_GB + layer_GB * layers_to_load) / min(numa_nodes, tp_size)


# Match loading_perf_profile_solver.cpp: element counts × fp16 bytes → GB.
_FP16_BYTES = 2.0
_BYTES_TO_GB = 1e-9


def _kvcache_GB(
    input_length: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    numa_nodes: int,
    tp_size: int,
) -> float:
    return (
        2.0
        * float(input_length)
        * head_dim
        * num_kv_heads
        * num_layers
        * _FP16_BYTES
        * _BYTES_TO_GB
        / min(numa_nodes, tp_size)
    )


def _hidden_GB(
    input_length: int,
    hidden_size: int,
    numa_nodes: int,
    tp_size: int,
) -> float:
    return (
        2.0
        * float(input_length)
        * hidden_size
        * max(tp_size // numa_nodes, 1)
        * _FP16_BYTES
        * _BYTES_TO_GB
    )


def _transfer_ms(transfer_GBps: float, volume_GB: float) -> float:
    if transfer_GBps <= 0.0:
        return float("inf")
    return 1000.0 * volume_GB / transfer_GBps


def _load_stage_weights_ms(
    pcie_GBps: float,
    embedding_GB: float,
    layer_GB: float,
    stage_idx: int,
    ni: int,
    numa_nodes: int,
    tp_size: int,
) -> float:
    emb = embedding_GB if stage_idx == 0 else 0.0
    return _transfer_ms(
        pcie_GBps, _weights_GB(emb, layer_GB, ni, numa_nodes, tp_size)
    )


def _trans_to_pp_stage_ms(
    stage_idx: int,
    nc: int,
    pcie_GBps: float,
    rdma_GBps: float,
    input_length: int,
    num_kv_heads: int,
    head_dim: int,
    hidden_size: int,
    numa_nodes: int,
    tp_size: int,
) -> float:
    if stage_idx == 0:
        if nc <= 0:
            return 0.0
        vol = _kvcache_GB(
            input_length, nc, num_kv_heads, head_dim, numa_nodes, tp_size
        ) + _hidden_GB(input_length, hidden_size, numa_nodes, tp_size)
        return _transfer_ms(pcie_GBps, vol)
    return _transfer_ms(
        rdma_GBps, _hidden_GB(input_length, hidden_size, numa_nodes, tp_size)
    )


def ttft_original_no_lazy_load(inputs: Mapping[str, Any]) -> float:
    """Baseline: load full model on GPU (PCIe), engine init, full GPU tokenwise prefill."""
    pcie = float(inputs["pcie_GBps"])
    emb = float(inputs["embedding_GB"])
    layer = float(inputs["layer_GB"])
    n = int(inputs["num_layers"])
    init_ms = float(inputs["fixed_ms"])
    T = int(inputs["input_length"])
    numa = int(inputs["numa_nodes"])
    tp = int(inputs["tp_size"])
    gpu_tw = inputs["gpu_tw"]
    gi = float(gpu_tw.get("intercept", 0.0))
    gc = float(gpu_tw.get("coef_tokens", 0.0))

    xfer = _transfer_ms(pcie, _weights_GB(emb, layer, n, numa, tp))
    compute = _tokenwise_ms(gi, gc, float(T))
    return xfer + init_ms + compute


def ttft_tokenwise_py(inputs: Mapping[str, Any], cpu_length: int, gpu_load_layers: int) -> float:
    pcie = float(inputs["pcie_GBps"])
    emb = float(inputs["embedding_GB"])
    layer = float(inputs["layer_GB"])
    n = int(inputs["num_layers"])
    init_ms = float(inputs["fixed_ms"])
    T = int(inputs["input_length"])
    hidden = int(inputs["hidden_size"])
    nkv = int(inputs["num_kv_heads"])
    hd = int(inputs["head_dim"])
    numa = int(inputs["numa_nodes"])
    tp = int(inputs["tp_size"])
    ctw, gtw = inputs["cpu_tw"], inputs["gpu_tw"]
    ci, cc = float(ctw.get("intercept", 0.0)), float(ctw.get("coef_tokens", 0.0))
    gi, gc = float(gtw.get("intercept", 0.0)), float(gtw.get("coef_tokens", 0.0))

    w1 = _weights_GB(emb, layer, gpu_load_layers, numa, tp)
    lat1 = max(
        _transfer_ms(pcie, w1),
        _tokenwise_ms(ci, cc, float(cpu_length)),
        init_ms,
    )
    kv = _kvcache_GB(cpu_length, n, nkv, hd, numa, tp)
    xfer_kv = _transfer_ms(pcie, kv)
    rem = n - gpu_load_layers
    w2 = _weights_GB(emb, layer, rem, numa, tp)
    gpu_tok = max(T - cpu_length, 0)
    lat2 = max(
        _tokenwise_ms(gi, gc, float(gpu_tok)),
        _transfer_ms(pcie, w2),
    )
    return lat1 + xfer_kv + lat2


def ttft_tokenwise_ranking_ms(
    inputs: Mapping[str, Any], second: int, gpu_load: int
) -> float:
    """
    TTFT used by the C++ solver to *rank* tokenwise plans (pp=1).

    Engine-overlap (cpu_decode_ms set, CPU full prefill < engine_init, first
    weight xfer <= engine_init): the winning plan's score is CPU full-input
    prefill ms, not gpu_handoff_ttft. The smoke test must use the same rule so
    ttft_solved_ms matches what the solver optimizes.
    """
    T = int(inputs["input_length"])
    init_ms = float(inputs["fixed_ms"])
    classic = ttft_tokenwise_py(inputs, min(second, T), gpu_load)

    cd = inputs.get("cpu_dec") or {}
    di = float(cd.get("intercept", 0.0))
    dc = float(cd.get("coef_kv_tokens", 0.0))
    decode_ok = di != 0.0 or dc != 0.0
    ctw = inputs["cpu_tw"]
    ci = float(ctw.get("intercept", 0.0))
    cc = float(ctw.get("coef_tokens", 0.0))
    cpu_pf = _tokenwise_ms(ci, cc, float(T))

    if not decode_ok or T <= 0:
        return classic
    if not (cpu_pf + 1e-9 < init_ms):
        return classic

    pcie = float(inputs["pcie_GBps"])
    emb = float(inputs["embedding_GB"])
    layer = float(inputs["layer_GB"])
    n = int(inputs["num_layers"])
    numa = int(inputs["numa_nodes"])
    tp = int(inputs["tp_size"])
    w1 = _weights_GB(emb, layer, gpu_load, numa, tp)
    if _transfer_ms(pcie, w1) > init_ms + 1e-9:
        return classic

    decode_step = di + dc * float(T)
    if decode_step <= 0.0:
        decode_step = 1e-6
    gpu_load_ov = 0
    for L in range(n, 0, -1):
        wgb = _weights_GB(emb, layer, L, numa, tp)
        if _transfer_ms(pcie, wgb) <= init_ms + 1e-9:
            gpu_load_ov = L
            break
    if gpu_load_ov <= 0:
        return classic

    ttft_ov = cpu_pf
    # Overlap identity for reporting: decode profile active, max preload layers,
    # and cpu_compute_length at least reaches full prefill. k can be zero.
    is_overlap_shape = second >= T and gpu_load == gpu_load_ov
    if not is_overlap_shape:
        return classic
    if classic > ttft_ov - 1e-9:
        return ttft_ov
    return classic


def ttft_layerwise_py(
    inputs: Mapping[str, Any],
    cpu_compute_layers: int,
    gpu_load_layers: int,
) -> float:
    pcie = float(inputs["pcie_GBps"])
    emb = float(inputs["embedding_GB"])
    layer = float(inputs["layer_GB"])
    n = int(inputs["num_layers"])
    init_ms = float(inputs["fixed_ms"])
    T = int(inputs["input_length"])
    hidden = int(inputs["hidden_size"])
    nkv = int(inputs["num_kv_heads"])
    hd = int(inputs["head_dim"])
    numa = int(inputs["numa_nodes"])
    tp = int(inputs["tp_size"])
    clw, glw = inputs["cpu_lw"], inputs["gpu_lw"]
    cli, clc = float(clw.get("intercept", 0.0)), float(clw.get("coef_tokens", 0.0))
    gli, glc = float(glw.get("intercept", 0.0)), float(glw.get("coef_tokens", 0.0))

    load_ms = _transfer_ms(pcie, _weights_GB(emb, layer, gpu_load_layers, numa, tp))
    cpu_ms = _layerwise_ms(cli, clc, float(T), float(cpu_compute_layers))
    p1 = max(load_ms, cpu_ms, init_ms)
    kv = _kvcache_GB(T, cpu_compute_layers, nkv, hd, numa, tp)
    act = _hidden_GB(T, hidden, numa, tp)
    p2 = _transfer_ms(pcie, kv + act)
    gpu_L = n - cpu_compute_layers
    p3 = _layerwise_ms(gli, glc, float(T), float(gpu_L))
    return p1 + p2 + p3


def _uniform_pp_stage_sizes(num_layers: int, pp_size: int) -> List[int]:
    target = num_layers // pp_size
    sizes: List[int] = []
    for i in range(pp_size):
        lo = i * target
        hi = num_layers if i == pp_size - 1 else (i + 1) * target
        sizes.append(hi - lo)
    return sizes


def pp_ttft_layer_pipeline_py(
    inputs: Mapping[str, Any],
    nc: int,
    n_gpu: List[int],
) -> float:
    pp_size = len(n_gpu)
    n = int(inputs["num_layers"])
    init_ms = float(inputs["fixed_ms"])
    pcie = float(inputs["pcie_GBps"])
    rdma = float(inputs["rdma_GBps"])
    emb = float(inputs["embedding_GB"])
    layer = float(inputs["layer_GB"])
    T = int(inputs["input_length"])
    nkv = int(inputs["num_kv_heads"])
    hd = int(inputs["head_dim"])
    hidden = int(inputs["hidden_size"])
    numa = int(inputs["numa_nodes"])
    tp = int(inputs["tp_size"])
    clw, glw = inputs["cpu_lw"], inputs["gpu_lw"]
    cli, clc = float(clw.get("intercept", 0.0)), float(clw.get("coef_tokens", 0.0))
    gli, glc = float(glw.get("intercept", 0.0)), float(glw.get("coef_tokens", 0.0))

    per_cpu = cli + clc * T
    t_cpu = nc * per_cpu
    if nc >= n:
        return max(init_ms, t_cpu)

    per_gpu = gli + glc * T
    ends: List[float] = []
    for i in range(pp_size):
        ni = n_gpu[i]
        if ni == 0:
            all_zero = all(n_gpu[j] == 0 for j in range(i + 1, pp_size))
            if all_zero:
                ends.append(t_cpu)
            else:
                ends.append(
                    ends[i - 1]
                    + _trans_to_pp_stage_ms(
                        i,
                        nc,
                        pcie,
                        rdma,
                        T,
                        nkv,
                        hd,
                        hidden,
                        numa,
                        tp,
                    )
                )
            continue
        upstream = ends[i - 1] if i > 0 else t_cpu
        trans = _trans_to_pp_stage_ms(
            i, nc, pcie, rdma, T, nkv, hd, hidden, numa, tp
        )
        ready = max(upstream + trans, init_ms)
        compute_done = ready + ni * per_gpu
        load_done = _load_stage_weights_ms(
            pcie, emb, layer, i, ni, numa, tp
        )
        ends.append(max(compute_done, load_done))
    return ends[-1]


def pp_ttft_token_cpu_py(
    inputs: Mapping[str, Any],
    n_gpu: List[int],
    s_cpu: int,
) -> float:
    pp_size = len(n_gpu)
    n = int(inputs["num_layers"])
    init_ms = float(inputs["fixed_ms"])
    pcie = float(inputs["pcie_GBps"])
    rdma = float(inputs["rdma_GBps"])
    emb = float(inputs["embedding_GB"])
    layer = float(inputs["layer_GB"])
    T = int(inputs["input_length"])
    nkv = int(inputs["num_kv_heads"])
    hd = int(inputs["head_dim"])
    hidden = int(inputs["hidden_size"])
    numa = int(inputs["numa_nodes"])
    tp = int(inputs["tp_size"])
    ctw, gtw = inputs["cpu_tw"], inputs["gpu_tw"]
    ci, cc = float(ctw.get("intercept", 0.0)), float(ctw.get("coef_tokens", 0.0))
    gi, gc = float(gtw.get("intercept", 0.0)), float(gtw.get("coef_tokens", 0.0))

    s_total = float(max(T, 1))
    frac_cpu = min(max(s_cpu, 0), T) / s_total
    frac_gpu = 1.0 - frac_cpu
    t_cpu_done = n * frac_cpu * _tokenwise_ms(ci, cc, float(T))
    if frac_gpu <= 0.0:
        return max(init_ms, t_cpu_done)

    per_gpu_full = gi + gc * T
    ends: List[float] = []
    for i in range(pp_size):
        ni = n_gpu[i]
        if ni == 0:
            all_zero = all(n_gpu[j] == 0 for j in range(i + 1, pp_size))
            if all_zero:
                ends.append(t_cpu_done)
            else:
                ends.append(
                    ends[i - 1]
                    + _trans_to_pp_stage_ms(
                        i, 0, pcie, rdma, T, nkv, hd, hidden, numa, tp
                    )
                )
            continue
        upstream = ends[i - 1] if i > 0 else t_cpu_done
        trans = _trans_to_pp_stage_ms(
            i, 0, pcie, rdma, T, nkv, hd, hidden, numa, tp
        )
        ready = max(upstream + trans, init_ms)
        compute_done = ready + ni * per_gpu_full * frac_gpu
        load_done = _load_stage_weights_ms(
            pcie, emb, layer, i, ni, numa, tp
        )
        ends.append(max(compute_done, load_done))
    return ends[-1]


def _n_gpu_from_layer_idxes_pp(layer_idxes: List[Any], pp_size: int) -> List[int]:
    stages: List[List[int]] = []
    for s in layer_idxes:
        if isinstance(s, list):
            stages.append([int(x) for x in s])
        else:
            stages.append([int(s)])
    assert len(stages) == pp_size
    return [len(st) for st in stages]


def ttft_for_solved_triple(
    inputs: Mapping[str, Any],
    triple: List[Any],
    pp_size: int,
) -> float:
    method, second, layer_idxes = triple[0], int(triple[1]), triple[2]
    if pp_size <= 1:
        if method == "tokenwise":
            gpu_load = len(layer_idxes)
            return ttft_tokenwise_ranking_ms(inputs, second, gpu_load)
        gpu_load = len(layer_idxes)
        return ttft_layerwise_py(inputs, second, gpu_load)

    n_gpu = _n_gpu_from_layer_idxes_pp(list(layer_idxes), pp_size)
    if method == "layerwise":
        return pp_ttft_layer_pipeline_py(inputs, second, n_gpu)
    return pp_ttft_token_cpu_py(inputs, n_gpu, second)


def ttft_original_pp_uniform(inputs: Mapping[str, Any], pp_size: int) -> float:
    """Baseline PP: no CPU layers, uniform stage split (same global layer count as std intervals)."""
    sizes = _uniform_pp_stage_sizes(int(inputs["num_layers"]), pp_size)
    return pp_ttft_layer_pipeline_py(inputs, 0, sizes)


def _flatten_layer_idxes(layer_idxes: List[Any]) -> List[int]:
    out: List[int] = []
    for x in layer_idxes:
        if isinstance(x, list):
            out.extend(_flatten_layer_idxes(x))
        else:
            out.append(int(x))
    return out


def _validate_triple(
    model_name: str,
    input_length: int,
    pp_size: int,
    num_layers: int,
    triple: Optional[List[Any]],
) -> None:
    if triple is None:
        raise AssertionError(
            f"solver returned None for {model_name} len={input_length} pp={pp_size} "
            "(C++ extension missing or inputs invalid?)"
        )
    method, second, layer_idxes = triple
    assert method in ("tokenwise", "layerwise"), method
    assert isinstance(second, int), type(second)

    if pp_size <= 1:
        assert isinstance(layer_idxes, list), type(layer_idxes)
        assert all(isinstance(i, int) for i in layer_idxes), layer_idxes
    else:
        assert isinstance(layer_idxes, list) and len(layer_idxes) == pp_size, (
            f"PP expected list of {pp_size} stage lists, got {layer_idxes!r}"
        )
        for i, stage in enumerate(layer_idxes):
            assert isinstance(stage, list), (i, stage)
            assert all(isinstance(j, int) for j in stage), stage

    flat = _flatten_layer_idxes(layer_idxes)
    assert all(0 <= i < num_layers for i in flat), (flat, num_layers)

    if pp_size > 1:
        if method == "layerwise":
            nc = second
            assert sorted(flat) == list(range(nc, num_layers)), (
                f"PP layerwise: expect GPU global layers [{nc}, {num_layers}), "
                f"got {sorted(flat)}"
            )
        else:
            assert sorted(flat) == list(range(num_layers)), (
                f"PP tokenwise: expect all global layers [0, {num_layers}), "
                f"got {sorted(flat)}"
            )
    elif method == "layerwise":
        assert second >= 0, second


def _profile_with_pp(
    base: Mapping[str, Any], model_name: str, pp_size: int
) -> dict[str, Any]:
    p = copy.deepcopy(dict(base))
    m = dict(p["models"][model_name])
    m["pp_size"] = pp_size
    p["models"] = dict(p["models"])
    p["models"][model_name] = m
    return p


def main() -> None:
    model_name = "Qwen3-8B"
    if model_name not in LOADING_PERF_PROFILE.get("models", {}):
        print(f"Model {model_name!r} not in default profile", file=sys.stderr)
        sys.exit(1)

    m = LOADING_PERF_PROFILE["models"][model_name]
    num_layers = int(m["num_layers"])

    print("=== Default profile (pp_size=1), Qwen3-8B ===")
    print(
        f"  {'len':>5}  {'method':10}  {'ttft_solved_ms':>14}  "
        f"{'ttft_baseline_ms':>16}  {'delta_ms':>10}  triple"
    )
    for prompt_len in range(128, 4096 + 1, 128):
        inputs = _prepare_solver_inputs(LOADING_PERF_PROFILE, model_name, prompt_len)
        assert inputs is not None
        solved = _solve_lazy_load_method_cpp_with_ttft(inputs)
        assert solved is not None
        triple = solved[:3]
        ranking_ttft_ms = solved[3]
        _validate_triple(model_name, prompt_len, 1, num_layers, triple)
        # t_sol = (
        #     float(ranking_ttft_ms)
        #     if ranking_ttft_ms is not None
        #     else ttft_for_solved_triple(inputs, triple, 1)
        # )
        t_sol = ranking_ttft_ms
        t_base = ttft_original_no_lazy_load(inputs)
        print(
            f"  {prompt_len:5d}  {triple[0]!s:10}  {t_sol:14.2f}  "
            f"{t_base:16.2f}  {t_sol - t_base:10.2f}  {triple}"
        )

    print("\n=== Synthetic PP (pp_size=4), same coefficients / hardware ===")
    print(
        f"  {'len':>5}  {'method':10}  {'ttft_solved_ms':>14}  "
        f"{'ttft_baseline_ms':>16}  {'delta_ms':>10}  triple"
    )
    pp_profile = _profile_with_pp(LOADING_PERF_PROFILE, model_name, pp_size=4)
    for prompt_len in (512, 2048):
        inputs = _prepare_solver_inputs(pp_profile, model_name, prompt_len)
        assert inputs is not None
        solved = _solve_lazy_load_method_cpp_with_ttft(inputs, debug=True)
        assert solved is not None
        triple = solved[:3]
        ranking_ttft_ms = solved[3]
        _validate_triple(model_name, prompt_len, 4, num_layers, triple)
        t_sol = (
            float(ranking_ttft_ms)
            if ranking_ttft_ms is not None
            else ttft_for_solved_triple(inputs, triple, 4)
        )
        t_base = ttft_original_pp_uniform(inputs, 4)
        print(
            f"  {prompt_len:5d}  {triple[0]!s:10}  {t_sol:14.2f}  "
            f"{t_base:16.2f}  {t_sol - t_base:10.2f}  {triple}"
        )

    print("\nAll checks passed.")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(
            "Import failed (install editable package from repo root):\n"
            "  pip install -e .\n"
            f"Detail: {e}",
            file=sys.stderr,
        )
        sys.exit(1)
