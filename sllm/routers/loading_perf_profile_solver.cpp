#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

constexpr double kBetaStd = 0.1;  // TTFT + kBetaStd * pen_std (test_pp.py)
constexpr int kTokenCpuStep = 32;
// KV / activation volumes are counted in elements; convert to GB for transfer_ms (fp16).
constexpr double kBF16Bytes = 2.0;
constexpr double kBytesToGB = 1e-9;

double tokenwise_ms(double intercept, double coef_tokens, double tokens) {
    return intercept + coef_tokens * tokens;
}

double layerwise_ms(
    double intercept,
    double coef_tokens,
    double tokens,
    double layers
) {
    return layers * (intercept + coef_tokens * tokens);
}

// Total weight volume (GB) moved for this load plan, per TP rank.
double weights_GB(
    double embedding_GB,
    double layer_GB,
    int layers_to_load,
    int numa_nodes,
    int tp_size
) {
    return (embedding_GB + layer_GB * layers_to_load) / std::min(numa_nodes, tp_size);
}

double kvcache_GB(
    int input_length,
    int num_layers,
    int num_kv_heads,
    int head_dim,
    int numa_nodes,
    int tp_size
) {
    // 2 = K + V per layer; * kFp16Bytes => bytes; / min(...) TP/NUMA sharding of heads.
    return 2.0 * static_cast<double>(input_length) * head_dim * num_kv_heads
           * num_layers * kBF16Bytes * kBytesToGB
           / std::min(numa_nodes, tp_size);
}

double hidden_GB(
    int input_length,
    int hidden_size,
    int numa_nodes,
    int tp_size
) {
    // 2 = two activation tensors (e.g. residual stream convention in profile).
    return 2.0 * static_cast<double>(input_length) * hidden_size
           * std::max(tp_size / numa_nodes, 1) * kBF16Bytes * kBytesToGB;
}

double transfer_ms(double transfer_GBps, double volume_GB) {
    if (transfer_GBps <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return 1000.0 * volume_GB / transfer_GBps;
}

// Tokenwise (single GPU), classic path: total_ms = gpu_handoff = phase1 + KV xfer + phase2.
//
// Engine-overlap branch (when cpu_decode_* configured and full-input prefill < engine_init):
//   TTFT ranking uses cpu_prefill(input_length) only.
//   cpu_compute_length = input_length + floor((engine_init - prefill_full) / decode_step),
//   decode_step = intercept + coef_kv_tokens * input_length.
//   gpu_load_layers = max L with first-batch weight xfer <= engine_init (prefix layers).
// Otherwise search the classic tokenwise grid (decode coeffs ignored for TTFT).

struct TokenwiseBreakdown {
    double weights1_GB;
    double xfer_first_weights_ms;
    double cpu_prefill_ms;
    double phase1_parallel_ms;
    double kv_GB;
    double kv_transfer_ms;
    double weights2_GB;
    double xfer_remaining_weights_ms;
    int    gpu_tail_tokens;
    double gpu_tail_prefill_ms;
    double phase2_parallel_ms;
    /// Time until GPU lazy-handoff path finishes tail prefill (phase1 + KV + phase2).
    double gpu_handoff_ttft_ms;
    /// Debug: prefill + one decode step at cpu_length KV.
    double cpu_first_token_path_ms;
    double cpu_decode_one_step_ms;
    /// Remaining time in engine-prep window after CPU prefill (0 if prefill >= engine).
    double engine_slack_after_prefill_ms;
    /// floor(slack / decode_step) when modeling CPU decode during engine prep.
    int decodes_during_engine_prep;
    double total_ms;
};

TokenwiseBreakdown compute_tokenwise_breakdown(
    double transfer_GBps,
    double embedding_GB,
    double layer_GB,
    double engine_init_ms,
    int    input_length,
    int    cpu_length,
    int    gpu_load_layers,
    int    num_layers,
    int    hidden_size,
    int    num_kv_heads,
    int    head_dim,
    int    tp_size,
    int    numa_nodes,
    double cpu_tw_intercept,
    double cpu_tw_coef_tokens,
    double gpu_tw_intercept,
    double gpu_tw_coef_tokens,
    double cpu_decode_intercept,
    double cpu_decode_coef_kv
) {
    TokenwiseBreakdown b{};
    b.weights1_GB = weights_GB(embedding_GB, layer_GB, gpu_load_layers, numa_nodes, tp_size);
    b.xfer_first_weights_ms = transfer_ms(transfer_GBps, b.weights1_GB);
    b.cpu_prefill_ms =
        tokenwise_ms(cpu_tw_intercept, cpu_tw_coef_tokens, static_cast<double>(cpu_length));
    b.phase1_parallel_ms =
        std::max({b.xfer_first_weights_ms, b.cpu_prefill_ms, engine_init_ms});
    b.kv_GB = kvcache_GB(
        cpu_length, num_layers, num_kv_heads, head_dim, numa_nodes, tp_size
    );
    b.kv_transfer_ms = transfer_ms(transfer_GBps, b.kv_GB);
    int rem_layers = num_layers - gpu_load_layers;
    b.weights2_GB = weights_GB(embedding_GB, layer_GB, rem_layers, numa_nodes, tp_size);
    b.xfer_remaining_weights_ms = transfer_ms(transfer_GBps, b.weights2_GB);
    b.gpu_tail_tokens     = std::max(input_length - cpu_length, 0);
    b.gpu_tail_prefill_ms = tokenwise_ms(
        gpu_tw_intercept, gpu_tw_coef_tokens, static_cast<double>(b.gpu_tail_tokens)
    );
    b.phase2_parallel_ms = std::max(b.gpu_tail_prefill_ms, b.xfer_remaining_weights_ms);
    b.gpu_handoff_ttft_ms = b.phase1_parallel_ms + b.kv_transfer_ms + b.phase2_parallel_ms;

    b.cpu_decode_one_step_ms =
        cpu_decode_intercept
        + cpu_decode_coef_kv * static_cast<double>(std::max(cpu_length, 0));
    b.engine_slack_after_prefill_ms = 0.0;
    b.decodes_during_engine_prep  = 0;

    if (b.cpu_prefill_ms < engine_init_ms) {
        b.engine_slack_after_prefill_ms = engine_init_ms - b.cpu_prefill_ms;
        if (b.cpu_decode_one_step_ms > 1e-12) {
            b.decodes_during_engine_prep = static_cast<int>(std::floor(
                b.engine_slack_after_prefill_ms / b.cpu_decode_one_step_ms
            ));
        }
    }
    b.cpu_first_token_path_ms = b.cpu_prefill_ms + b.cpu_decode_one_step_ms;

    b.total_ms = b.gpu_handoff_ttft_ms;
    return b;
}

void log_tokenwise_breakdown(
    const TokenwiseBreakdown& b,
    int    cpu_length,
    int    gpu_load_layers,
    double engine_init_ms,
    bool   debug
) {
    if (!debug) {
        return;
    }
    std::fprintf(
        stderr,
        "\n[loading_perf_solver] tokenwise breakdown (cpu_tokens=%d gpu_preload_layers=%d)\n",
        cpu_length, gpu_load_layers
    );
    std::fprintf(
        stderr,
        "  phase1 parallel = max(first_weight_xfer_ms=%.3f, cpu_prefill_ms=%.3f, "
        "engine_init_ms=%.3f) = %.3f ms\n",
        b.xfer_first_weights_ms, b.cpu_prefill_ms, engine_init_ms, b.phase1_parallel_ms
    );
    std::fprintf(
        stderr,
        "  weights_batch1_GB=%.6f  weights_batch2_GB=%.6f\n",
        b.weights1_GB, b.weights2_GB
    );
    std::fprintf(
        stderr,
        "  kv_transfer sequential: kv_GB=%.6f  kv_xfer_ms=%.3f ms\n",
        b.kv_GB, b.kv_transfer_ms
    );
    std::fprintf(
        stderr,
        "  phase2 parallel = max(gpu_tail_prefill_ms=%.3f, remaining_weight_xfer_ms=%.3f) "
        "= %.3f ms (gpu_tail_tokens=%d)\n",
        b.gpu_tail_prefill_ms,
        b.xfer_remaining_weights_ms,
        b.phase2_parallel_ms,
        b.gpu_tail_tokens
    );
    std::fprintf(
        stderr,
        "  gpu_handoff_TTFT_ms = phase1 + kv + phase2 = %.3f + %.3f + %.3f = %.3f ms\n",
        b.phase1_parallel_ms,
        b.kv_transfer_ms,
        b.phase2_parallel_ms,
        b.gpu_handoff_ttft_ms
    );
    if (b.cpu_decode_one_step_ms > 1e-12) {
        std::fprintf(
            stderr,
            "  engine_slack_after_prefill_ms=%.3f  decode_step_ms=%.3f  "
            "decodes_during_engine_prep=%d (debug)\n",
            b.engine_slack_after_prefill_ms,
            b.cpu_decode_one_step_ms,
            b.decodes_during_engine_prep
        );
        std::fprintf(
            stderr,
            "  cpu prefill + one decode_step (debug) = %.3f + %.3f = %.3f\n",
            b.cpu_prefill_ms,
            b.cpu_decode_one_step_ms,
            b.cpu_first_token_path_ms
        );
    }
    std::fprintf(
        stderr,
        "  TTFT (classic tokenwise ranking) = gpu_handoff = %.3f ms\n\n",
        b.total_ms
    );
}

/// Debug log for engine-overlap tokenwise: TTFT ranks as CPU full-prefill only;
/// cpu_compute_tokens may include extra decode steps (KV length > input_length).
void log_tokenwise_engine_overlap_breakdown(
    int    input_length,
    int    cpu_compute_tokens,
    int    gpu_load_layers,
    double engine_init_ms,
    double cpu_prefill_full_ms,
    double cpu_decode_intercept,
    double cpu_decode_coef_kv,
    double pcie_GBps,
    double embedding_GB,
    double layer_GB,
    int    num_layers,
    int    numa_nodes,
    int    tp_size
) {
    const int k_extra_decode = cpu_compute_tokens - input_length;
    double    decode_step =
        cpu_decode_intercept
        + cpu_decode_coef_kv * static_cast<double>(input_length);
    double wgb = weights_GB(embedding_GB, layer_GB, gpu_load_layers, numa_nodes, tp_size);
    double xfer_first = transfer_ms(pcie_GBps, wgb);
    std::fprintf(
        stderr,
        "\n[loading_perf_solver] tokenwise ENGINE-OVERLAP plan "
        "(cpu_compute_tokens=%d = input_length=%d + decode_steps=%d, "
        "gpu_preload_layers=%d)\n",
        cpu_compute_tokens,
        input_length,
        k_extra_decode,
        gpu_load_layers
    );
    std::fprintf(
        stderr,
        "  ranking TTFT = CPU full prefill only = %.3f ms (not GPU handoff)\n",
        cpu_prefill_full_ms
    );
    std::fprintf(
        stderr,
        "  engine_init_ms=%.3f  cpu_prefill_full_ms=%.3f  decode_step_ms=%.3f "
        "(intercept + coef_kv*input_len)\n",
        engine_init_ms,
        cpu_prefill_full_ms,
        decode_step
    );
    std::fprintf(
        stderr,
        "  first-batch weights: gpu_load_layers=%d (indices [0, %d))  "
        "weights_GB=%.6f  xfer_ms=%.3f (must be <= engine_init for overlap preload)\n",
        gpu_load_layers,
        gpu_load_layers,
        wgb,
        xfer_first
    );
    std::fprintf(
        stderr,
        "  remaining layers on GPU after init: %d\n\n",
        num_layers - gpu_load_layers
    );
}

struct LayerwiseBreakdown {
    double weights1_GB;
    double load_first_ms;
    double cpu_layers_ms;
    double phase1_parallel_ms;
    double kv_GB;
    double act_GB;
    double kv_act_transfer_ms;
    int    gpu_compute_layers;
    double gpu_remaining_layers_ms;
    double total_ms;
};

LayerwiseBreakdown compute_layerwise_breakdown(
    double pcie_GBps,
    double embedding_GB,
    double layer_GB,
    double engine_init_ms,
    int    input_length,
    int    cpu_compute_layers,
    int    gpu_load_layers,
    int    num_layers,
    int    hidden_size,
    int    num_kv_heads,
    int    head_dim,
    int    tp_size,
    int    numa_nodes,
    double cpu_lw_intercept,
    double cpu_lw_coef_tokens,
    double gpu_lw_intercept,
    double gpu_lw_coef_tokens
) {
    LayerwiseBreakdown b{};
    b.weights1_GB = weights_GB(embedding_GB, layer_GB, gpu_load_layers, numa_nodes, tp_size);
    b.load_first_ms = transfer_ms(pcie_GBps, b.weights1_GB);
    b.cpu_layers_ms = layerwise_ms(
        cpu_lw_intercept, cpu_lw_coef_tokens,
        static_cast<double>(input_length),
        static_cast<double>(cpu_compute_layers)
    );
    b.phase1_parallel_ms = std::max({b.load_first_ms, b.cpu_layers_ms, engine_init_ms});
    b.kv_GB = kvcache_GB(
        input_length, cpu_compute_layers, num_kv_heads, head_dim, numa_nodes, tp_size
    );
    b.act_GB = hidden_GB(input_length, hidden_size, numa_nodes, tp_size);
    b.kv_act_transfer_ms = transfer_ms(pcie_GBps, b.kv_GB + b.act_GB);
    b.gpu_compute_layers = num_layers - cpu_compute_layers;
    b.gpu_remaining_layers_ms = layerwise_ms(
        gpu_lw_intercept, gpu_lw_coef_tokens,
        static_cast<double>(input_length),
        static_cast<double>(b.gpu_compute_layers)
    );
    b.total_ms = b.phase1_parallel_ms + b.kv_act_transfer_ms + b.gpu_remaining_layers_ms;
    return b;
}

void log_layerwise_breakdown(
    const LayerwiseBreakdown& b,
    int    cpu_compute_layers,
    int    gpu_preload_layers,
    double engine_init_ms,
    bool   debug
) {
    if (!debug) {
        return;
    }
    std::fprintf(
        stderr,
        "\n[loading_perf_solver] layerwise breakdown (cpu_layers=%d gpu_preload_layers=%d)\n",
        cpu_compute_layers, gpu_preload_layers
    );
    std::fprintf(
        stderr,
        "  phase1 parallel = max(load_first_ms=%.3f, cpu_layers_ms=%.3f, engine_init_ms=%.3f) "
        "= %.3f ms\n",
        b.load_first_ms, b.cpu_layers_ms, engine_init_ms, b.phase1_parallel_ms
    );
    std::fprintf(stderr, "  weights_preload_GB=%.6f\n", b.weights1_GB);
    std::fprintf(
        stderr,
        "  phase2 sequential KV+hidden xfer: kv_GB=%.6f act_GB=%.6f xfer_ms=%.3f\n",
        b.kv_GB, b.act_GB, b.kv_act_transfer_ms
    );
    std::fprintf(
        stderr,
        "  phase3 gpu remaining layers=%d  gpu_layerwise_ms=%.3f\n",
        b.gpu_compute_layers, b.gpu_remaining_layers_ms
    );
    std::fprintf(
        stderr,
        "  TTFT total = phase1 + kv_act_xfer + phase3 = %.3f + %.3f + %.3f = %.3f ms\n\n",
        b.phase1_parallel_ms,
        b.kv_act_transfer_ms,
        b.gpu_remaining_layers_ms,
        b.total_ms
    );
}

// Time to load one transformer layer's weights (no embedding), for pen_std scaling.
double load_ms_per_layer(
    double pcie_GBps,
    double layer_GB,
    int numa_nodes,
    int tp_size
) {
    return transfer_ms(pcie_GBps, weights_GB(0.0, layer_GB, 1, numa_nodes, tp_size));
}

double load_stage_weights_ms(
    double pcie_GBps,
    double embedding_GB,
    double layer_GB,
    int stage_idx,
    int ni,
    int numa_nodes,
    int tp_size
) {
    double emb = (stage_idx == 0) ? embedding_GB : 0.0;
    return transfer_ms(pcie_GBps, weights_GB(emb, layer_GB, ni, numa_nodes, tp_size));
}

// Activation / KV bubble between upstream and this PP stage (ms).
double trans_to_pp_stage_ms(
    int stage_idx,
    int nc,
    double pcie_GBps,
    double rdma_GBps,
    int input_length,
    int num_kv_heads,
    int head_dim,
    int hidden_size,
    int numa_nodes,
    int tp_size
) {
    if (stage_idx == 0) {
        if (nc <= 0) {
            return 0.0;
        }
        double vol = kvcache_GB(input_length, nc, num_kv_heads, head_dim, numa_nodes, tp_size)
                   + hidden_GB(input_length, hidden_size, numa_nodes, tp_size);
        return transfer_ms(pcie_GBps, vol);
    }
    return transfer_ms(rdma_GBps, hidden_GB(input_length, hidden_size, numa_nodes, tp_size));
}

double penalty_std_shortfall(
    int nc,
    const std::vector<int>& n_gpu,
    int num_layers,
    int pp_size,
    int target_l,
    double t_load_per_layer_ms,
    bool layer_pipeline
) {
    int off = nc;
    double total_missing = 0.0;
    for (int i = 0; i < pp_size; ++i) {
        int std_lo = i * target_l;
        int std_hi = (i == pp_size - 1) ? num_layers : (i + 1) * target_l;
        if (std_hi <= std_lo) {
            continue;
        }
        int act_lo = 0;
        int act_hi = 0;
        if (layer_pipeline && i == 0) {
            act_lo = 0;
            act_hi = nc + n_gpu[0];
            off    = act_hi;
        } else {
            act_lo = off;
            act_hi = off + n_gpu[i];
            off    = act_hi;
        }
        int overlap = std::max(0, std::min(std_hi, act_hi) - std::max(std_lo, act_lo));
        int missing_i = (std_hi - std_lo) - overlap;
        total_missing += static_cast<double>(std::max(0, missing_i));
    }
    return total_missing * t_load_per_layer_ms;
}

// Recursively enumerate monotone GPU layer counts (Ni >= 0, sum = remaining_n).
void generate_partitions_full_n_on_gpu(
    int remaining_n,
    int remaining_p,
    int prev,
    int l_min,
    int l_max,
    std::vector<int>& cur,
    std::vector<std::vector<int>>& out
) {
    if (remaining_p == 1) {
        if (remaining_n == 0) {
            if (prev < 0 || 0 >= prev) {
                cur.push_back(0);
                out.push_back(cur);
                cur.pop_back();
            }
        } else if (remaining_n >= l_min && remaining_n <= l_max) {
            if (prev < 0 || remaining_n >= prev) {
                cur.push_back(remaining_n);
                out.push_back(cur);
                cur.pop_back();
            }
        }
        return;
    }
    for (int li = 0; li <= remaining_n; ++li) {
        if (li > 0 && (li < l_min || li > l_max)) {
            continue;
        }
        if (prev >= 0 && li < prev) {
            continue;
        }
        cur.push_back(li);
        generate_partitions_full_n_on_gpu(
            remaining_n - li, remaining_p - 1, li, l_min, l_max, cur, out
        );
        cur.pop_back();
    }
}

// Layer pipeline TTFT (test_pp.py pp_ttft_layer_pipeline), analytic timings from profile.
double pp_ttft_layer_pipeline(
    int nc,
    const std::vector<int>& n_gpu,
    int pp_size,
    int num_layers,
    double engine_init_ms,
    double pcie_GBps,
    double rdma_GBps,
    double embedding_GB,
    double layer_GB,
    int input_length,
    int num_kv_heads,
    int head_dim,
    int hidden_size,
    int numa_nodes,
    int tp_size,
    double cpu_lw_intercept,
    double cpu_lw_coef_tokens,
    double gpu_lw_intercept,
    double gpu_lw_coef_tokens,
    bool debug = false
) {
    double per_layer_cpu = cpu_lw_intercept + cpu_lw_coef_tokens * input_length;
    double t_cpu         = nc * per_layer_cpu;

    if (debug) {
        std::fprintf(
            stderr,
            "[loading_perf_solver] pp_ttft_layer_pipeline: nc=%d pp_size=%d input_len=%d "
            "per_layer_cpu_ms=%.4f t_cpu_total_ms=%.3f\n",
            nc, pp_size, input_length, per_layer_cpu, t_cpu
        );
        std::fprintf(stderr, "  n_gpu per stage:");
        for (int x : n_gpu) {
            std::fprintf(stderr, " %d", x);
        }
        std::fprintf(stderr, "\n");
    }

    if (nc >= num_layers) {
        double t = std::max(engine_init_ms, t_cpu);
        if (debug) {
            std::fprintf(
                stderr,
                "  all layers on CPU: TTFT = max(engine_init=%.3f, t_cpu=%.3f) = %.3f ms\n",
                engine_init_ms, t_cpu, t
            );
        }
        return t;
    }

    double per_layer_gpu = gpu_lw_intercept + gpu_lw_coef_tokens * input_length;

    std::vector<double> ends;
    ends.reserve(pp_size);

    for (int i = 0; i < pp_size; ++i) {
        int ni = n_gpu[i];
        if (ni == 0) {
            bool all_zero_suffix = true;
            for (int j = i + 1; j < pp_size; ++j) {
                if (n_gpu[j] != 0) {
                    all_zero_suffix = false;
                    break;
                }
            }
            double end_i = 0.0;
            if (all_zero_suffix) {
                end_i = t_cpu;
                if (debug) {
                    std::fprintf(
                        stderr,
                        "  stage %d: ni=0 (idle / wait CPU) end=%.3f ms\n", i, end_i
                    );
                }
                ends.push_back(end_i);
            } else {
                end_i = ends[static_cast<std::size_t>(i - 1)]
                        + trans_to_pp_stage_ms(
                            i, nc, pcie_GBps, rdma_GBps, input_length,
                            num_kv_heads, head_dim, hidden_size, numa_nodes, tp_size
                        );
                if (debug) {
                    std::fprintf(
                        stderr,
                        "  stage %d: ni=0 bubble end=%.3f ms\n", i, end_i
                    );
                }
                ends.push_back(end_i);
            }
            continue;
        }

        double upstream = (i > 0) ? ends[static_cast<std::size_t>(i - 1)] : t_cpu;
        double trans    = trans_to_pp_stage_ms(
            i, nc, pcie_GBps, rdma_GBps, input_length,
            num_kv_heads, head_dim, hidden_size, numa_nodes, tp_size
        );
        double ready        = std::max(upstream + trans, engine_init_ms);
        double compute_done = ready + static_cast<double>(ni) * per_layer_gpu;
        double load_done    = load_stage_weights_ms(
            pcie_GBps, embedding_GB, layer_GB, i, ni, numa_nodes, tp_size
        );
        double end_i        = std::max(compute_done, load_done);
        if (debug) {
            std::fprintf(
                stderr,
                "  stage %d: ni=%d upstream=%.3f trans_ms=%.3f ready=%.3f "
                "load_done=%.3f compute_done=%.3f end=%.3f\n",
                i, ni, upstream, trans, ready, load_done, compute_done, end_i
            );
        }
        ends.push_back(end_i);
    }
    if (debug) {
        std::fprintf(
            stderr,
            "  PP layer_pipeline TTFT (last stage end) = %.3f ms\n\n",
            ends.back()
        );
    }
    return ends.back();
}

// Token-parallel CPU + PP GPUs (test_pp.py pp_ttft_token_cpu).
double pp_ttft_token_cpu(
    const std::vector<int>& n_gpu,
    int pp_size,
    int num_layers,
    int input_length,
    int s_cpu,
    double engine_init_ms,
    double pcie_GBps,
    double rdma_GBps,
    double embedding_GB,
    double layer_GB,
    int num_kv_heads,
    int head_dim,
    int hidden_size,
    int numa_nodes,
    int tp_size,
    double cpu_tw_intercept,
    double cpu_tw_coef_tokens,
    double gpu_tw_intercept,
    double gpu_tw_coef_tokens,
    bool debug = false
) {
    double s_total = static_cast<double>(std::max(input_length, 1));
    double frac_cpu = std::min(std::max(s_cpu, 0), input_length) / s_total;
    double frac_gpu = 1.0 - frac_cpu;

    double t_cpu_done = static_cast<double>(num_layers) * frac_cpu
                        * tokenwise_ms(cpu_tw_intercept, cpu_tw_coef_tokens, input_length);

    if (debug) {
        std::fprintf(
            stderr,
            "[loading_perf_solver] pp_ttft_token_cpu: s_cpu=%d input_len=%d "
            "frac_cpu=%.4f frac_gpu=%.4f t_cpu_done_ms=%.3f\n",
            s_cpu, input_length, frac_cpu, frac_gpu, t_cpu_done
        );
        std::fprintf(stderr, "  n_gpu per stage:");
        for (int x : n_gpu) {
            std::fprintf(stderr, " %d", x);
        }
        std::fprintf(stderr, "\n");
    }

    if (frac_gpu <= 0.0) {
        double t = std::max(engine_init_ms, t_cpu_done);
        if (debug) {
            std::fprintf(
                stderr,
                "  all tokens on CPU: TTFT = max(engine_init=%.3f, t_cpu_done=%.3f) = %.3f ms\n\n",
                engine_init_ms, t_cpu_done, t
            );
        }
        return t;
    }

    double per_layer_gpu_full = gpu_tw_intercept + gpu_tw_coef_tokens * input_length;

    std::vector<double> ends;
    ends.reserve(pp_size);

    for (int i = 0; i < pp_size; ++i) {
        int ni = n_gpu[i];
        if (ni == 0) {
            bool all_zero_suffix = true;
            for (int j = i + 1; j < pp_size; ++j) {
                if (n_gpu[j] != 0) {
                    all_zero_suffix = false;
                    break;
                }
            }
            double end_i = 0.0;
            if (all_zero_suffix) {
                end_i = t_cpu_done;
                if (debug) {
                    std::fprintf(
                        stderr,
                        "  stage %d: ni=0 (wait CPU) end=%.3f ms\n", i, end_i
                    );
                }
                ends.push_back(end_i);
            } else {
                end_i = ends[static_cast<std::size_t>(i - 1)]
                        + trans_to_pp_stage_ms(
                            i, 0, pcie_GBps, rdma_GBps, input_length,
                            num_kv_heads, head_dim, hidden_size, numa_nodes, tp_size
                        );
                if (debug) {
                    std::fprintf(
                        stderr,
                        "  stage %d: ni=0 bubble end=%.3f ms\n", i, end_i
                    );
                }
                ends.push_back(end_i);
            }
            continue;
        }

        double upstream = (i > 0) ? ends[static_cast<std::size_t>(i - 1)] : t_cpu_done;
        double trans    = trans_to_pp_stage_ms(
            i, 0, pcie_GBps, rdma_GBps, input_length,
            num_kv_heads, head_dim, hidden_size, numa_nodes, tp_size
        );
        double ready        = std::max(upstream + trans, engine_init_ms);
        double compute_done = ready + static_cast<double>(ni) * per_layer_gpu_full * frac_gpu;
        double load_done    = load_stage_weights_ms(
            pcie_GBps, embedding_GB, layer_GB, i, ni, numa_nodes, tp_size
        );
        double end_i        = std::max(compute_done, load_done);
        if (debug) {
            std::fprintf(
                stderr,
                "  stage %d: ni=%d upstream=%.3f trans_ms=%.3f ready=%.3f "
                "load_done=%.3f compute_done=%.3f (ni*per_gpu*frac_gpu) end=%.3f\n",
                i, ni, upstream, trans, ready, load_done, compute_done, end_i
            );
        }
        ends.push_back(end_i);
    }
    if (debug) {
        std::fprintf(
            stderr,
            "  PP token_cpu TTFT (last stage end) = %.3f ms\n\n",
            ends.back()
        );
    }
    return ends.back();
}

double ttft_tokenwise(
    double transfer_GBps,
    double embedding_GB,
    double layer_GB,
    double engine_init_ms,
    int input_length,
    int cpu_length,
    int gpu_load_layers,
    int num_layers,
    int hidden_size,
    int num_kv_heads,
    int head_dim,
    int tp_size,
    int numa_nodes,
    double cpu_tw_intercept,
    double cpu_tw_coef_tokens,
    double gpu_tw_intercept,
    double gpu_tw_coef_tokens,
    double cpu_decode_intercept,
    double cpu_decode_coef_kv
) {
    return compute_tokenwise_breakdown(
               transfer_GBps, embedding_GB, layer_GB, engine_init_ms, input_length,
               cpu_length, gpu_load_layers, num_layers, hidden_size, num_kv_heads,
               head_dim, tp_size, numa_nodes, cpu_tw_intercept, cpu_tw_coef_tokens,
               gpu_tw_intercept, gpu_tw_coef_tokens, cpu_decode_intercept,
               cpu_decode_coef_kv
    )
        .total_ms;
}

// Layerwise TTFT (single GPU, non-PP).
double ttft_layerwise(
    double pcie_GBps,
    double embedding_GB,
    double layer_GB,
    double engine_init_ms,
    int input_length,
    int cpu_compute_layers,
    int gpu_load_layers,
    int num_layers,
    int hidden_size,
    int num_kv_heads,
    int head_dim,
    int tp_size,
    int numa_nodes,
    double cpu_lw_intercept,
    double cpu_lw_coef_tokens,
    double gpu_lw_intercept,
    double gpu_lw_coef_tokens
) {
    return compute_layerwise_breakdown(
               pcie_GBps, embedding_GB, layer_GB, engine_init_ms, input_length,
               cpu_compute_layers, gpu_load_layers, num_layers, hidden_size,
               num_kv_heads, head_dim, tp_size, numa_nodes, cpu_lw_intercept,
               cpu_lw_coef_tokens, gpu_lw_intercept, gpu_lw_coef_tokens
    )
        .total_ms;
}

py::object solve_pp_lazy_load(
    double fixed_ms,
    double pcie_GBps,
    double rdma_GBps,
    int numa_nodes,
    double embedding_GB,
    double layer_GB,
    int num_layers,
    int num_kv_heads,
    int head_dim,
    int hidden_size,
    int tp_size,
    int pp_size,
    int input_length,
    double cpu_tw_intercept,
    double cpu_tw_coef_tokens,
    double gpu_tw_intercept,
    double gpu_tw_coef_tokens,
    double cpu_lw_intercept,
    double cpu_lw_coef_tokens,
    double gpu_lw_intercept,
    double gpu_lw_coef_tokens,
    double cpu_decode_intercept,
    double cpu_decode_coef_kv,
    bool debug
) {
    if (pp_size <= 1 || num_layers < 1) {
        return py::none();
    }
    // PP: CPU-first-decode shortcut not modeled here (single-GPU tokenwise only).

    const int max_diff = 2;
    int target_l       = std::max(1, num_layers / pp_size);
    int l_min          = std::max(1, target_l - max_diff);
    int l_max          = target_l + max_diff;

    double t_load_layer = load_ms_per_layer(pcie_GBps, layer_GB, numa_nodes, tp_size);

    std::vector<std::vector<int>> gpu_partitions;
    std::vector<int> cur;
    generate_partitions_full_n_on_gpu(
        num_layers, pp_size, -1, l_min, l_max, cur, gpu_partitions
    );

    // --- Scenario A: layer pipeline (Nc, N0..N_{P-1}), Nc + sum Ni = N
    double best_a_score = std::numeric_limits<double>::infinity();
    int best_nc         = 0;
    std::vector<int> best_n_gpu_a(pp_size, 0);

    for (int nc = 0; nc <= num_layers; ++nc) {
        int threshold = num_layers / pp_size - 5;
        if (threshold > 0 && nc >= threshold) {
            continue;
        }
        int r = num_layers - nc;
        std::vector<std::vector<int>> parts;
        std::vector<int> c2;
        generate_partitions_full_n_on_gpu(r, pp_size, -1, l_min, l_max, c2, parts);

        for (const auto& n_gpu : parts) {
            double ttft = pp_ttft_layer_pipeline(
                nc, n_gpu, pp_size, num_layers, fixed_ms, pcie_GBps, rdma_GBps,
                embedding_GB, layer_GB, input_length, num_kv_heads, head_dim,
                hidden_size, numa_nodes, tp_size,
                cpu_lw_intercept, cpu_lw_coef_tokens,
                gpu_lw_intercept, gpu_lw_coef_tokens
            );
            double pen = penalty_std_shortfall(
                nc, n_gpu, num_layers, pp_size, target_l, t_load_layer, true
            );
            double sc = ttft + kBetaStd * pen;
            if (sc < best_a_score) {
                best_a_score = sc;
                best_nc      = nc;
                best_n_gpu_a = n_gpu;
            }
        }
    }

    // --- Scenario B: token split + same GPU partitions as full N
    double best_b_score = std::numeric_limits<double>::infinity();
    int best_s_cpu      = 0;
    std::vector<int> best_n_gpu_b(pp_size, 0);

    for (int s_cpu = 0; s_cpu <= input_length; s_cpu += kTokenCpuStep) {
        for (const auto& n_gpu : gpu_partitions) {
            double ttft = pp_ttft_token_cpu(
                n_gpu, pp_size, num_layers, input_length, s_cpu, fixed_ms,
                pcie_GBps, rdma_GBps, embedding_GB, layer_GB,
                num_kv_heads, head_dim, hidden_size, numa_nodes, tp_size,
                cpu_tw_intercept, cpu_tw_coef_tokens,
                gpu_tw_intercept, gpu_tw_coef_tokens
            );
            double pen = penalty_std_shortfall(
                0, n_gpu, num_layers, pp_size, target_l, t_load_layer, false
            );
            double sc = ttft + kBetaStd * pen;
            if (sc < best_b_score) {
                best_b_score = sc;
                best_s_cpu   = s_cpu;
                best_n_gpu_b = n_gpu;
            }
        }
    }

    bool use_layer = best_a_score <= best_b_score;
    int second     = use_layer ? best_nc : best_s_cpu;
    const std::vector<int>& n_gpu = use_layer ? best_n_gpu_a : best_n_gpu_b;

    py::list stages;
    int g = use_layer ? best_nc : 0;
    for (int i = 0; i < pp_size; ++i) {
        py::list stage_layers;
        for (int j = 0; j < n_gpu[i]; ++j) {
            stage_layers.append(g++);
        }
        stages.append(stage_layers);
    }

    std::string method = use_layer ? "layerwise" : "tokenwise";

    if (debug) {
        std::fprintf(stderr, "\n[loading_perf_solver] ========== PP chosen plan ==========\n");
        std::fprintf(
            stderr,
            "  method=%s second=%d score_layerwise=%.3f score_tokenwise=%.3f (incl. %.2f*pen_std)\n",
            method.c_str(), second, best_a_score, best_b_score, kBetaStd
        );
        if (use_layer) {
            pp_ttft_layer_pipeline(
                second, n_gpu, pp_size, num_layers, fixed_ms, pcie_GBps, rdma_GBps,
                embedding_GB, layer_GB, input_length, num_kv_heads, head_dim,
                hidden_size, numa_nodes, tp_size, cpu_lw_intercept, cpu_lw_coef_tokens,
                gpu_lw_intercept, gpu_lw_coef_tokens, true
            );
        } else {
            pp_ttft_token_cpu(
                n_gpu, pp_size, num_layers, input_length, second, fixed_ms,
                pcie_GBps, rdma_GBps, embedding_GB, layer_GB, num_kv_heads, head_dim,
                hidden_size, numa_nodes, tp_size, cpu_tw_intercept, cpu_tw_coef_tokens,
                gpu_tw_intercept, gpu_tw_coef_tokens, true
            );
        }
    }

    return py::make_tuple(method, second, stages);
}

}  // namespace

py::object solve_lazy_load_method_cpp(
    double fixed_ms,
    double pcie_GBps,
    double rdma_GBps,
    int numa_nodes,
    double embedding_GB,
    double layer_GB,
    int num_layers,
    int num_kv_heads,
    int head_dim,
    int hidden_size,
    int tp_size,
    int pp_size,
    int input_length,
    double cpu_tw_intercept,
    double cpu_tw_coef_tokens,
    double gpu_tw_intercept,
    double gpu_tw_coef_tokens,
    double cpu_lw_intercept,
    double cpu_lw_coef_tokens,
    double gpu_lw_intercept,
    double gpu_lw_coef_tokens,
    double cpu_decode_intercept,
    double cpu_decode_coef_kv,
    bool debug = false
) {
    if (pp_size > 1) {
        return solve_pp_lazy_load(
            fixed_ms, pcie_GBps, rdma_GBps, numa_nodes, embedding_GB, layer_GB,
            num_layers, num_kv_heads, head_dim, hidden_size, tp_size, pp_size,
            input_length, cpu_tw_intercept, cpu_tw_coef_tokens,
            gpu_tw_intercept, gpu_tw_coef_tokens, cpu_lw_intercept, cpu_lw_coef_tokens,
            gpu_lw_intercept, gpu_lw_coef_tokens, cpu_decode_intercept,
            cpu_decode_coef_kv, debug
        );
    }

    double min_tokenwise_ttft = std::numeric_limits<double>::infinity();
    int best_tokenwise_cpu_compute_length = 0;
    int best_tokenwise_gpu_load_layers    = 0;
    bool best_tokenwise_is_engine_overlap = false;

    auto consider_classic_tokenwise = [&](int cc, int gl) {
        double t = ttft_tokenwise(
            pcie_GBps, embedding_GB, layer_GB, fixed_ms, input_length, cc, gl,
            num_layers, hidden_size, num_kv_heads, head_dim, tp_size, numa_nodes,
            cpu_tw_intercept, cpu_tw_coef_tokens, gpu_tw_intercept, gpu_tw_coef_tokens,
            0.0, 0.0
        );
        if (t < min_tokenwise_ttft - 1e-9) {
            min_tokenwise_ttft                 = t;
            best_tokenwise_cpu_compute_length = cc;
            best_tokenwise_gpu_load_layers    = gl;
            best_tokenwise_is_engine_overlap  = false;
        } else if (std::abs(t - min_tokenwise_ttft) <= 1e-9
                   && gl > best_tokenwise_gpu_load_layers) {
            best_tokenwise_cpu_compute_length = cc;
            best_tokenwise_gpu_load_layers    = gl;
            best_tokenwise_is_engine_overlap  = false;
        }
    };

    if (input_length > 0) {
        const int step = 64;
        int c0         = std::min(32, input_length);
        for (int cc = c0; cc <= input_length; cc += step) {
            for (int gl = 1; gl <= num_layers; ++gl) {
                consider_classic_tokenwise(cc, gl);
            }
        }
        if (input_length > 32 && (input_length - 32) % step != 0) {
            for (int gl = 1; gl <= num_layers; ++gl) {
                consider_classic_tokenwise(input_length, gl);
            }
        }
    }

    const double cpu_prefill_full = tokenwise_ms(
        cpu_tw_intercept, cpu_tw_coef_tokens, static_cast<double>(input_length)
    );
    const bool decode_profile_ok =
        (cpu_decode_intercept != 0.0 || cpu_decode_coef_kv != 0.0);
    if (decode_profile_ok && input_length > 0 && cpu_prefill_full + 1e-9 < fixed_ms) {
        double decode_step = cpu_decode_intercept + cpu_decode_coef_kv;
        if (decode_step <= 0.0) {
            decode_step = 1e-6;
        }
        int k = static_cast<int>(
            std::floor((fixed_ms - cpu_prefill_full) / decode_step)
        );
        if (k < 0) {
            k = 0;
        }
        const int cpu_len_ov = input_length + k;
        int       gpu_load_ov = 0;
        for (int L = num_layers; L >= 1; --L) {
            double wgb = weights_GB(embedding_GB, layer_GB, L, numa_nodes, tp_size);
            if (transfer_ms(pcie_GBps, wgb) <= fixed_ms + 1e-9) {
                gpu_load_ov = L;
                break;
            }
        }
        if (gpu_load_ov > 0) {
            const double ttft_ov = cpu_prefill_full;
            if (ttft_ov < min_tokenwise_ttft - 1e-9) {
                min_tokenwise_ttft                 = ttft_ov;
                best_tokenwise_cpu_compute_length = cpu_len_ov;
                best_tokenwise_gpu_load_layers    = gpu_load_ov;
                best_tokenwise_is_engine_overlap  = true;
            } else if (std::abs(ttft_ov - min_tokenwise_ttft) <= 1e-9
                       && gpu_load_ov > best_tokenwise_gpu_load_layers) {
                best_tokenwise_cpu_compute_length = cpu_len_ov;
                best_tokenwise_gpu_load_layers    = gpu_load_ov;
                best_tokenwise_is_engine_overlap  = true;
            }
        }
    }

    py::list stages;

    py::list best_tokenwise_layers;
    for (int i = 0; i < best_tokenwise_gpu_load_layers; ++i) {
        best_tokenwise_layers.append(i);
    }
    stages.append(best_tokenwise_layers);

    double min_layerwise_ttft = std::numeric_limits<double>::infinity();
    int best_layerwise_cpu_compute_layers = 0;
    int best_layerwise_gpu_load_layers    = 0;

    for (int gpu_load_layers = 1; gpu_load_layers <= num_layers; ++gpu_load_layers) {
        for (int cpu_compute_layers = 1; cpu_compute_layers <= num_layers;
             ++cpu_compute_layers) {
            if (gpu_load_layers < num_layers - cpu_compute_layers) {
                continue;
            }
            double current_ttft = ttft_layerwise(
                pcie_GBps, embedding_GB, layer_GB, fixed_ms, input_length,
                cpu_compute_layers, gpu_load_layers, num_layers, hidden_size,
                num_kv_heads, head_dim, tp_size, numa_nodes,
                cpu_lw_intercept, cpu_lw_coef_tokens,
                gpu_lw_intercept, gpu_lw_coef_tokens
            );
            if (current_ttft < min_layerwise_ttft) {
                min_layerwise_ttft                 = current_ttft;
                best_layerwise_cpu_compute_layers = cpu_compute_layers;
                best_layerwise_gpu_load_layers    = gpu_load_layers;
            }
            else if (current_ttft == min_layerwise_ttft && gpu_load_layers > best_layerwise_gpu_load_layers) {
                best_layerwise_gpu_load_layers = gpu_load_layers;
            }
        }
    }

    py::list best_layerwise_layers;
    int start_idx = num_layers - best_layerwise_gpu_load_layers;
    for (int i = 0; i < best_layerwise_gpu_load_layers; ++i) {
        best_layerwise_layers.append(start_idx + i);
    }
    stages.append(best_layerwise_layers);

    if (!std::isfinite(min_tokenwise_ttft) && !std::isfinite(min_layerwise_ttft)) {
        return py::none();
    }

    if (debug) {
        std::fprintf(
            stderr,
            "\n[loading_perf_solver] ========== single-GPU solve (pp_size=1) ==========\n"
        );
        std::fprintf(
            stderr,
            "  input_length=%d num_layers=%d pcie_GBps=%.3f engine_init_ms=%.3f\n",
            input_length, num_layers, pcie_GBps, fixed_ms
        );
        if (std::isfinite(min_tokenwise_ttft)) {
            if (best_tokenwise_is_engine_overlap) {
                const double cpu_pf_full = tokenwise_ms(
                    cpu_tw_intercept,
                    cpu_tw_coef_tokens,
                    static_cast<double>(input_length)
                );
                log_tokenwise_engine_overlap_breakdown(
                    input_length,
                    best_tokenwise_cpu_compute_length,
                    best_tokenwise_gpu_load_layers,
                    fixed_ms,
                    cpu_pf_full,
                    cpu_decode_intercept,
                    cpu_decode_coef_kv,
                    pcie_GBps,
                    embedding_GB,
                    layer_GB,
                    num_layers,
                    numa_nodes,
                    tp_size
                );
            } else {
                TokenwiseBreakdown tb = compute_tokenwise_breakdown(
                    pcie_GBps, embedding_GB, layer_GB, fixed_ms, input_length,
                    best_tokenwise_cpu_compute_length, best_tokenwise_gpu_load_layers,
                    num_layers, hidden_size, num_kv_heads, head_dim, tp_size, numa_nodes,
                    cpu_tw_intercept, cpu_tw_coef_tokens, gpu_tw_intercept, gpu_tw_coef_tokens,
                    cpu_decode_intercept, cpu_decode_coef_kv
                );
                log_tokenwise_breakdown(
                    tb, best_tokenwise_cpu_compute_length, best_tokenwise_gpu_load_layers,
                    fixed_ms, true
                );
            }
        }
        if (std::isfinite(min_layerwise_ttft)) {
            LayerwiseBreakdown lb = compute_layerwise_breakdown(
                pcie_GBps, embedding_GB, layer_GB, fixed_ms, input_length,
                best_layerwise_cpu_compute_layers, best_layerwise_gpu_load_layers,
                num_layers, hidden_size, num_kv_heads, head_dim, tp_size, numa_nodes,
                cpu_lw_intercept, cpu_lw_coef_tokens, gpu_lw_intercept, gpu_lw_coef_tokens
            );
            log_layerwise_breakdown(
                lb, best_layerwise_cpu_compute_layers, best_layerwise_gpu_load_layers,
                fixed_ms, true
            );
        }
        std::fprintf(
            stderr,
            "[loading_perf_solver] compare: ttft_tokenwise=%.3f ms  ttft_layerwise=%.3f ms  "
            "-> winner: %s\n\n",
            min_tokenwise_ttft,
            min_layerwise_ttft,
            min_tokenwise_ttft < min_layerwise_ttft ? "tokenwise" : "layerwise"
        );
    }

    if (min_tokenwise_ttft < min_layerwise_ttft) {
        return py::make_tuple(
            std::string("tokenwise"),
            best_tokenwise_cpu_compute_length,
            best_tokenwise_layers,
            min_tokenwise_ttft
        );
    }
    return py::make_tuple(
        std::string("layerwise"),
        best_layerwise_cpu_compute_layers,
        best_layerwise_layers,
        min_layerwise_ttft
    );
}

PYBIND11_MODULE(_loading_perf_profile_solver, m) {
    m.doc() = "Pybind11 lazy-load performance solver";
    m.def(
        "solve_lazy_load_method_cpp",
        &solve_lazy_load_method_cpp,
        py::arg("fixed_ms"),
        py::arg("pcie_GBps"),
        py::arg("rdma_GBps"),
        py::arg("numa_nodes"),
        py::arg("embedding_GB"),
        py::arg("layer_GB"),
        py::arg("num_layers"),
        py::arg("num_kv_heads"),
        py::arg("head_dim"),
        py::arg("hidden_size"),
        py::arg("tp_size"),
        py::arg("pp_size"),
        py::arg("input_length"),
        py::arg("cpu_tw_intercept"),
        py::arg("cpu_tw_coef_tokens"),
        py::arg("gpu_tw_intercept"),
        py::arg("gpu_tw_coef_tokens"),
        py::arg("cpu_lw_intercept"),
        py::arg("cpu_lw_coef_tokens"),
        py::arg("gpu_lw_intercept"),
        py::arg("gpu_lw_coef_tokens"),
        py::arg("cpu_decode_intercept"),
        py::arg("cpu_decode_coef_kv"),
        py::arg("debug") = false
    );
}
