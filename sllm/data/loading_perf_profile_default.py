# ---------------------------------------------------------------------------- #
#  Bundled default when no profile file is set.                                #
#  Override: set env SLLM_LOADING_PERF_PROFILE to a .json or .py file whose    #
#  root / LOADING_PERF_PROFILE is the full dict (no merge with this file).     #
# ---------------------------------------------------------------------------- #

LOADING_PERF_PROFILE = {
    "hardware": {
        # Effective CPU host -> GPU device memory bandwidth (GB/s) for weight DMA.
        "pcie_cpu_to_gpu_GBps": 23,
        # Typical NIC / IB line rate between nodes when weights move over RDMA.
        "rdma_node_to_node_GBps": 25,
        "numa_nodes": 2
    },
    # Per logical model id (HuggingFace name or short id). Missing keys fall back to "default".
    "models": {
        "Qwen3-8B": {
            "engine_init_ms": 750,
            "layer_GB": 0.3861111,
            "embedding_GB": 1.25,
            "num_layers": 36,
            "num_kv_heads": 8,
            "head_dim": 128,
            "hidden_size": 4096,
            "tp_size": 1,
            "pp_size": 1,
            # CPU prefill time for tokenwise path: ms = intercept + coef_tokens * T
            "cpu_tokenwise_ms": {
                "intercept": 140,
                "coef_tokens": 0.51,
            },
            "gpu_tokenwise_ms": {
                "intercept": 17,
                "coef_tokens": 0.075,
            },
            # Optional: CPU decode during engine prep when prefill < engine_init.
            # decode_step_ms = intercept + coef_kv_tokens * cpu_prefill_len;
            # n_decode_in_window = floor((engine_init - prefill) / decode_step_ms).
            # Search requires first weight xfer <= engine_init when the above applies.
            "cpu_decode_ms": {
                "intercept": 85,
                "coef_kv_tokens": 1,
            },
            # CPU time for layerwise path: ms = intercept + coef_tokens*T + coef_layers*L + coef_tokens_layers*T*L
            "cpu_layerwise_ms": {
                "intercept": 4,
                "coef_tokens": 0.013,
            },
            # Optional: GPU prefill term for ranking (same shape as cpu_layerwise_ms).
            "gpu_layerwise_ms": {
                "intercept": 0.4,
                "coef_tokens": 0.002,
            },
        },
    },
}
