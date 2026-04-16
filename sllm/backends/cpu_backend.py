# ---------------------------------------------------------------------------- #
#  vLLM on CPU: ShmConnector kv_producer, SHM layout probe, layerwise prefill   #
# ---------------------------------------------------------------------------- #
import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from vllm import AsyncLLMEngine, RequestOutput, SamplingParams

from sllm.backends.backend_utils import (
    BackendStatus,
    LLMEngineStatusDict,
    SllmBackend,
    async_engine_args_from_dict,
    build_core_filtered_engine_config,
    parse_vllm_generate_request,
    process_output,
)

logger = logging.getLogger("ray")


class CPUBackend(SllmBackend):
    """vLLM ``AsyncLLMEngine`` on CPU: KV producer for shared-memory handoff to GPU."""

    _vllm_device_log_label = "CPU"

    def __init__(
        self,
        instance_id: str,
        model: str,
        device: str,
        backend_config: Optional[Dict[str, Any]] = None,
        runtime_env: Optional[Dict[str, Any]] = None,
    ) -> None:
        if backend_config is None:
            raise ValueError("Backend config is missing")
        if device != "cpu":
            raise ValueError(f"CPUBackend expects device 'cpu', got {device!r}")

        self.status = BackendStatus.UNINITIALIZED
        self.status_lock = asyncio.Lock()
        self.instance_id = instance_id
        self.model_name = model
        self.backend_config = backend_config
        self.runtime_env = runtime_env
        self.request_trace = LLMEngineStatusDict()
        self.trace_debug = backend_config.get("trace_debug", False)
        self.load_method = backend_config.get("load_method", "tokenwise")

        filtered = build_core_filtered_engine_config(model, backend_config)
        filtered["tensor_parallel_size"] = 2
        filtered["max_model_len"] = 5120
        filtered["max_num_batched_tokens"] = 4096
        filtered["enforce_eager"] = True
        filtered["enable_prefix_caching"] = False
        filtered["task"] = "auto"
        filtered["dtype"] = "bfloat16"
        filtered["load_method"] = "tokenwise"

        if filtered["load_format"] == "shm":
            filtered["kv_transfer_config"] = {
                "kv_connector": "ShmConnector",
                "kv_role": "kv_producer",
                "kv_rank": 0,
            }

        self.engine_args = async_engine_args_from_dict(filtered)
        self.engine: Optional[AsyncLLMEngine] = None

    async def init_backend(self) -> None:
        os.environ["VLLM_CPU_OMP_THREADS_BIND"] = "66-94|98-126"
        os.environ["VLLM_SLEEP_WHEN_IDLE"] = "1"

        logger.info("Initializing vLLM CPU backend...")
        async with self.status_lock:
            if self.status != BackendStatus.UNINITIALIZED:
                logger.warning("vLLM CPU backend already initialized")
                return
            start_time = time.time()
            self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
            load_time = time.time() - start_time
            logger.info("vLLM CPU backend initialized in %.2f ms", load_time * 1000)
            self.status = BackendStatus.RUNNING

    async def generate(self, request_data: Dict[str, Any]):
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Engine is not running"}

        assert self.engine is not None

        if request_data is None:
            return {"error": "Request data is missing"}

        try:
            model_name, inputs, request_id, sampling_kw = parse_vllm_generate_request(
                request_data, self.model_name
            )
            sampling_params = SamplingParams(**sampling_kw)
        except Exception as e:
            return {"error": f"Invalid request or sampling parameters: {e}"}

        results_generator = self.engine.generate(
            inputs, sampling_params, request_id
        )

        latency_metrics: Dict[str, Any] = {}
        start_time = time.perf_counter()
        final_output = None
        first_chunk_time = None
        itl_token_count = 0
        ttft = 0.0
        itl_list: List[float] = []
        most_recent_timestamp = start_time
        async for response_output in results_generator:
            final_output = response_output
            await self.request_trace.update_status(request_id, response_output)
            if response_output.outputs:
                current_time = time.perf_counter()
                if first_chunk_time is None:
                    first_chunk_time = current_time
                    ttft = first_chunk_time - start_time
                else:
                    itl_token_count += 1
                    itl_list.append(current_time - most_recent_timestamp)
                most_recent_timestamp = current_time
        end_time = time.perf_counter()
        assert final_output is not None
        latency_metrics["e2e"] = end_time - start_time
        latency_metrics["ttft"] = ttft
        latency_metrics["first_token_time"] = first_chunk_time
        latency_metrics["tpot"] = (
            (end_time - first_chunk_time) / itl_token_count
            if itl_token_count > 0
            else 0.0
        )
        latency_metrics["itls"] = itl_list if itl_token_count > 0 else []
        latency_metrics["output_length"] = len(final_output.outputs[0].token_ids)

        if not self.trace_debug:
            await self.request_trace.delete_request(request_id)
        return process_output(final_output, latency_metrics, model_name)

    async def update_computing_layers(self, computing_layers: int):
        if self.load_method != "layerwise" or self.engine is None:
            return
        if self.engine is None:
            return
        await self.engine.update_computing_layers(computing_layers)

    async def get_shm_kv_cache_info(self) -> Optional[Tuple[int, int, int, int]]:
        if self.engine is None:
            return None
        return await self.engine.get_shm_kv_cache_info()
