# ---------------------------------------------------------------------------- #
#  vLLM on GPU: ShmConnector kv_consumer, lazy load, scheduler / router hooks    #
# ---------------------------------------------------------------------------- #
import asyncio
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import ray
from vllm import AsyncLLMEngine, RequestOutput, SamplingParams

from sllm.backends.backend_utils import (
    BackendStatus,
    LLMEngineStatusDict,
    SllmBackend,
    async_engine_args_from_dict,
    build_core_filtered_engine_config,
    parse_vllm_generate_request,
    process_output,
    read_numa_cpu_affinity,
)

logger = logging.getLogger("ray")


class GPUBackend(SllmBackend):
    """vLLM ``AsyncLLMEngine`` on GPU: KV consumer, lazy weights, NUMA affinity."""

    _vllm_device_log_label = "GPU"

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
        if device != "gpu":
            raise ValueError(f"GPUBackend expects device 'gpu', got {device!r}")

        self.status = BackendStatus.UNINITIALIZED
        self.status_lock = asyncio.Lock()
        self.instance_id = instance_id
        self.model_name = model
        self.backend_config = backend_config
        self.runtime_env = runtime_env
        self.request_trace = LLMEngineStatusDict()
        self.trace_debug = backend_config.get("trace_debug", False)
        self.load_method = backend_config.get("load_method", "tokenwise")
        self.lazy_load = backend_config.get("lazy_load", True)

        filtered = build_core_filtered_engine_config(model, backend_config)
        filtered["enforce_eager"] = True
        filtered["enable_prefix_caching"] = False
        filtered["dtype"] = "bfloat16"
        filtered["load_method"] = "tokenwise"
        
        if filtered["load_format"] == "shm":
            filtered["shm_tp_size"] = backend_config.get("shm_tp_size", 2)
            filtered["shm_kv_cache_size"] = backend_config.get("shm_kv_cache_size", 4293918720)
            filtered["shm_num_blocks"] = backend_config.get("shm_num_blocks", 455)
            filtered["shm_block_len"] = backend_config.get("shm_block_len", 131072)
            filtered["kv_transfer_config"] = {
                "kv_connector": "ShmConnector",
                "kv_role": "kv_consumer",
                "kv_rank": 1,
            }
            self.lazy_load = bool(filtered.get("lazy_load", False))

        self.engine_args = async_engine_args_from_dict(filtered)
        self.engine: Optional[AsyncLLMEngine] = None
        self.weights_loaded = False

        self.gpu_router = ray.get_actor(self.model_name, namespace="gpu_models")
        self.scheduler = ray.get_actor("model_loading_scheduler")

    async def lazy_load_weights(
        self, layer_idxes: list[list[int]], request_id: str = None
    ):
        start_time = time.time()
        await self.engine.lazy_init(
            layer_idxes=layer_idxes, no_warmup=False, request_id=request_id
        )
        logger.info(
            "lazy_load_weights done instance_id=%s elapsed_s=%.3f",
            self.instance_id,
            time.time() - start_time,
        )
        self.weights_loaded = True
        return True

    async def init_backend(self) -> None:
        if self.runtime_env is not None:
            os.environ.update(self.runtime_env.get("env_vars", {}))
        local_gpu_numa = self.backend_config.get("local_gpu_numa_node")
        if local_gpu_numa is None:
            local_gpu_numa = os.environ.get("SLLM_LOCAL_NUMA_NODE")
        if local_gpu_numa is not None:
            numa_cpus = read_numa_cpu_affinity(int(local_gpu_numa))
            if numa_cpus:
                os.sched_setaffinity(0, set(numa_cpus))

        os.environ["VLLM_SLEEP_WHEN_IDLE"] = "1"

        logger.info("Initializing vLLM GPU backend...")
        async with self.status_lock:
            if self.status != BackendStatus.UNINITIALIZED:
                logger.warning("vLLM GPU backend already initialized")
                return
            start_time = time.time()
            if self.lazy_load:
                logger.info("init engine except load model weigths")
            self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
            load_time = time.time() - start_time
            logger.info("vLLM GPU backend initialized in %.2f ms", load_time * 1000)
            self.status = BackendStatus.RUNNING

    async def get_visible_devices(self) -> List[int]:
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible_devices:
            import torch

            return list(range(torch.cuda.device_count()))
        return [int(x) for x in visible_devices.split(",") if x.strip()]

    async def set_shm_kv_cache_info(self, shm_kv_cache_info: tuple[int, int, int, int]):
        if self.engine is None:
            return
        await self.engine.set_shm_kv_cache_info(
            shm_kv_cache_size=shm_kv_cache_info[0],
            shm_num_blocks=shm_kv_cache_info[1],
            shm_block_len=shm_kv_cache_info[2],
            shm_tp_size=shm_kv_cache_info[3],
        )

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
        final_output: Optional[RequestOutput] = None
        first_chunk_time: Optional[float] = None
        ttft: float = 0.0
        itl_token_count = 0
        itl_list: List[float] = []
        most_recent_timestamp = start_time
        async for response_output in results_generator:
            final_output = response_output
            if getattr(response_output, "load_weights_finished", False):
                self.gpu_router.notify_weights_loaded.remote(self.instance_id)
            await self.request_trace.update_status(request_id, response_output)
            if response_output.outputs:
                current_time = time.perf_counter()
                if first_chunk_time is None:
                    first_chunk_time = current_time
                    ttft = first_chunk_time - start_time
                    try:
                        self.scheduler.notify_first_token_by_instance.remote(
                            self.instance_id, self.model_name
                        )
                    except Exception:
                        pass
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
            if itl_token_count > 0 and first_chunk_time is not None
            else 0.0
        )
        latency_metrics["itls"] = itl_list if itl_token_count > 0 else []
        latency_metrics["output_length"] = len(final_output.outputs[0].token_ids)

        if not self.trace_debug:
            await self.request_trace.delete_request(request_id)
        return process_output(final_output, latency_metrics, model_name)
