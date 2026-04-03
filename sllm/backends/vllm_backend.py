# ---------------------------------------------------------------------------- #
#  Distributed Inference - CPU Backend                                        #
#  CPU backend that loads weights and starts prefill inference                #
# ---------------------------------------------------------------------------- #
import asyncio
import logging
import os
import time
import uuid
from dataclasses import fields
from typing import Any, Dict, List, Optional, Union

import ray


from vllm import (
    AsyncEngineArgs,
    AsyncLLMEngine,
    RequestOutput,
    SamplingParams,
)
from vllm.inputs import TokensPrompt

from sllm.backends.backend_utils import (
    BackendStatus,
    SllmBackend,
)

logger = logging.getLogger("ray")

def process_output(output: RequestOutput, latency_metrics: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    choices: List[Dict[str, Any]] = [
        {
            "index": idx,
            "text": result.text,
            "logprobs": result.logprobs,
            "finish_reason": result.finish_reason,
        }
        for idx, result in enumerate(output.outputs)
    ]

    api_response = {
        "id": output.request_id,
        "object": "chat.completion",
        "created": (
            int(time.time())
            if output.metrics is None
            else output.metrics.arrival_time
        ),
        "model": model_name,
        "choices": choices,
        "usage": {
            "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": sum(
                len(result.token_ids) for result in output.outputs
            ),
            "total_tokens": len(output.prompt_token_ids)
            + sum(len(result.token_ids) for result in output.outputs),
        },
    }

    return api_response, latency_metrics


class LLMEngineStatusDict:
    def __init__(self):
        self.status_dict: Dict[str, Union[RequestOutput, str]] = {}
        self.lock = asyncio.Lock()

    async def update_status(
        self, request_id: str, request_output: Union[RequestOutput, str]
    ):
        async with self.lock:
            self.status_dict[request_id] = request_output

    async def delete_request(self, request_id: str):
        async with self.lock:
            del self.status_dict[request_id]

    async def return_all_results(self) -> List[Union[RequestOutput, str]]:
        async with self.lock:
            return list(self.status_dict.values())

    async def return_all_request_ids(self) -> List[str]:
        async with self.lock:
            return list(self.status_dict.keys())

    async def request_count(self) -> int:
        async with self.lock:
            return len(self.status_dict)


class VllmBackend(SllmBackend):
    """
    CPU Backend that loads model weights immediately and starts prefill.
    This backend is used for initial inference while GPU backend loads weights.
    """

    def __init__(
        self, instance_id: str, model: str, device: str, backend_config: Optional[Dict[str, Any]] = None, runtime_env: Optional[Dict[str, Any]] = None
    ) -> None:
        if backend_config is None:
            raise ValueError("Backend config is missing")
        if device not in ["cpu", "gpu"]:
            raise ValueError("Invalid device")

        self.status: BackendStatus = BackendStatus.UNINITIALIZED
        self.status_lock = asyncio.Lock()
        self.instance_id = instance_id
        self.model_name = model
        self.device = device
        self.backend_config = backend_config
        self.request_trace = LLMEngineStatusDict()
        self.trace_debug = backend_config.get("trace_debug", False)
        self.enforce_eager = backend_config.get("enforce_eager", False)
        self.enable_prefix_caching = backend_config.get(
            "enable_prefix_caching", True
        )
        self.task = backend_config.get("task", "auto")

        async_engine_fields = {f.name for f in fields(AsyncEngineArgs)}
        filtered_engine_config = {
            k: v for k, v in backend_config.items() if k in async_engine_fields
        }

        load_format = backend_config.get("load_format")
        torch_dtype = backend_config.get("torch_dtype")
        if torch_dtype is not None:
            filtered_engine_config["dtype"] = torch_dtype

        if load_format is not None:
            filtered_engine_config["load_format"] = load_format
            filtered_engine_config["model"] = backend_config.get(
                "pretrained_model_name_or_path"
            )
        else:
            storage_path = os.getenv("STORAGE_PATH", "./models")
            model_path = os.path.join(storage_path, "vllm", model)
            filtered_engine_config["model"] = model_path
            filtered_engine_config["load_format"] = "shm"

        filtered_engine_config["enforce_eager"] = self.enforce_eager
        filtered_engine_config["enable_prefix_caching"] = (
            self.enable_prefix_caching
        )
        filtered_engine_config["task"] = self.task

        # 可通过 backend_config 控制 load_method（tokenwise / layerwise）
        load_method = backend_config.get("load_method", None)
        if load_method is not None:
            filtered_engine_config["load_method"] = load_method

        self.lazy_load = False
        if filtered_engine_config["load_format"] == "shm":
            if self.device == "cpu":
                filtered_engine_config["kv_transfer_config"] = {
                    "kv_connector": "ShmConnector",
                    "kv_role": "kv_producer",
                    "kv_rank": 0
                }
            elif self.device == "gpu":
                filtered_engine_config["lazy_load"] = backend_config.get("lazy_load", False)
                filtered_engine_config["shm_tp_size"] = backend_config.get("shm_tp_size", 2)
                filtered_engine_config["kv_transfer_config"] = {
                    "kv_connector": "ShmConnector",
                    "kv_role": "kv_consumer",
                    "kv_rank": 1,
                    "shm_size": backend_config.get("shm_size", 4293918720),
                    "shm_num_blocks": backend_config.get("shm_num_blocks", 455),
                    "shm_block_len": backend_config.get("shm_block_len", 131072),
                    "shm_tp_size": backend_config.get("shm_tp_size", 2)
                }
                self.lazy_load = filtered_engine_config.get("lazy_load", False)

        self.runtime_env = runtime_env

        logger.info(
            f"Creating new VLLM engine with config: {filtered_engine_config}"
        )

        self.engine_args = AsyncEngineArgs(**filtered_engine_config)

        self.engine = None
        self.weights_loaded = False

        self.gpu_router = ray.get_actor(self.model_name, namespace="gpu_models")
        self.scheduler = ray.get_actor("model_loading_scheduler")

    async def start_profile(self):
        await self.engine.start_profile()

    async def stop_profile(self):
        await self.engine.stop_profile()

    async def init_backend(self) -> None:
        """Initialize the vLLM backend and load model weights."""
        if self.device == "cpu":
            os.environ["VLLM_CPU_OMP_THREADS_BIND"] = "64-92|96-124"
        elif self.device == "gpu":
            os.sched_setaffinity(0, {126})
            # 设置环境变量
            if self.runtime_env is not None:
                os.environ.update(self.runtime_env)

        logger.info(f"Initializing vLLM backend on {self.device}...")
        async with self.status_lock:
            if self.status != BackendStatus.UNINITIALIZED:
                logger.warning("vLLM backend already initialized")
                return
            start_time = time.time()
            # Create engine and load weights
            if self.lazy_load:
                logger.info("init engine except load model weigths")
            self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
            load_time = time.time() - start_time
            logger.info(f"vLLM backend initialized in {load_time * 1000:.2f} ms")
            self.status = BackendStatus.RUNNING

    async def lazy_load_weights(self, layer_idxes: list[int]) -> bool:
        start_time = time.time()
        await self.engine.lazy_init(layer_idxes=layer_idxes, no_warmup=False)
        print(f"DEBUG: GPU weights loaded, id(self)={id(self)}, time={time.time() - start_time}")
        self.weights_loaded = True
        return True

    async def get_visible_devices(self) -> List[int]:
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible_devices:
            # If CUDA_VISIBLE_DEVICES is not set, assume all devices are visible or handled by Ray
            # For simplicity in this context, we might need a better way to identify devices if not set
            # But usually Ray sets this.
            import torch
            return list(range(torch.cuda.device_count()))
        
        return [int(x) for x in visible_devices.split(",") if x.strip()]

    async def generate(self, request_data: Dict[str, Any], stream: bool = False):
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Engine is not running"}

        assert self.engine is not None

        if request_data is None:
            return {"error": "Request data is missing"}

        model_name: str = request_data.pop("model", self.model_name)
        messages: Dict[Dict[str, str], str] = request_data.pop("messages", [])
        construct_prompt: str = "\n".join(
            [
                f"{message['role']}: {message['content']}"
                for message in messages
                if "content" in message
            ]
        )

        input_tokens = request_data.get("input_tokens")
        inputs: Union[str, TokensPrompt] = request_data.pop(
            "prompt", construct_prompt
        )
        if input_tokens is not None:
            inputs = TokensPrompt(
                prompt_token_ids=request_data.pop("input_tokens"),
            )

        request_id: str = request_data.pop(
            "request_id", f"chatcmpl-{uuid.uuid4()}"
        )

        try:
            sampling_params = SamplingParams(**request_data)
        except Exception as e:
            return {"error": f"Invalid sampling parameters: {e}"}

        results_generator = self.engine.generate(
            inputs, sampling_params, request_id
        )

        latency_metrics = {}
        start_time = time.perf_counter()
        if not stream:
            start_time = time.perf_counter()
            final_output = None
            first_token_notified = False
            async for response_output in results_generator:
                final_output = response_output
                # Notify scheduler on first token (for layerwise cold-start readiness)
                if not first_token_notified and self.device == "gpu" and response_output.outputs:
                    try:
                        self.scheduler.notify_first_token_by_instance.remote(
                            self.instance_id, self.model_name
                        )
                    except Exception:
                        pass
                    first_token_notified = True
                await self.request_trace.update_status(request_id, response_output)
            end_time = time.perf_counter()
            latency_metrics["e2e"] = end_time - start_time
            latency_metrics["output_length"] = len(final_output.outputs[0].token_ids)
        else:
            start_time = time.perf_counter()
            final_output = None
            first_chunk_time = None
            itl_token_count = 0
            ttft = 0.0
            itl_list = []
            most_recent_timestamp = start_time
            async for response_output in results_generator:
                final_output = response_output
                load_weights_finished = response_output.load_weights_finished
                if load_weights_finished:
                    self.gpu_router.notify_weights_loaded.remote(self.instance_id)
                await self.request_trace.update_status(request_id, response_output)
                if response_output.outputs:
                    current_time = time.perf_counter()
                    if first_chunk_time is None:
                        first_chunk_time = current_time
                        ttft = first_chunk_time - start_time
                        # Notify scheduler on first token (for layerwise cold-start readiness)
                        if self.device == "gpu":
                            try:
                                self.scheduler.notify_first_token_by_instance.remote(
                                    self.instance_id, self.model_name
                                )
                            except Exception:
                                pass
                    else:
                        itl_token_count += 1
                        itl = current_time - most_recent_timestamp
                        itl_list.append(itl)
                    most_recent_timestamp = current_time
            end_time = time.perf_counter()
            latency_metrics["e2e"] = end_time - start_time
            latency_metrics["ttft"] = ttft
            latency_metrics["first_token_time"] = first_chunk_time
            latency_metrics["tpot"] = (end_time - first_chunk_time) / itl_token_count if itl_token_count > 0 else 0.0
            latency_metrics["itls"] = itl_list if itl_token_count > 0 else []
            latency_metrics["output_length"] = len(final_output.outputs[0].token_ids)

        assert final_output is not None
        if not self.trace_debug:
            await self.request_trace.delete_request(request_id)
        return process_output(final_output, latency_metrics, model_name)

    async def shutdown(self):
        """Shutdown the vLLM backend."""
        async with self.status_lock:
            if self.status == BackendStatus.DELETING:
                return
            self.status = BackendStatus.DELETING

        # Abort all requests
        requests = await self.request_trace.return_all_request_ids()
        tasks = [self.engine.abort(request_id) for request_id in requests]
        await asyncio.gather(*tasks)

        if hasattr(self, "engine") and self.engine is not None:
            print("Shutting down vLLM engine...")
            self.engine.shutdown()
            del self.engine
            self.engine = None

        print("vllm backend shut down")

    async def stop(self):
        """Wait for all requests to finish and shutdown the backend."""
        async with self.status_lock:
            if self.status.value >= BackendStatus.STOPPING.value:
                return
            self.status = BackendStatus.STOPPING
        while await self.request_trace.request_count() > 0:
            logger.info("Waiting for all requests to finish")
            await asyncio.sleep(1)
        logger.info("All requests finished. Shutting down the backend.")
        await self.shutdown()

    async def get_current_tokens(self) -> List[List[int]]:
        """Return a list of all ongoing request tokens."""
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return []
        results = await self.request_trace.return_all_results()
        ongoing_results: List[RequestOutput] = [
            result for result in results if isinstance(result, RequestOutput)
        ]
        tokens: List[List[int]] = [
            result.prompt_token_ids + result.outputs[0].token_ids
            for result in ongoing_results
        ]
        return tokens

    async def resume_kv_cache(self, request_datas: List[List[int]]) -> None:
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return
        constructed_inputs = [
            {
                "input_tokens": request_data,
                "max_tokens": 1,
            }
            for request_data in request_datas
        ]
        tasks = [self.generate(inputs, stream=False) for inputs in constructed_inputs]
        await asyncio.gather(*tasks)

    async def update_computing_layers(self, computing_layers: int):
        if self.device == "gpu" or self.engine_config["load_method"] != "layerwise":
            return
        await self.engine.update_computing_layers(computing_layers)