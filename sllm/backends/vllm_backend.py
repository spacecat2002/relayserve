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
            "message": {
                "role": "assistant",
                "content": result.text,
            },
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
        self, model: str, device: str, backend_config: Optional[Dict[str, Any]] = None
    ) -> None:
        if backend_config is None:
            raise ValueError("Backend config is missing")
        if device not in ["cpu", "gpu"]:
            raise ValueError("Invalid device")

        self.status: BackendStatus = BackendStatus.UNINITIALIZED
        self.status_lock = asyncio.Lock()
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

        logger.info(
            f"Creating new VLLM engine with config: {filtered_engine_config}"
        )

        self.engine_args = AsyncEngineArgs(**filtered_engine_config)

        self.engine = None
        self.weights_loaded = False

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

    async def lazy_load_weigths(self, end_layer: int = -1, warmup: bool = False):
        if self.device == "cpu" or not self.lazy_load or self.weights_loaded:
            return
        start_time = time.perf_counter()
        await self.engine.lazy_init(start_layer=0, end_layer=end_layer, warmup=warmup)
        load_time = time.perf_counter() - start_time
        logger.info(f"Lazy load model weigths in {load_time * 1000:.2f} ms")
        self.weights_loaded = True

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

        # If prompt is not provided, construct it from messages
        inputs: Union[str, TokensPrompt] = request_data.pop(
            "prompt", construct_prompt
        )
        if request_data.get("input_tokens") is not None:
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
            async for response_output in results_generator:
                final_output = response_output
                await self.request_trace.update_status(request_id, response_output)
            end_time = time.perf_counter()
            latency_metrics["e2e"] = end_time - start_time
        else:
            start_time = time.perf_counter()
            final_output = None
            first_chunk_time = None
            itl_token_count = 0
            ttft = 0.0
            itl_list = []
            most_recent_timestamp = start_time
            prefill_completed = (
                request_data["extra_args"]["kv_transfer_params"]["max_num_prefill_compute_tokens"] in [len(inputs), -1]
            )
            async for response_output in results_generator:
                final_output = response_output
                await self.request_trace.update_status(request_id, response_output)
                if response_output.outputs:
                    current_time = time.perf_counter()
                    if first_chunk_time is None:
                        first_chunk_time = current_time
                        if prefill_completed:
                            ttft = first_chunk_time - start_time
                    else:
                        itl_token_count += 1
                        itl = current_time - most_recent_timestamp
                        itl_list.append(itl)
                    most_recent_timestamp = current_time
            end_time = time.perf_counter()
            latency_metrics["e2e"] = end_time - start_time
            latency_metrics["ttft"] = ttft
            latency_metrics["tpot"] = (end_time - first_chunk_time) / itl_token_count if itl_token_count > 0 else 0.0
            latency_metrics["itls"] = itl_list if itl_token_count > 0 else []

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

        print("CPU backend shut down")

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