# ---------------------------------------------------------------------------- #
#  SLLM backend ABC + vLLM helpers and package exports                          #
#  (merged former backend_utils + vllm_shared + this module)                  #
# ---------------------------------------------------------------------------- #
import asyncio
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import fields
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

from vllm import AsyncEngineArgs, AsyncLLMEngine, RequestOutput, SamplingParams
from vllm.inputs import TokensPrompt

logger = logging.getLogger("ray")

# ----------------------------------------------------------------------------- #
#  Backend ABC (all SLLM model backends)                                       #
# ----------------------------------------------------------------------------- #


class BackendStatus(Enum):
    UNINITIALIZED = auto()
    RUNNING = auto()
    STOPPING = auto()
    DELETING = auto()


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


class SllmBackend(ABC):
    """Model backends; vLLM CPU/GPU share profiling, shutdown, and token trace helpers."""

    engine: Optional[AsyncLLMEngine]
    instance_id: str
    model_name: str
    status: BackendStatus
    status_lock: asyncio.Lock
    request_trace: LLMEngineStatusDict
    trace_debug: bool

    _vllm_device_log_label: str = "device"

    @abstractmethod
    def __init__(
        self,
        instance_id: str,
        model_name: str,
        device: str,
        backend_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        pass

    async def start_profile(self):
        await self.engine.start_profile()

    async def stop_profile(self):
        await self.engine.stop_profile()

    @abstractmethod
    async def init_backend(self) -> None:
        pass

    @abstractmethod
    async def generate(self, request_data: Dict[str, Any]):
        pass

    async def shutdown(self):
        async with self.status_lock:
            if self.status == BackendStatus.DELETING:
                return
            self.status = BackendStatus.DELETING

        requests = await self.request_trace.return_all_request_ids()
        tasks = [self.engine.abort(request_id) for request_id in requests]
        await asyncio.gather(*tasks)

        if self.engine is not None:
            logger.info(
                "Shutting down vLLM %s engine instance_id=%s",
                self._vllm_device_log_label,
                self.instance_id,
            )
            self.engine.shutdown()
            self.engine = None

        logger.info(
            "vLLM %s backend shut down instance_id=%s",
            self._vllm_device_log_label,
            self.instance_id,
        )

    async def stop(self):
        async with self.status_lock:
            if self.status.value >= BackendStatus.STOPPING.value:
                return
            self.status = BackendStatus.STOPPING
        while await self.request_trace.request_count() > 0:
            logger.info("Waiting for all requests to finish")
            await asyncio.sleep(1)
        logger.info(
            "All requests finished. Shutting down the %s backend.",
            self._vllm_device_log_label,
        )
        await self.shutdown()

    async def get_current_tokens(self) -> List[List[int]]:
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return []
        results = await self.request_trace.return_all_results()
        ongoing_results: List[RequestOutput] = [
            result for result in results if isinstance(result, RequestOutput)
        ]
        return [
            result.prompt_token_ids + result.outputs[0].token_ids
            for result in ongoing_results
        ]

    async def resume_kv_cache(self, request_datas: List[List[int]]) -> None:
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return
        constructed_inputs = [
            {"input_tokens": request_data, "max_tokens": 1}
            for request_data in request_datas
        ]
        tasks = [self.generate(inputs) for inputs in constructed_inputs]
        await asyncio.gather(*tasks)


# ----------------------------------------------------------------------------- #
#  vLLM shared helpers (CPU / GPU backends import from here)                    #
# ----------------------------------------------------------------------------- #


def process_output(
    output: RequestOutput, latency_metrics: Dict[str, Any], model_name: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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


def build_core_filtered_engine_config(
    model: str, backend_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Fields for ``AsyncEngineArgs`` before ShmConnector / lazy-load overrides."""
    async_engine_fields = {f.name for f in fields(AsyncEngineArgs)}
    filtered: Dict[str, Any] = {
        k: v for k, v in backend_config.items() if k in async_engine_fields
    }

    torch_dtype = backend_config.get("torch_dtype")
    if torch_dtype is not None:
        filtered["dtype"] = torch_dtype

    load_format = backend_config.get("load_format")
    if load_format is not None:
        filtered["load_format"] = load_format
        filtered["model"] = backend_config.get("pretrained_model_name_or_path")
    else:
        storage_path = os.getenv("STORAGE_PATH", "./models")
        model_path = os.path.join(storage_path, "vllm", model)
        filtered["model"] = model_path
        filtered["load_format"] = "shm"

    filtered["enforce_eager"] = backend_config.get("enforce_eager", False)
    filtered["enable_prefix_caching"] = backend_config.get(
        "enable_prefix_caching", True
    )
    filtered["task"] = backend_config.get("task", "auto")

    load_method = backend_config.get("load_method", "tokenwise")
    if load_method is not None:
        filtered["load_method"] = load_method

    return filtered


def parse_vllm_generate_request(
    request_data: Dict[str, Any], default_model_name: str
) -> Tuple[str, Union[str, TokensPrompt], str, Dict[str, Any]]:
    """
    Returns ``(model_name, inputs, request_id, sampling_kwargs)`` for
    ``SamplingParams(**sampling_kwargs)`` after consuming prompt fields.
    """
    rd = dict(request_data)
    model_name: str = rd.pop("model", default_model_name)
    messages: List[Dict[str, str]] = rd.pop("messages", [])
    construct_prompt = "\n".join(
        [
            f"{message['role']}: {message['content']}"
            for message in messages
            if "content" in message
        ]
    )
    input_tokens = rd.get("input_tokens")
    inputs: Union[str, TokensPrompt] = rd.pop("prompt", construct_prompt)
    if input_tokens is not None:
        inputs = TokensPrompt(prompt_token_ids=rd.pop("input_tokens"))
    request_id: str = rd.pop("request_id", f"chatcmpl-{uuid.uuid4()}")
    return model_name, inputs, request_id, rd


def _parse_cpu_list(cpu_list: str) -> List[int]:
    cpus: List[int] = []
    for item in cpu_list.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            cpus.extend(range(int(start), int(end) + 1))
        else:
            cpus.append(int(item))
    return cpus


def read_numa_cpu_affinity(numa_node_id: int) -> List[int]:
    cpulist_path = f"/sys/devices/system/node/node{numa_node_id}/cpulist"
    if not os.path.exists(cpulist_path):
        return []
    try:
        with open(cpulist_path, "r", encoding="utf-8") as f:
            return _parse_cpu_list(f.read().strip())
    except Exception:
        return []


def async_engine_args_from_dict(filtered: Dict[str, Any]) -> AsyncEngineArgs:
    logger.info("Creating new VLLM engine with config: %s", filtered)
    return AsyncEngineArgs(**filtered)
