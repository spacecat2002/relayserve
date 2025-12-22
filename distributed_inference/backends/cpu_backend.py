# ---------------------------------------------------------------------------- #
#  Distributed Inference - CPU Backend                                        #
#  CPU backend that loads weights and starts prefill inference                #
# ---------------------------------------------------------------------------- #
import asyncio
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from vllm import (
    AsyncEngineArgs,
    AsyncLLMEngine,
    SamplingParams,
)

import ray

from sllm.backends.backend_utils import BackendStatus, SllmBackend

from sllm.logger import init_logger

logger = init_logger(__name__)

@ray.remote(resources={"cpu_node": 1})
class CpuBackend(SllmBackend):
    """
    CPU Backend that loads model weights immediately and starts prefill.
    This backend is used for initial inference while GPU backend loads weights.
    """

    def __init__(
        self, model: str, backend_config: Optional[Dict[str, Any]] = None
    ) -> None:
        if backend_config is None:
            raise ValueError("Backend config is missing")

        self.status: BackendStatus = BackendStatus.UNINITIALIZED
        self.status_lock = asyncio.Lock()
        self.backend_config = backend_config
        self.model_name = model
        self.engine = None
        self.engine_args = None

        # Configuration for CPU backend
        storage_path = os.getenv("STORAGE_PATH", "./models")
        model_path = os.path.join(storage_path, "vllm", model)

        # CPU-specific engine args
        self.engine_config = {
            "model": model_path,
            "load_format": "shm",
            "enforce_eager": True,
            "enable_prefix_caching": False,  # Disable prefix caching
            "task": "auto",
            "dtype": "bfloat16",
            "tensor_parallel_size": 2,
            "max_model_len": 4096,
            "load_method": "layerwise",
        }
        logger.info(f"Creating CPU backend with config: {self.engine_config}")
        self.engine_args = AsyncEngineArgs(**self.engine_config)

    async def start_profile(self):
        await self.engine.start_profile()

    async def stop_profile(self):
        await self.engine.stop_profile()

    async def init_backend(self) -> None:
        """Initialize the CPU backend and load model weights."""
        async with self.status_lock:
            if self.status != BackendStatus.UNINITIALIZED:
                logger.warning("CPU backend already initialized")
                return

            logger.info("Initializing CPU backend and loading weights...")
            start_time = time.time()

            # os.sched_setaffinity(0, {0, 32})
            os.environ["VLLM_CPU_OMP_THREADS_BIND"] = "64-92|96-124"
            #os.environ["VLLM_CPU_OMP_THREADS_BIND"] = "0-28|32-60" 

            # Create engine and load weights
            self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)

            load_time = time.time() - start_time
            logger.info(f"CPU backend initialized in {load_time:.2f} seconds")
            self.status = BackendStatus.RUNNING

    async def generate(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate tokens (non-streaming).
        This method is required by SllmBackend abstract class.
        For streaming, use generate_stream instead.
        """
        # For non-streaming, use generate_stream and return the final result
        result = await self.generate_stream(request_data)
        return result

    async def update_computing_layers(self, computing_layers: int):
        if self.engine_config["load_method"] != "layerwise":
            return
        await self.engine.update_computing_layers(computing_layers)

    async def generate_stream(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate tokens with streaming output.
        Calculates TTFT and TPOT in real-time as tokens are generated.
        
        Supports two execution modes:
        1. Only partial prefill (if max_prefill_tokens is set)
        2. Prefill + decode (if max_prefill_tokens is not set or None)
        
        Returns:
            Dict containing output data and metrics calculated in real-time
        """
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "CPU backend is not running", "done": True}

        if self.engine is None:
            return {"error": "CPU engine not initialized", "done": True}

        if request_data is None:
            return {"error": "Request data is missing", "done": True}

        # Extract request parameters
        model_name: str = request_data.pop("model", self.model_name)
        messages = request_data.pop("messages", [])
        construct_prompt: str = "\n".join(
            [
                f"{message['role']}: {message['content']}"
                for message in messages
                if "content" in message
            ]
        )

        inputs = request_data.pop("prompt", construct_prompt)

        request_id: str = request_data.pop(
            "request_id", f"cpu-{uuid.uuid4()}"
        )

        try:
            sampling_params = SamplingParams(**request_data)
        except Exception as e:
            return {"error": f"Invalid sampling parameters: {e}", "done": True}

        # Generate tokens with streaming
        results_generator = self.engine.generate(
            inputs, sampling_params, request_id
        )

        # Record start time for timing metrics
        stream_start_time = time.perf_counter()
        first_token_time = None
        token_count = 0
        prefill_token_count = 0
        decode_token_count = 0
        generated_text = ""
        generated_token_ids = []
        prompt_token_ids = []
        itl_list = []  # Inter-token latencies
        most_recent_timestamp = stream_start_time

        # Process stream results and calculate metrics in real-time
        async for response_output in results_generator:
            if response_output.outputs:
                # Record timestamp for this output
                current_time = time.perf_counter()
                
                # Calculate TTFT on first token
                if first_token_time is None:
                    first_token_time = current_time
                    ttft = first_token_time - stream_start_time
                else:
                    token_count += 1
                    # Calculate inter-token latency (ITL)
                    itl = current_time - most_recent_timestamp
                    itl_list.append(itl)
                
                # Calculate TPOT (average time per token so far)
                time_since_first = current_time - first_token_time
                tpot = time_since_first / token_count if token_count > 0 else 0.0
                
                # Extract text and tokens from output
                text = response_output.outputs[0].text
                token_ids = response_output.outputs[0].token_ids
                current_prompt_token_ids = response_output.prompt_token_ids
                
                generated_text = text
                generated_token_ids = token_ids
                prompt_token_ids = current_prompt_token_ids
                
                most_recent_timestamp = current_time

        # Calculate final metrics
        if first_token_time is not None:
            total_time = time.perf_counter() - stream_start_time
            final_tpot = (time.perf_counter() - first_token_time) / token_count if token_count > 0 else 0.0
            print(f"CPU TTFT: {first_token_time - stream_start_time}, CPU TPOT: {final_tpot}")
            return {
                "done": True,
                "ttft": first_token_time - stream_start_time,
                "tpot": final_tpot,
            }
        else:
            return {"done": True, "error": "No tokens generated"}

    async def get_current_tokens(self) -> List[List[int]]:
        """Get current tokens for migration."""
        if self.engine is None:
            return []
        # This would need to be implemented based on vLLM's internal state
        return []

    async def resume_kv_cache(self, request_datas: List[List[int]]) -> None:
        """Resume KV cache (not typically used in CPU backend)."""
        pass

    async def encode(self, request_data: Dict[str, Any]):
        """Encode input (not implemented for CPU backend)."""
        return {"error": "Encode not supported in CPU backend"}

    async def shutdown(self):
        """Shutdown the CPU backend."""
        async with self.status_lock:
            if self.status == BackendStatus.DELETING:
                return
            self.status = BackendStatus.DELETING

        if hasattr(self, "engine") and self.engine is not None:
            print("Shutting down CPU engine...")
            self.engine.shutdown()
            del self.engine
            self.engine = None

        print("CPU backend shut down")

    async def stop(self):
        """Stop the CPU backend gracefully."""
        async with self.status_lock:
            if self.status.value >= BackendStatus.STOPPING.value:
                return
            self.status = BackendStatus.STOPPING

        await self.shutdown()

    async def lazy_load_weigths(self, end_layer: int = -1, warmup: bool = False):
        pass

