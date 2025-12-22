# ---------------------------------------------------------------------------- #
#  Distributed Inference - GPU Backend                                         #
#  GPU backend with lazy loading (loads weights after CPU starts prefill)     #
# ---------------------------------------------------------------------------- #
import asyncio
import gc
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import torch
from vllm import (
    AsyncEngineArgs,
    AsyncLLMEngine,
    RequestOutput,
    SamplingParams,
)
from vllm.inputs import TokensPrompt

import ray

from sllm.backends.backend_utils import BackendStatus, SllmBackend
from distributed_inference.backends.cgroup_utils import (
    set_cpu_affinity_with_cgroup,
    cleanup_cgroup,
    setup_cgroup_cpuset,
    move_process_to_cgroup,
)

logger = logging.getLogger(__name__)


@ray.remote(resources={"gpu_node": 1}, num_gpus=1)
class GpuBackend(SllmBackend):
    """
    GPU Backend with lazy loading.
    This backend waits for weights to load while CPU backend handles initial prefill.
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
        self.weights_loaded = False
        self.loading_task = None
        self.cgroup_path = None  # 存储 cgroup 路径以便后续清理

        # Configuration for GPU backend with lazy loading
        storage_path = os.getenv("STORAGE_PATH", "./models")
        model_path = os.path.join(storage_path, "vllm", model)

        # GPU-specific engine args with lazy loading
        engine_config = {
            "model": model_path,
            "load_format": "shm",
            "enforce_eager": True,
            "enable_prefix_caching": False,  # Disable prefix caching
            "lazy_load": True,  # Enable lazy loading
            "dtype": "bfloat16",
            # "shm_tp_size": 1,
            # "kv_transfer_config": {
            #     "kv_connector": "ShmConnector",
            #     "kv_role": "kv_consumer",
            #     "kv_rank": 1,
            #     "shm_size": 4284481536,
            #     "shm_num_blocks": 227,
            #     "shm_block_len": 262144,
            #     "shm_tp_size": 1
            # }
            "load_method": "layerwise",
            "shm_tp_size": 2,
            "shm_kv_cache_size": 4293918720,
            "shm_num_blocks": 455,
            "shm_block_len": 131072,
            "block_size": 128
        }

        logger.info(f"Creating GPU backend with lazy loading config: {engine_config}")
        self.engine_args = AsyncEngineArgs(**engine_config)

    async def start_profile(self):
        await self.engine.start_profile()

    async def stop_profile(self):
        await self.engine.stop_profile()

    async def init_backend(self) -> None:
        """Initialize the GPU backend and load model weights."""
        async with self.status_lock:
            if self.status != BackendStatus.UNINITIALIZED:
                logger.warning("GPU backend already initialized")
                return

            logger.info("Initializing GPU backend and loading weights...")
            start_time = time.time()

            os.sched_setaffinity(0, {126})
            #os.sched_setaffinity(0, {62})
            # 如果 cgroup 不可用，会自动回退到 os.sched_setaffinity
            # cpus = {62}  # 可以从 backend_config 中读取配置
            # cgroup_name = f"gpu_backend_{os.getpid()}"
                        
            # self.cgroup_path = setup_cgroup_cpuset(
            #     cgroup_name=cgroup_name,
            #     cpus=cpus,
            #     parent_cgroup=None,  # 可以设置为 "ray.slice" 等父 cgroup
            # )
            
            # if self.cgroup_path:
            #     # 将当前进程移动到 cgroup
            #     move_process_to_cgroup(0, self.cgroup_path)
            #     logger.info(f"Using cgroup for CPU affinity: {self.cgroup_path}")
            # else:
            #     # 回退到 os.sched_setaffinity
            #     set_cpu_affinity_with_cgroup(
            #         cpus=cpus,
            #         cgroup_name=cgroup_name,
            #         fallback_to_sched=True,
            #     )            


            # Create engine and load weights
            self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)

            load_time = time.time() - start_time
            logger.info(f"CPU backend initialized in {load_time:.2f} seconds")
            self.status = BackendStatus.RUNNING

    async def wait_for_weights(self, layer_idxes: list[int]) -> bool:
        start_time = time.time()
        await self.engine.lazy_init(layer_idxes=layer_idxes, no_warmup=False)
        print(f"DEBUG: GPU weights loaded, id(self)={id(self)}, time={time.time() - start_time}")
        self.weights_loaded = True
        return True

    async def generate(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate tokens (non-streaming).
        This method is required by SllmBackend abstract class.
        For streaming, use generate_stream instead.
        """
        # For non-streaming, use generate_stream and return the final result
        result = await self.generate_stream(request_data)
        return result

    async def generate_stream(
        self, request_data: Dict[str, Any], migrated_tokens: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Generate tokens with streaming output.
        Calculates TTFT and TPOT in real-time as tokens are generated.
        
        Args:
            request_data: Request data
            migrated_tokens: Optional tokens migrated from CPU backend
            
        Returns:
            Dict containing output data and metrics calculated in real-time
        """
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "GPU backend is not running", "done": True}

        # Wait for weights to be loaded
        if not self.weights_loaded:
            return {"error": "GPU weights not loaded", "done": True}

        if self.engine is None:
            return {"error": "GPU engine not initialized", "done": True}

        if request_data is None:
            return {"error": "Request data is missing", "done": True}

        # Extract request parameters
        model_name: str = request_data.pop("model", self.model_name)
        request_id: str = request_data.pop(
            "request_id", f"gpu-{uuid.uuid4()}"
        )

        messages = request_data.pop("messages", [])
        construct_prompt: str = "\n".join(
            [
                f"{message['role']}: {message['content']}"
                for message in messages
                if "content" in message
            ]
        )
        inputs = request_data.pop("prompt", construct_prompt)

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
        prefill_completed = False
        
        # Process stream results and calculate metrics in real-time
        async for response_output in results_generator:
            if response_output.outputs:
                # Record timestamp for this output
                current_time = time.perf_counter()
                
                # Calculate TTFT on first token
                if first_token_time is None:
                    first_token_time = current_time
                    ttft = first_token_time - stream_start_time
                    print(f"ttft: {ttft}")
                else:
                    token_count += 1
                    # Calculate inter-token latency (ITL)
                    itl = current_time - most_recent_timestamp
                    print(f"itl: {itl}")
                    itl_list.append(itl)
                
                
                # Calculate TPOT (average time per token so far)
                time_since_first = current_time - first_token_time
                tpot = time_since_first / token_count if token_count > 0 else 0.0
                
                most_recent_timestamp = current_time
                

        # Calculate final metrics
        if first_token_time is not None:
            total_time = time.perf_counter() - stream_start_time
            final_tpot = (time.perf_counter() - first_token_time) / token_count if token_count > 0 else 0.0
            print(f"GPU TTFT: {first_token_time - stream_start_time}, GPU TPOT: {final_tpot}")
            print(f"itl_list: {itl_list}")
            return {
                "done": True,
                "ttft": first_token_time,
                "tpot": final_tpot,
            }
        else:
            return {"done": True, "error": "No tokens generated"}

    async def shutdown(self):
        """Shutdown the GPU backend."""
        async with self.status_lock:
            if self.status == BackendStatus.DELETING:
                return
            self.status = BackendStatus.DELETING

        if hasattr(self, "engine") and self.engine is not None:
            print("Shutting down GPU engine...")
            self.engine.shutdown()
            del self.engine
            self.engine = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print("GPU backend shut down")

    async def stop(self):
        """Stop the GPU backend gracefully."""
        async with self.status_lock:
            if self.status.value >= BackendStatus.STOPPING.value:
                return
            self.status = BackendStatus.STOPPING

        await self.shutdown()

    async def encode(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encode input tokens.
        This method is required by SllmBackend abstract class.
        """
        return {"error": "Encode not supported in GPU backend"}

    async def resume_kv_cache(self, request_datas: List[List[int]]) -> None:
        """Resume KV cache (not typically used in GPU backend)."""
        pass

    async def get_current_tokens(self) -> List[List[int]]:
        """Get current tokens for migration."""
        return []

    async def lazy_load_weigths(self, end_layer: int = -1, warmup: bool = False):
        pass

