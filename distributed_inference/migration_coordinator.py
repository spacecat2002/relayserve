# ---------------------------------------------------------------------------- #
#  Distributed Inference - Migration Coordinator                              #
#  Coordinates inference migration from CPU to GPU backend                    #
# ---------------------------------------------------------------------------- #
import asyncio
import logging
from typing import Any, Dict, List, Optional

import ray

from .backends.start_backend import start_backend

from sllm.logger import init_logger

logger = init_logger(__name__)


@ray.remote
class MigrationCoordinator:
    """
    Coordinates inference migration from CPU to GPU backend.
    Manages the lifecycle of CPU and GPU backends and handles migration.
    """

    def __init__(
        self,
        model_name: str,
        cpu_backend_config: Dict[str, Any],
        gpu_backend_config: Dict[str, Any],
        cpu_backend_name: Optional[str] = None,
        gpu_backend_name: Optional[str] = None,
    ):
        self.model_name = model_name
        self.cpu_backend_config = cpu_backend_config
        self.gpu_backend_config = gpu_backend_config

        # Backend names (used to get actors via ray.get_actor)
        self.cpu_backend_name = cpu_backend_name or f"{model_name}_cpu_backend"
        self.gpu_backend_name = gpu_backend_name or f"{model_name}_gpu_backend"

        # Store actor handles instead of direct instances
        self.cpu_backend: Optional[ray.actor.ActorHandle] = None
        self.gpu_backend: Optional[ray.actor.ActorHandle] = None

        self.migration_lock = asyncio.Lock()
        self.migrated_requests: Dict[str, Dict[str, Any]] = {}

        self.gpu_weights_ready = False

    async def start_profile(self):
        #await self.cpu_backend.start_profile.remote()
        await self.gpu_backend.start_profile.remote()

    async def stop_profile(self):
        #await self.cpu_backend.stop_profile.remote()
        await self.gpu_backend.stop_profile.remote()

    async def initialize(self):
        """Initialize both CPU and GPU backends using start_backend."""
        logger.info(f"Initializing backends for model {self.model_name}")

        # Start CPU backend on CPU node (loads weights immediately)
        cpu_startup_config = {
            "num_cpus": 64,
            "num_gpus": 0,
            "resources": {"cpu_node": 1},
        }

        await start_backend.options(
            resources={"cpu_node": 1}, num_cpus=64
        ).remote(
            backend_name=self.cpu_backend_name,
            backend_type="cpu",
            model_name=self.model_name,
            backend_config=self.cpu_backend_config,
            startup_config=cpu_startup_config,
        )
        logger.info(f"Started CPU backend: {self.cpu_backend_name}")

        # Get CPU backend actor handle
        self.cpu_backend = ray.get_actor(self.cpu_backend_name)
        await self.cpu_backend.init_backend.remote()
        logger.info("CPU backend initialized and ready")

        # Start GPU backend on GPU node (lazy loading)
        gpu_startup_config = {
            "num_cpus": 4,
            "num_gpus": 2,
            "resources": {"gpu_node": 1},
        }

        await start_backend.options(
            resources={"gpu_node": 1}, num_gpus=2
        ).remote(
            backend_name=self.gpu_backend_name,
            backend_type="gpu",
            model_name=self.model_name,
            backend_config=self.gpu_backend_config,
            startup_config=gpu_startup_config,
        )
        logger.info(f"Started GPU backend: {self.gpu_backend_name}")

        # Get GPU backend actor handle
        self.gpu_backend = ray.get_actor(self.gpu_backend_name)
        await self.gpu_backend.init_backend.remote()
        logger.info("GPU backend initialization started (lazy loading)")

    async def generate_stream(self, request_data: Dict[str, Any]):
        """
        Generate tokens with streaming output and automatic migration.
        Metrics are calculated in real-time by backends as tokens are generated.
        
        Returns:
            List of stream chunks (due to Ray limitation with async generators)
        """
        import time
        
        # Ensure request_id is set and consistent across CPU and GPU
        request_id = request_data.get("request_id", f"req-{asyncio.get_event_loop().time()}")
        request_data["request_id"] = request_id  # Ensure it's in request_data
        prompt = request_data.get("prompt", [])

        # Record coordinator start time for overall metrics
        start_time = time.perf_counter()

        assert self.cpu_backend and self.gpu_backend, "CPU and GPU backends must be initialized"

        print(f"DEBUG: At start of generate_stream, gpu_weights_ready={self.gpu_weights_ready}, id(self)={id(self)}")

        if self.gpu_weights_ready:
            # GPU is ready, use GPU directly
            # Metrics are calculated in real-time by backend
            # Ensure request_id is preserved
            gpu_request = request_data.copy()
            gpu_request["request_id"] = request_id
            # NOTE: Use keyword argument for dict parameter (Ray issue #26283)
            result = await self.gpu_backend.generate_stream.remote(request_data=gpu_request)
            result["start_time"] = start_time
            return result
        # else:
        #     request = request_data.copy()
        #     request["request_id"] = request_id
        #     #layer_idxes = [list(range(5, 34))]
        #     layer_idxes = [[3,4,5,6,7,8,9,10,11,12,13,14,15,0,1,2], list(range(16, 36))]
        #     await self.cpu_backend.update_computing_layers.remote(computing_layers=8)
        #     await self.cpu_backend.enable_pp_cleanup.remote()
        #     self.cpu_backend.generate_stream.remote(request_data=request)
        #     time1 = time.perf_counter()
        #     await self.gpu_backend.wait_for_weights.remote(layer_idxes=layer_idxes, request_id=request_id)
        #     time2 = time.perf_counter()
        #     # 将时间写入文件
        #     # with open("gpu_latency.txt", "a") as f:
        #     #     f.write(f"gpu_weights_task: {time2 - time1}\n")
        #     print(f"DEBUG: gpu_weights_task: {time2 - time1}")
            
        #     gpu_task = self.gpu_backend.generate_stream.remote(request_data=request)
        #     gpu_result = await gpu_task

        else:
            computed_tokens = -1
            # CPU does prefill + decode, check GPU periodically
            request = request_data.copy()
            request["request_id"] = request_id  # Ensure same request_id
            request["extra_args"] = {}
            request["extra_args"]["kv_transfer_params"] = {
                "do_remote_decode": True,
                "do_remote_prefill": False,
                "max_num_prefill_compute_tokens": computed_tokens,
            }
            
            # Start both CPU generation and GPU weight waiting concurrently
            # Ray's .remote() returns ObjectRef which can be awaited directly
            # NOTE: Use keyword argument for dict parameter (Ray issue #26283)
            cpu_task = self.cpu_backend.generate_stream.remote(request_data=request)
            #layer_idxes = [list(range(0, 16)), list(range(16, 36))]
            layer_idxes = [list(range(0,36))]
            gpu_weights_task = self.gpu_backend.wait_for_weights.remote(layer_idxes=layer_idxes, request_id=request_id)
            # Do not run a full GPU generate in parallel with CPU on the same request_id:
            # both sides may block in KV transfer (CPU waiting for GPU / GPU waiting for CPU).
            # Wait for CPU to finish the producer phase, then run GPU continuation once below.
            cpu_result = await cpu_task
            await gpu_weights_task

            print(f"DEBUG: CPU result: {cpu_result}")
            
            # if cpu_result is not None and "error" in cpu_result:
            #     # Cancel GPU weights task if CPU failed (ObjectRef cannot be cancelled, but we can ignore it)
            #     print(f"DEBUG: CPU generation failed, returning CPU result: {cpu_result}")
            #     return cpu_result
            
            # Check if GPU weights are ready (may have completed before or after CPU)
            # ObjectRef can be awaited directly

            # print(f"DEBUG: After await gpu_weights_task, weights_ready={weights_ready}, id(self)={id(self)}")
            # # Only set gpu_weights_ready if weights are actually ready
            
            # Case 1: CPU did prefill + some decode, GPU does remaining decode
            print(f"Migrating request {request_id} from CPU to GPU")
            
            # Get accumulated tokens from CPU result
            
            new_prompt = prompt + cpu_result.get("generated_text", "")
            
            # Continue on GPU for remaining decode
            gpu_request = request_data.copy()
            gpu_request["request_id"] = request_id  # Ensure same request_id
            gpu_request["prompt"] = new_prompt

            gpu_request["extra_args"] = {}
            gpu_request["extra_args"]["kv_transfer_params"] = {
                "do_remote_decode": False,     # Enable remote decode
                "do_remote_prefill": True,   # This is the prefill instance
                "num_computed_tokens": computed_tokens,         # prompt中最多计算的token数，-1代表计算所有token
                "remote_block_ids": cpu_result.get("kv_transfer_params", {}).get("remote_block_ids", []),
            }
            # NOTE: Use keyword argument for dict parameter (Ray issue #26283)

            #cpu_task2 = self.cpu_backend.generate_stream.remote(request_data=cpu_request)
            gpu_result = await self.gpu_backend.generate_stream.remote(request_data=gpu_request)
            print(f"DEBUG: GPU result: {gpu_result}")

            
            # Calculate overall metrics
            if cpu_result is not None:
                cpu_ttft = cpu_result.get("ttft", 0.0)
                cpu_tpot = cpu_result.get("tpot", 0.0)
            else:
                cpu_ttft = 0.0
                cpu_tpot = 0.0

            if gpu_result is not None:
                gpu_ttft = gpu_result.get("ttft", 0.0)
                gpu_tpot = gpu_result.get("tpot", 0.0)
            else:
                gpu_ttft = 0.0
                gpu_tpot = 0.0
            
            return {
                "done": True,
                "cpu_ttft": cpu_ttft,
                "gpu_ttft": gpu_ttft,
                "cpu_tpot": cpu_tpot,
                "gpu_tpot": gpu_tpot,
                "start_time": start_time,
            }

    async def shutdown(self):
        """Shutdown both backends."""
        print("Shutting down migration coordinator")

        if self.gpu_backend is not None:
            try:
                await self.gpu_backend.shutdown.remote()
                ray.kill(self.gpu_backend)
            except Exception as e:
                print(f"Error shutting down GPU backend: {e}")
                # Try to kill the actor anyway
                try:
                    ray.kill(self.gpu_backend)
                except Exception as kill_error:
                    print(f"Error killing GPU backend actor: {kill_error}")

        if self.cpu_backend is not None:
            try:
                await self.cpu_backend.shutdown.remote()
                ray.kill(self.cpu_backend)
            except Exception as e:
                print(f"Error shutting down CPU backend: {e}")
                # Try to kill the actor anyway
                try:
                    ray.kill(self.cpu_backend)
                except Exception as kill_error:
                    print(f"Error killing CPU backend actor: {kill_error}")

        print("Migration coordinator shut down")

