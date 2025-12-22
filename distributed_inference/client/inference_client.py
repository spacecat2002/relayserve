# ---------------------------------------------------------------------------- #
#  Distributed Inference - Client                                            #
#  Client for sending requests to distributed inference system               #
# ---------------------------------------------------------------------------- #
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import ray

logger = logging.getLogger(__name__)


class InferenceClient:
    """
    Client for distributed inference system.
    Sends requests to the migration coordinator.
    """

    def __init__(self, coordinator_name: str):
        self.coordinator_name = coordinator_name
        self.coordinator = None

    def connect(self):
        """Connect to the migration coordinator."""
        try:
            self.coordinator = ray.get_actor(self.coordinator_name, namespace="vllm")
            logger.info(f"Connected to coordinator: {self.coordinator_name}")
        except Exception as e:
            logger.error(f"Failed to connect to coordinator: {e}")
            raise

    async def start_profile(self):
        await self.coordinator.start_profile.remote()

    async def stop_profile(self):
        await self.coordinator.stop_profile.remote()

    async def generate(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 100,
        temperature: float = 0.0,
        repetition_penalty: float = 1.0,
        request_id: Optional[str] = None,
        stream: bool = False,
    ):
        """
        Send a generation request.

        Args:
            prompt: Input prompt
            messages: List of messages in chat format
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            request_id: Optional request ID
            stream: If True, return async generator for streaming output
            **kwargs: Additional generation parameters

        Returns:
            If stream=False: Generation result dict
            If stream=True: Async generator yielding stream chunks with metrics
        """
        if self.coordinator is None:
            raise RuntimeError("Client not connected. Call connect() first.")

        request_data = {
            "prompt": prompt,
            "messages": messages or [],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "request_id": request_id,
            "repetition_penalty": repetition_penalty,
        }

        if stream:
            return self._generate_stream(request_data)

    async def _generate_stream(self, request_data: Dict[str, Any]):
        """
        Generate tokens with streaming output.
        TTFT and TPOT are automatically calculated by the system in real-time.

        Yields:
            Dict containing:
                - "text": Generated text
                - "ttft": Time to first token (automatically calculated)
                - "tpot": Time per output token (automatically calculated)
                - "metrics": Dict with timing metrics (automatically calculated)
        """
        # Get result from coordinator (metrics calculated in real-time)
        result = await self.coordinator.generate_stream.remote(request_data=request_data)
        
        # Yield result with pre-calculated metrics
        yield {
            "cpu_ttft": result.get("cpu_ttft", 0.0),
            "gpu_ttft": result.get("gpu_ttft", 0.0),
            "cpu_tpot": result.get("cpu_tpot", 0.0),
            "gpu_tpot": result.get("gpu_tpot", 0.0),
            "done": result.get("done", True),
            # 保持兼容：透传 start_time（如果协调器提供）
            "start_time": result.get("start_time", None),
        }

