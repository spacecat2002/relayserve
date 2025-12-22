#!/usr/bin/env python3
# ---------------------------------------------------------------------------- #
#  Distributed Inference - Streaming Client Example                            #
#  Example usage of streaming inference with TTFT and TPOT metrics            #
# ---------------------------------------------------------------------------- #
import argparse
import asyncio
import logging
import sys
import os
import uuid
from transformers import AutoTokenizer
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import ray
from distributed_inference.client.inference_client import InferenceClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_prompt(num_tokens: int, tokenizer: AutoTokenizer) -> str:
    random_part = np.arange(num_tokens * 2).tolist()
    token_ids = random_part

    prompt = tokenizer.decode(token_ids)
    reencoded = tokenizer.encode(prompt, add_special_tokens=False)[:num_tokens]

    prompt = tokenizer.decode(reencoded)
    return prompt


async def main():
    """Example streaming client usage with TTFT and TPOT metrics."""
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)
        logger.info("Ray initialized")

    # Create client
    client = InferenceClient(coordinator_name="migration_coordinator")
    client.connect()

    # Example 1: Streaming generation with metrics
    logger.info("=" * 60)
    logger.info("Example 1: Streaming generation with TTFT and TPOT")
    logger.info("=" * 60)


    prompt_len = 2048
    tokenizer = AutoTokenizer.from_pretrained("/home/zwh/.cache/huggingface/Qwen3-8B")
    prompt = build_prompt(prompt_len, tokenizer)

    print("\nStreaming output:")
    print("-" * 60)

    request_id = f"req-{uuid.uuid4()}"

    profile = True
    if profile:
        await client.start_profile()

    result = await client.generate(
        prompt=prompt,
        max_tokens=5,
        temperature=0.0,
        stream=True,
        request_id=request_id
    )
    
    # Get the result (since Ray doesn't support async generators, we get the full result)
    async for data in result:
        text = data.get("text", "")
        cpu_ttft = data.get("cpu_ttft", 0.0)
        cpu_tpot = data.get("cpu_tpot", 0.0)    
        gpu_ttft = data.get("gpu_ttft", 0.0) - data.get("start_time", 0.0)
        gpu_tpot = data.get("gpu_tpot", 0.0)    
        backend = data.get("backend", "unknown")
        done = data.get("done", False)

        print(f"CPU TTFT: {cpu_ttft}, CPU TPOT: {cpu_tpot}, GPU TTFT: {gpu_ttft}, GPU TPOT: {gpu_tpot}")
        
        # Print generated text
        if text:
            print(text, end="", flush=True)
    
    if profile:
        await client.stop_profile()


if __name__ == "__main__":
    asyncio.run(main())

