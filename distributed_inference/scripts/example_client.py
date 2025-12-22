#!/usr/bin/env python3
# ---------------------------------------------------------------------------- #
#  Distributed Inference - Example Client                                     #
#  Example usage of the distributed inference client                          #
# ---------------------------------------------------------------------------- #
import asyncio
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import ray
from distributed_inference.client.inference_client import InferenceClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Example client usage."""
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)
        logger.info("Ray initialized")

    # Create client
    client = InferenceClient(coordinator_name="migration_coordinator")
    client.connect()

    # Example 1: Single request
    logger.info("Sending single request...")
    result = await client.generate(
        prompt="What is the capital of France?",
        max_tokens=50,
        temperature=0.7,
    )
    logger.info(f"Result: {result}")

    # Example 2: Batch requests
    logger.info("Sending batch requests...")
    prompts = [
        "What is machine learning?",
        "Explain quantum computing.",
        "Describe the water cycle.",
    ]

    results = await client.generate_batch(
        prompts=prompts,
        max_tokens=100,
        temperature=0.7,
    )

    for i, result in enumerate(results):
        logger.info(f"Request {i+1} result: {result}")

    # Example 3: Chat format
    logger.info("Sending chat format request...")
    messages = [
        {"role": "user", "content": "Hello, how are you?"},
    ]

    result = await client.generate(
        messages=messages,
        max_tokens=50,
        temperature=0.7,
    )
    logger.info(f"Chat result: {result}")


if __name__ == "__main__":
    asyncio.run(main())

