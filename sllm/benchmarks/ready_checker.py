# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Portions adapted from the vLLM project
"""Wait until the serving endpoint responds successfully."""

import asyncio
import time

import aiohttp
from tqdm.asyncio import tqdm

from sllm.benchmarks.endpoint_request_func import RequestFuncInput, RequestFuncOutput


async def wait_for_endpoint(
    request_func,
    test_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    timeout_seconds: int = 600,
    retry_interval: int = 5,
) -> RequestFuncOutput:
    deadline = time.perf_counter() + timeout_seconds
    output = RequestFuncOutput(success=False)
    print(f"Waiting for endpoint to become up in {timeout_seconds} seconds")

    with tqdm(
        total=timeout_seconds,
        bar_format="{desc} |{bar}| {elapsed} elapsed, {remaining} remaining",
        unit="s",
    ) as pbar:
        while True:
            remaining = deadline - time.perf_counter()
            elapsed = timeout_seconds - remaining
            update_amount = min(elapsed - pbar.n, timeout_seconds - pbar.n)
            pbar.update(update_amount)
            pbar.refresh()
            if remaining <= 0:
                pbar.close()
                break

            try:
                output = await request_func(
                    request_func_input=test_input, session=session
                )
                if output.success:
                    pbar.close()
                    return output
            except aiohttp.ClientConnectorError:
                pass

            sleep_duration = min(retry_interval, remaining)
            if sleep_duration > 0:
                await asyncio.sleep(sleep_duration)

    return output
