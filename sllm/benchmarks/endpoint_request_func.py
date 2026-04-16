# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Portions adapted from the vLLM project
"""HTTP request helpers for SLLM benchmarks (non-OpenAI-streaming server)."""

import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import aiohttp
from tqdm.asyncio import tqdm

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    model_name: Optional[str] = None
    logprobs: Optional[int] = None
    extra_body: Optional[dict] = None
    multi_modal_content: Optional[Union[dict, list[dict]]] = None
    ignore_eos: bool = False
    language: Optional[str] = None
    request_id: Optional[str] = None


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0
    itl: list[float] = field(default_factory=list)
    tpot: float = 0.0
    prompt_len: int = 0
    error: str = ""


def _parse_sllm_json_body(data: Any) -> tuple[Optional[dict], Optional[dict], Optional[str]]:
    """SLLM ``/v1/chat/completions`` returns ``[api_response, latency_metrics]`` JSON."""
    if isinstance(data, dict) and data.get("error"):
        return None, None, str(data["error"])
    if isinstance(data, (list, tuple)) and len(data) == 2:
        body, metrics = data[0], data[1]
        if isinstance(body, dict) and isinstance(metrics, dict):
            return body, metrics, None
    if isinstance(data, dict) and "choices" in data:
        return data, {}, None
    return None, None, f"Unexpected response shape: {type(data).__name__}"


async def async_request_sllm_chat_completions(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    """POST JSON to SLLM (no SSE); parse ``(completion, metrics)`` tuple."""
    api_url = request_func_input.api_url
    if not api_url.rstrip("/").endswith("chat/completions"):
        raise AssertionError(
            "SLLM benchmark expects api_url ending with .../v1/chat/completions"
        )

    model = (
        request_func_input.model_name
        if request_func_input.model_name
        else request_func_input.model
    )
    payload: dict[str, Any] = {
        "model": model,
        "prompt": request_func_input.prompt,
        "max_tokens": request_func_input.output_len,
        "temperature": 0.0,
    }
    if request_func_input.request_id:
        payload["request_id"] = request_func_input.request_id
    if request_func_input.ignore_eos:
        payload["ignore_eos"] = request_func_input.ignore_eos
    if request_func_input.logprobs is not None:
        payload["logprobs"] = request_func_input.logprobs
    if request_func_input.extra_body:
        payload.update(request_func_input.extra_body)

    headers = {"Content-Type": "application/json"}
    output = RequestFuncOutput()
    output.prompt_len = request_func_input.prompt_len
    st = time.perf_counter()

    try:
        async with session.post(
            url=api_url, json=payload, headers=headers
        ) as response:
            raw_text = await response.text()
            if response.status != 200:
                output.success = False
                output.error = raw_text or (response.reason or "")
                if pbar:
                    pbar.update(1)
                return output

            try:
                data = json.loads(raw_text)
            except json.JSONDecodeError as e:
                output.success = False
                output.error = f"Invalid JSON: {e}"
                if pbar:
                    pbar.update(1)
                return output

            body, metrics, err = _parse_sllm_json_body(data)
            if err:
                output.success = False
                output.error = err
                if pbar:
                    pbar.update(1)
                return output

            assert body is not None
            choices = body.get("choices") or []
            generated_text = ""
            if choices:
                generated_text = choices[0].get("text") or ""

            usage = body.get("usage") or {}
            completion_tokens = int(usage.get("completion_tokens") or 0)

            output.generated_text = generated_text
            output.latency = float(metrics.get("e2e", time.perf_counter() - st)) if metrics else time.perf_counter() - st
            output.ttft = float(metrics.get("ttft", 0.0) or 0.0) if metrics else 0.0
            output.tpot = float(metrics.get("tpot", 0.0) or 0.0) if metrics else 0.0
            raw_itls = metrics.get("itls") if metrics else None
            if isinstance(raw_itls, list):
                output.itl = [float(x) for x in raw_itls]
            output.output_tokens = completion_tokens or int(
                metrics.get("output_length", 0) if metrics else 0
            )
            output.success = True
    except Exception:
        output.success = False
        output.error = "".join(traceback.format_exception(*sys.exc_info()))

    if pbar:
        pbar.update(1)
    return output


ASYNC_REQUEST_FUNCS = {
    "sllm": async_request_sllm_chat_completions,
}

OPENAI_COMPATIBLE_BACKENDS = ["sllm"]
