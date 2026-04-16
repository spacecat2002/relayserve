# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Portions adapted from the vLLM project
r"""Benchmark SLLM HTTP serving (vLLM ``bench serve``-style driver).

Start the SLLM API (e.g. ``sllm`` CLI / Ray controller + uvicorn), register a model,
then run from the *relayserve* repo root::

    python -m sllm.benchmarks.serve_sllm \\
        --model <registered_model_name> \\
        --tokenizer <hf_model_id_for_token_counting> \\
        --host 127.0.0.1 --port 8343 \\
        --endpoint /v1/chat/completions \\
        --num-prompts 100 --request-rate 10

Multi-model + per-model SLO (JSON values in milliseconds)::

    python -m sllm.benchmarks.serve_sllm \\
        --models llama-7b qwen-7b --model-mix round_robin \\
        --tokenizer-config tokmap.json \\
        --model-slo-config slos.json \\
        --num-prompts 200 --request-rate 20

This driver uses **JSON** ``/v1/chat/completions`` (same shape as ``app_lib``:
``model``, ``prompt``, ``max_tokens``, …) and parses the response body
``[completion_dict, latency_metrics]`` returned by the SLLM stack.

Only ``--dataset-name random`` is built in; extend ``dataset_random.py`` for
other datasets or point the upstream vLLM bench at an HTTP proxy.
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Any, Literal, Optional

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from sllm.benchmarks.dataset_random import SampleRequest, get_samples
from sllm.benchmarks.endpoint_request_func import (
    ASYNC_REQUEST_FUNCS,
    OPENAI_COMPATIBLE_BACKENDS,
    RequestFuncInput,
    RequestFuncOutput,
)
from sllm.benchmarks.ready_checker import wait_for_endpoint
from sllm.benchmarks.traffic import TaskType, calculate_metrics, get_request
from sllm.benchmarks.utils import convert_to_pytorch_benchmark_format, write_to_json


def check_goodput_args(args: argparse.Namespace) -> dict[str, float]:
    goodput_config_dict: dict[str, float] = {}
    valid_names = ["ttft", "tpot", "e2el"]
    if args.goodput:
        if getattr(args, "model_slo_config", None):
            print(
                "Note: --goodput is ignored when --model-slo-config is set "
                "(SLO is per model only).",
                file=sys.stderr,
            )
        else:
            goodput_config_dict = parse_goodput(args.goodput)
            for slo_name, slo_val in goodput_config_dict.items():
                if slo_name not in valid_names:
                    raise ValueError(
                        f"Invalid metric name {slo_name}. Expected one of {valid_names}."
                    )
                if slo_val < 0:
                    raise ValueError(f"Invalid SLO value {slo_name}: {slo_val}")
    return goodput_config_dict


def parse_model_weights(s: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid --model-weights fragment: {part!r} (use name:weight)")
        name, w = part.rsplit(":", 1)
        out[name.strip()] = float(w)
    return out


def load_model_slo_config(path: str) -> dict[str, dict[str, float]]:
    valid = frozenset({"ttft", "tpot", "e2el"})
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("model SLO JSON must be an object: {model_name: {ttft: ms, ...}, ...}")
    out: dict[str, dict[str, float]] = {}
    for model_name, spec in raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"SLO for {model_name!r} must be an object")
        inner: dict[str, float] = {}
        for k, v in spec.items():
            if k not in valid:
                raise ValueError(f"Unknown SLO key {k!r} for model {model_name}")
            inner[k] = float(v)
        out[str(model_name)] = inner
    return out


def load_tokenizer_config(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("tokenizer config must be {registered_model: hf_tokenizer_id, ...}")
    return {str(k): str(v) for k, v in raw.items()}


def parse_goodput(slo_pairs: list[str]) -> dict[str, float]:
    goodput_config_dict: dict[str, float] = {}
    try:
        for slo_pair in slo_pairs:
            slo_name, slo_val = slo_pair.split(":")
            goodput_config_dict[slo_name] = float(slo_val)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            'Use "KEY:VALUE" pairs in milliseconds (e.g. ttft:200 e2el:5000).'
        ) from err
    return goodput_config_dict


def save_to_pytorch_benchmark_format(
    args: argparse.Namespace, results: dict[str, Any], file_name: str
) -> None:
    metrics = [
        "median_ttft_ms",
        "mean_ttft_ms",
        "std_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "std_tpot_ms",
        "p99_tpot_ms",
        "median_itl_ms",
        "mean_itl_ms",
        "std_itl_ms",
        "p99_itl_ms",
    ]
    ignored_metrics = ["ttfts", "itls", "generated_texts", "errors"]
    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={k: [results[k]] for k in metrics if k in results},
        extra_info={
            k: results[k]
            for k in results
            if k not in metrics and k not in ignored_metrics
        },
    )
    if pt_records:
        pt_file = f"{os.path.splitext(file_name)[0]}.pytorch.json"
        write_to_json(pt_file, pt_records)


def add_dataset_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="random",
        choices=["random"],
        help="Only `random` is bundled; extend dataset_random.py for more.",
    )
    random_group = parser.add_argument_group("random dataset options")
    random_group.add_argument("--random-input-len", type=int, default=1024)
    random_group.add_argument("--random-output-len", type=int, default=128)
    random_group.add_argument(
        "--random-range-ratio",
        type=float,
        default=0.0,
        help="Must be in [0, 1).",
    )
    random_group.add_argument("--random-prefix-len", type=int, default=0)


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    add_dataset_parser(parser)
    parser.add_argument(
        "--endpoint-type",
        type=str,
        default="sllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Prefix for saved result filenames.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Full base URL (overrides host/port), e.g. http://127.0.0.1:8343",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8343)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/chat/completions",
        help="Must match SLLM route (default: OpenAI-style chat path).",
    )
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Single registered model (alternative to --models).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Multiple registered SLLM models; combine with --model-mix / --model-weights.",
    )
    parser.add_argument(
        "--model-mix",
        type=str,
        default="round_robin",
        choices=["round_robin", "random"],
        help="How to assign requests to models when using --models.",
    )
    parser.add_argument(
        "--model-weights",
        type=str,
        default=None,
        help='With --model-mix random: comma weights, e.g. ``llama:0.7,qwen:0.3``.',
    )
    parser.add_argument(
        "--model-slo-config",
        type=str,
        default=None,
        help="JSON file: per-model SLO in ms, "
        'e.g. ``{"m1":{"ttft":500,"e2el":8000},"m2":{"ttft":200}}``. '
        "Drives overall & per-model goodput when set.",
    )
    parser.add_argument(
        "--tokenizer-config",
        type=str,
        default=None,
        help="JSON file mapping registered model name → HF tokenizer id/path.",
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="If set, overrides the JSON ``model`` field sent to the server.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="HF model id or path for token counting (single-model or default for multi).",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--logprobs", type=int, default=None)
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Target requests/s (inf = fire all at t=0).",
    )
    parser.add_argument("--burstiness", type=float, default=1.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Not supported for SLLM (ignored).",
    )
    parser.add_argument("--save-result", action="store_true")
    parser.add_argument("--save-detailed", action="store_true")
    parser.add_argument("--append-result", action="store_true")
    parser.add_argument("--metadata", metavar="KEY=VALUE", nargs="*")
    parser.add_argument("--result-dir", type=str, default=None)
    parser.add_argument("--result-filename", type=str, default=None)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument(
        "--percentile-metrics",
        type=str,
        default="ttft,tpot,itl",
    )
    parser.add_argument("--metric-percentiles", type=str, default="99")
    parser.add_argument(
        "--goodput",
        nargs="+",
        required=False,
        help="Global SLO (ms) as ttft:200 e2el:5000 — ignored if --model-slo-config is set.",
    )
    parser.add_argument(
        "--request-id-prefix",
        type=str,
        default="benchmark-serving",
    )

    sampling_group = parser.add_argument_group("sampling parameters")
    sampling_group.add_argument("--top-p", type=float, default=None)
    sampling_group.add_argument("--top-k", type=int, default=None)
    sampling_group.add_argument("--min-p", type=float, default=None)
    sampling_group.add_argument("--temperature", type=float, default=None)

    parser.add_argument(
        "--ramp-up-strategy",
        type=str,
        default=None,
        choices=["linear", "exponential"],
    )
    parser.add_argument("--ramp-up-start-rps", type=int, default=None)
    parser.add_argument("--ramp-up-end-rps", type=int, default=None)
    parser.add_argument("--ready-check-timeout-sec", type=int, default=600)
    parser.add_argument(
        "--skip-initial-test",
        action="store_true",
        help="Skip readiness probe (first request).",
    )


async def benchmark(
    endpoint_type: str,
    api_url: str,
    base_url: str,
    model_name: Optional[str],
    tokenizer: PreTrainedTokenizerBase,
    tokenizers_by_model: dict[str, PreTrainedTokenizerBase],
    input_requests: list[SampleRequest],
    logprobs: Optional[int],
    request_rate: float,
    burstiness: float,
    disable_tqdm: bool,
    profile: bool,
    selected_percentile_metrics: list[str],
    selected_percentiles: list[float],
    ignore_eos: bool,
    goodput_config_dict: dict[str, float],
    model_slo_config: Optional[dict[str, dict[str, float]]],
    max_concurrency: Optional[int],
    lora_modules: Optional[list[str]],
    extra_body: Optional[dict],
    ramp_up_strategy: Optional[Literal["linear", "exponential"]] = None,
    ramp_up_start_rps: Optional[int] = None,
    ramp_up_end_rps: Optional[int] = None,
    ready_check_timeout_sec: int = 600,
    skip_initial_test: bool = False,
) -> dict[str, Any]:
    del base_url, lora_modules
    task_type = (
        TaskType.EMBEDDING
        if api_url.endswith("/v1/embeddings")
        else TaskType.GENERATION
    )
    if task_type == TaskType.EMBEDDING:
        raise ValueError("SLLM benchmark driver does not support embeddings yet.")

    request_func = ASYNC_REQUEST_FUNCS[endpoint_type]

    if profile:
        print("Note: --profile is not supported for SLLM; ignoring.", file=sys.stderr)

    connector = aiohttp.TCPConnector(
        limit=max_concurrency or 0,
        limit_per_host=max_concurrency or 0,
        ttl_dns_cache=300,
        use_dns_cache=True,
        keepalive_timeout=60,
        enable_cleanup_closed=True,
        force_close=False,
        ssl=("https://" in api_url),
    )
    session = aiohttp.ClientSession(
        connector=connector,
        trust_env=True,
        timeout=aiohttp.ClientTimeout(total=6 * 60 * 60),
    )

    test_prompt = ""
    test_prompt_len = 0
    test_output_len = 0
    test_mm_content = None

    if not skip_initial_test:
        print("Starting initial single prompt test run...")
        test_prompt = str(input_requests[0].prompt)
        test_prompt_len = input_requests[0].prompt_len
        test_output_len = input_requests[0].expected_output_len
        test_mm_content = input_requests[0].multi_modal_data
        if test_mm_content is not None:
            raise ValueError("Multimodal requests are not supported in this driver.")

        probe_model = input_requests[0].target_model
        if not probe_model:
            raise ValueError("Internal error: first request missing target_model")
        test_input = RequestFuncInput(
            model=probe_model,
            model_name=model_name,
            prompt=test_prompt,
            api_url=api_url,
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            logprobs=logprobs,
            multi_modal_content=test_mm_content,
            ignore_eos=ignore_eos,
            extra_body=extra_body,
        )
        test_output = await wait_for_endpoint(
            request_func,
            test_input,
            session,
            timeout_seconds=ready_check_timeout_sec,
        )
        if not test_output.success:
            raise ValueError(
                "Initial test run failed. Check --model(s), server URL, and registration. "
                f"Error: {test_output.error}"
            )
        print("Initial test run completed. Starting main benchmark run...")
    else:
        print("Skipping initial test run. Starting main benchmark run...")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_request_func(
        request_func_input: RequestFuncInput,
        sess: aiohttp.ClientSession,
        bar: Any,
    ) -> RequestFuncOutput:
        if semaphore is None:
            return await request_func(request_func_input=request_func_input, session=sess, pbar=bar)
        async with semaphore:
            return await request_func(request_func_input=request_func_input, session=sess, pbar=bar)

    benchmark_start_time = time.perf_counter()
    tasks: list[asyncio.Task[RequestFuncOutput]] = []
    async for request, _current_request_rate in get_request(
        input_requests,
        request_rate,
        burstiness,
        ramp_up_strategy,
        ramp_up_start_rps,
        ramp_up_end_rps,
    ):
        prompt, prompt_len, output_len, mm_content, request_id, target_model = (
            request.prompt,
            request.prompt_len,
            request.expected_output_len,
            request.multi_modal_data,
            request.request_id,
            request.target_model,
        )
        if mm_content is not None:
            raise ValueError("Multimodal requests are not supported.")
        if not target_model:
            raise ValueError("Internal error: SampleRequest missing target_model")
        request_func_input = RequestFuncInput(
            model=target_model,
            model_name=model_name,
            prompt=str(prompt),
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            logprobs=logprobs,
            multi_modal_content=mm_content,
            ignore_eos=ignore_eos,
            extra_body=extra_body,
            request_id=request_id,
        )
        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input, session, pbar)
            )
        )
    outputs: list[RequestFuncOutput] = await asyncio.gather(*tasks)
    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, actual_output_lens, slo_report = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentiles=selected_percentiles,
        goodput_config_dict=goodput_config_dict,
        tokenizer_by_model=tokenizers_by_model,
        model_slo_config=model_slo_config,
    )

    print("{s:{c}^{n}}".format(s=" Serving Benchmark Result (SLLM) ", n=50, c="="))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    if max_concurrency is not None:
        print("{:<40} {:<10}".format("Maximum request concurrency:", max_concurrency))
    if request_rate != float("inf"):
        print("{:<40} {:<10.2f}".format("Request rate configured (RPS):", request_rate))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):", metrics.request_throughput))
    if goodput_config_dict or model_slo_config:
        print("{:<40} {:<10.2f}".format("Request goodput (req/s):", metrics.request_goodput))
        print(
            "{:<40} {:<10.4f}".format(
                "Overall SLO satisfied rate:",
                slo_report["overall_slo_satisfied_rate"],
            )
        )
        print("Per-model SLO (completed / satisfied / rate):")
        for mk, mv in slo_report["goodput_by_model"].items():
            print(
                f"  {mk!r}: completed={mv['completed']}, "
                f"slo_satisfied={mv['slo_satisfied']}, "
                f"rate={mv['slo_satisfied_rate']:.4f}"
            )
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):", metrics.output_throughput))
    print("{:<40} {:<10.2f}".format("Total Token throughput (tok/s):", metrics.total_token_throughput))

    result: dict[str, Any] = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "request_goodput": metrics.request_goodput if goodput_config_dict else None,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
        "request_target_models": [getattr(r, "target_model", None) for r in input_requests],
        "slo_report": slo_report,
    }

    def process_one_metric(metric_attribute_name: str, metric_name: str, metric_header: str) -> None:
        if metric_attribute_name not in selected_percentile_metrics:
            return
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c="-"))
        print(
            "{:<40} {:<10.2f}".format(
                f"Mean {metric_name} (ms):",
                getattr(metrics, f"mean_{metric_attribute_name}_ms"),
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                f"Median {metric_name} (ms):",
                getattr(metrics, f"median_{metric_attribute_name}_ms"),
            )
        )
        result[f"mean_{metric_attribute_name}_ms"] = getattr(
            metrics, f"mean_{metric_attribute_name}_ms"
        )
        result[f"median_{metric_attribute_name}_ms"] = getattr(
            metrics, f"median_{metric_attribute_name}_ms"
        )
        result[f"std_{metric_attribute_name}_ms"] = getattr(
            metrics, f"std_{metric_attribute_name}_ms"
        )
        for p, value in getattr(metrics, f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):", value))
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tpot", "TPOT", "Time per Output Token (excl. 1st token)")
    process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")
    print("=" * 50)

    await session.close()
    return result


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.ramp_up_strategy is not None:
        if args.request_rate != float("inf"):
            raise ValueError("With ramp-up, omit --request-rate (use ramp-up RPS only).")
        if args.ramp_up_start_rps is None or args.ramp_up_end_rps is None:
            raise ValueError("Ramp-up requires --ramp-up-start-rps and --ramp-up-end-rps.")
        if args.ramp_up_start_rps < 0 or args.ramp_up_end_rps < 0:
            raise ValueError("Ramp-up RPS must be non-negative.")
        if args.ramp_up_start_rps > args.ramp_up_end_rps:
            raise ValueError("Ramp-up start RPS must be <= end RPS.")
        if args.ramp_up_strategy == "exponential" and args.ramp_up_start_rps == 0:
            raise ValueError("Exponential ramp-up cannot start at 0 RPS.")

    if args.base_url is not None:
        api_url = f"{args.base_url.rstrip('/')}{args.endpoint}"
        base_url = args.base_url.rstrip("/")
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        base_url = f"http://{args.host}:{args.port}"

    if args.models:
        models_list = list(args.models)
    elif args.model:
        models_list = [args.model]
    else:
        raise ValueError("Specify --model for a single workload or --models for multiple.")

    model_slo_config: Optional[dict[str, dict[str, float]]] = None
    if args.model_slo_config:
        model_slo_config = load_model_slo_config(args.model_slo_config)
        for m in models_list:
            if m not in model_slo_config:
                print(
                    f"Warning: model {m!r} missing from SLO JSON; "
                    "that model has no latency SLO (always counts as satisfied).",
                    file=sys.stderr,
                )

    tok_cfg = load_tokenizer_config(args.tokenizer_config) if args.tokenizer_config else {}
    default_tid = args.tokenizer if args.tokenizer is not None else models_list[0]
    tokenizers_by_model: dict[str, PreTrainedTokenizerBase] = {}
    for m in models_list:
        tid = tok_cfg.get(m, default_tid)
        tokenizers_by_model[m] = AutoTokenizer.from_pretrained(
            tid, trust_remote_code=args.trust_remote_code
        )

    args.models_list = models_list
    args.tokenizers_by_model = tokenizers_by_model
    args.model_weights_dict = (
        parse_model_weights(args.model_weights) if args.model_weights else None
    )
    if args.model_weights_dict:
        for k in args.model_weights_dict:
            if k not in models_list:
                raise ValueError(
                    f"--model-weights key {k!r} must appear in --models {models_list!r}"
                )

    tokenizer = tokenizers_by_model[models_list[0]]
    tokenizer_id = default_tid

    input_requests = get_samples(args, tokenizer)
    goodput_config_dict = check_goodput_args(args)

    sampling_params = {
        k: v
        for k, v in {
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "temperature": args.temperature,
        }.items()
        if v is not None
    }
    if sampling_params and args.endpoint_type not in OPENAI_COMPATIBLE_BACKENDS:
        raise ValueError("Sampling params require an OpenAI-compatible backend key.")
    if "temperature" not in sampling_params:
        sampling_params["temperature"] = 0.0

    gc.collect()
    gc.freeze()

    benchmark_result = await benchmark(
        endpoint_type=args.endpoint_type,
        api_url=api_url,
        base_url=base_url,
        model_name=args.served_model_name,
        tokenizer=tokenizer,
        tokenizers_by_model=tokenizers_by_model,
        input_requests=input_requests,
        logprobs=args.logprobs,
        request_rate=args.request_rate,
        burstiness=args.burstiness,
        disable_tqdm=args.disable_tqdm,
        profile=args.profile,
        selected_percentile_metrics=args.percentile_metrics.split(","),
        selected_percentiles=[float(p) for p in args.metric_percentiles.split(",")],
        ignore_eos=args.ignore_eos,
        goodput_config_dict=goodput_config_dict,
        model_slo_config=model_slo_config,
        max_concurrency=args.max_concurrency,
        lora_modules=None,
        extra_body=sampling_params or None,
        ramp_up_strategy=args.ramp_up_strategy,
        ramp_up_start_rps=args.ramp_up_start_rps,
        ramp_up_end_rps=args.ramp_up_end_rps,
        ready_check_timeout_sec=args.ready_check_timeout_sec,
        skip_initial_test=args.skip_initial_test,
    )

    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_json: dict[str, Any] = {
        "date": current_dt,
        "endpoint_type": args.endpoint_type,
        "label": args.label,
        "model_ids": models_list,
        "tokenizer_id": tokenizer_id,
        "num_prompts": args.num_prompts,
        "request_rate": args.request_rate if args.request_rate < float("inf") else "inf",
        "burstiness": args.burstiness,
        "max_concurrency": args.max_concurrency,
    }
    if args.metadata:
        for item in args.metadata:
            if "=" in item:
                k, v = item.split("=", 1)
                result_json[k.strip()] = v.strip()
            else:
                raise ValueError("Invalid metadata; use KEY=VALUE.")
    if args.ramp_up_strategy is not None:
        result_json["ramp_up_strategy"] = args.ramp_up_strategy
        result_json["ramp_up_start_rps"] = args.ramp_up_start_rps
        result_json["ramp_up_end_rps"] = args.ramp_up_end_rps

    result_json = {**result_json, **benchmark_result}

    if not args.save_detailed:
        for field in [
            "input_lens",
            "output_lens",
            "ttfts",
            "itls",
            "generated_texts",
            "errors",
            "request_target_models",
        ]:
            result_json.pop(field, None)

    if args.save_result or args.append_result:
        base_model_id = "-".join(m.split("/")[-1] for m in models_list)
        max_concurrency_str = (
            f"-concurrency{args.max_concurrency}" if args.max_concurrency is not None else ""
        )
        label = args.label or args.endpoint_type
        if args.ramp_up_strategy is not None:
            file_name = (
                f"{label}-ramp-up-{args.ramp_up_strategy}-{args.ramp_up_start_rps}qps-"
                f"{args.ramp_up_end_rps}qps{max_concurrency_str}-{base_model_id}-{current_dt}.json"
            )
        else:
            file_name = f"{label}-{args.request_rate}qps{max_concurrency_str}-{base_model_id}-{current_dt}.json"
        if args.result_filename:
            file_name = args.result_filename
        if args.result_dir:
            os.makedirs(args.result_dir, exist_ok=True)
            file_name = os.path.join(args.result_dir, file_name)
        mode = "a+" if args.append_result else "w"
        with open(file_name, mode=mode, encoding="utf-8") as outfile:
            if args.append_result and outfile.tell() != 0:
                outfile.write("\n")
            json.dump(result_json, outfile)
        save_to_pytorch_benchmark_format(args, result_json, file_name)

    return result_json


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="SLLM serving benchmark (vLLM bench-style).")
    add_cli_args(parser)
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
