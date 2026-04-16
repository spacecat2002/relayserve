# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Portions adapted from the vLLM project
"""Synthetic random prompts compatible with vLLM ``RandomDataset`` (text-only)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@dataclass
class SampleRequest:
    prompt: Union[str, list[str]]
    prompt_len: int
    expected_output_len: int
    multi_modal_data: Optional[Union[dict, list[dict]]] = None
    request_id: Optional[str] = None
    #: SLLM registered model name for this request (multi-model workloads).
    target_model: Optional[str] = None


def _get_prefix(rng: np.random.Generator, tokenizer: PreTrainedTokenizerBase, prefix_len: int) -> list[int]:
    if prefix_len <= 0:
        return []
    return rng.integers(0, tokenizer.vocab_size, size=prefix_len).tolist()


def _get_sampling_params(
    rng: np.random.Generator,
    num_requests: int,
    range_ratio: float,
    input_len: int,
    output_len: int,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 <= range_ratio < 1.0):
        raise ValueError("range_ratio must be in [0, 1).")
    num_special_tokens = int(tokenizer.num_special_tokens_to_add())
    real_input_len = max(0, int(input_len) - num_special_tokens)
    input_low = math.floor(real_input_len * (1 - range_ratio))
    input_high = math.ceil(real_input_len * (1 + range_ratio))
    output_low = math.floor(output_len * (1 - range_ratio))
    output_high = math.ceil(output_len * (1 + range_ratio))
    output_low = max(output_low, 1)
    if input_low > input_high:
        raise ValueError(f"Invalid input sampling interval: low={input_low} > high={input_high}")
    if output_low > output_high:
        raise ValueError(f"Invalid output sampling interval: low={output_low} > high={output_high}")

    input_lens = rng.integers(input_low, input_high + 1, size=num_requests)
    output_lens = rng.integers(output_low, output_high + 1, size=num_requests)
    offsets = rng.integers(0, tokenizer.vocab_size, size=num_requests)
    return input_lens, output_lens, offsets


def _generate_token_sequence(
    tokenizer: PreTrainedTokenizerBase,
    prefix_token_ids: list[int],
    prefix_len: int,
    vocab_size: int,
    input_len: int,
    offset: int,
    index: int,
) -> tuple[str, int]:
    inner_seq = ((offset + index + np.arange(input_len)) % vocab_size).tolist()
    token_sequence = prefix_token_ids + inner_seq
    prompt = tokenizer.decode(token_sequence)
    total_input_len = prefix_len + int(input_len)
    re_encoded_sequence = tokenizer.encode(prompt, add_special_tokens=False)[:total_input_len]
    prompt = tokenizer.decode(re_encoded_sequence)
    total_input_len = len(re_encoded_sequence)
    return prompt, total_input_len


def get_random_samples(
    tokenizer: PreTrainedTokenizerBase,
    num_requests: int,
    *,
    random_seed: int,
    request_id_prefix: str,
    random_input_len: int,
    random_output_len: int,
    random_range_ratio: float,
    random_prefix_len: int,
) -> list[SampleRequest]:
    rng = np.random.default_rng(random_seed)
    input_lens, output_lens, offsets = _get_sampling_params(
        rng, num_requests, random_range_ratio, random_input_len, random_output_len, tokenizer
    )
    prefix_token_ids = _get_prefix(rng, tokenizer, random_prefix_len)
    vocab_size = tokenizer.vocab_size
    requests: list[SampleRequest] = []
    for i in range(num_requests):
        prompt, total_input_len = _generate_token_sequence(
            tokenizer=tokenizer,
            prefix_token_ids=prefix_token_ids,
            prefix_len=random_prefix_len,
            vocab_size=vocab_size,
            input_len=int(input_lens[i]),
            offset=int(offsets[i]),
            index=i,
        )
        requests.append(
            SampleRequest(
                prompt=prompt,
                prompt_len=total_input_len,
                expected_output_len=int(output_lens[i]),
                request_id=request_id_prefix + str(i),
            )
        )
    return requests


def _assign_models_to_indices(
    models: list[str],
    num_requests: int,
    mix: str,
    weights: Optional[dict[str, float]],
    rng: np.random.Generator,
) -> list[str]:
    if len(models) == 1:
        return [models[0]] * num_requests
    if mix == "round_robin":
        return [models[i % len(models)] for i in range(num_requests)]
    if mix == "random":
        if weights:
            names = list(weights.keys())
            p = np.array([weights[m] for m in names], dtype=float)
            s = p.sum()
            if s <= 0:
                raise ValueError("model weights must sum to a positive value")
            p /= s
            idx = rng.choice(len(names), size=num_requests, p=p)
            return [names[int(j)] for j in idx]
        idx = rng.integers(0, len(models), size=num_requests)
        return [models[int(j)] for j in idx]
    raise ValueError(f"Unknown --model-mix: {mix}")


def get_multi_model_random_samples(
    tokenizers: dict[str, "PreTrainedTokenizerBase"],
    models: list[str],
    num_requests: int,
    *,
    random_seed: int,
    request_id_prefix: str,
    random_input_len: int,
    random_output_len: int,
    random_range_ratio: float,
    random_prefix_len: int,
    model_mix: str,
    model_weights: Optional[dict[str, float]],
) -> list[SampleRequest]:
    """One random prompt per request, encoded with that request's target model tokenizer."""
    rng = np.random.default_rng(random_seed)
    assigned = _assign_models_to_indices(models, num_requests, model_mix, model_weights, rng)
    requests: list[SampleRequest] = []
    per_model_rng = {m: np.random.default_rng(random_seed + hash(m) % (2**31)) for m in models}
    for i, target_model in enumerate(assigned):
        tok = tokenizers[target_model]
        mrng = per_model_rng[target_model]
        input_lens, output_lens, offsets = _get_sampling_params(
            mrng, 1, random_range_ratio, random_input_len, random_output_len, tok
        )
        prefix_token_ids = _get_prefix(mrng, tok, random_prefix_len)
        prompt, total_input_len = _generate_token_sequence(
            tokenizer=tok,
            prefix_token_ids=prefix_token_ids,
            prefix_len=random_prefix_len,
            vocab_size=tok.vocab_size,
            input_len=int(input_lens[0]),
            offset=int(offsets[0]),
            index=i,
        )
        requests.append(
            SampleRequest(
                prompt=prompt,
                prompt_len=total_input_len,
                expected_output_len=int(output_lens[0]),
                request_id=request_id_prefix + str(i),
                target_model=target_model,
            )
        )
    return requests


def get_samples(args: Any, tokenizer: PreTrainedTokenizerBase) -> list[SampleRequest]:
    if getattr(args, "dataset_name", "random") != "random":
        raise ValueError(
            "SLLM benchmark driver currently supports only --dataset-name random. "
            "For ShareGPT / HF / etc., run the upstream vLLM bench against a "
            "proxy that forwards to SLLM, or extend sllm/benchmarks/dataset_random.py."
        )
    models: list[str] = list(getattr(args, "models_list", []) or [])
    if len(models) > 1:
        tokenizers = getattr(args, "tokenizers_by_model", {})
        return get_multi_model_random_samples(
            tokenizers,
            models,
            args.num_prompts,
            random_seed=args.seed,
            request_id_prefix=getattr(args, "request_id_prefix", "benchmark-serving"),
            random_input_len=args.random_input_len,
            random_output_len=args.random_output_len,
            random_range_ratio=args.random_range_ratio,
            random_prefix_len=args.random_prefix_len,
            model_mix=getattr(args, "model_mix", "round_robin"),
            model_weights=getattr(args, "model_weights_dict", None),
        )

    m = models[0] if models else getattr(args, "model", None)
    reqs = get_random_samples(
        tokenizer,
        args.num_prompts,
        random_seed=args.seed,
        request_id_prefix=getattr(args, "request_id_prefix", "benchmark-serving"),
        random_input_len=args.random_input_len,
        random_output_len=args.random_output_len,
        random_range_ratio=args.random_range_ratio,
        random_prefix_len=args.random_prefix_len,
    )
    if m:
        for r in reqs:
            r.target_model = m
    return reqs
