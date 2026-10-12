# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate HYV3 Engram hashing and basic vLLM generation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prompts() -> list[str]:
    return [
        "Hello, please introduce yourself briefly.",
        "请简要介绍一下你自己。",
        "The capital of France is",
        "人工智能的发展历史可以追溯到",
    ]


def _input_ids(model_path: str) -> list[list[int]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    sequences = []
    for prompt in _prompts():
        if getattr(tokenizer, "chat_template", None):
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered = prompt
        sequences.append(tokenizer(rendered, add_special_tokens=False).input_ids)
    return sequences


def validate_hash(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoConfig

    from vllm.model_executor.models.hy_v3_engram import HYV3EngramHash

    model_path = str(Path(args.model).resolve())
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    reference = _load_module(
        "hyv3_engram_reference_utils", Path(model_path) / "engram_utils.py"
    )
    map_path = Path(model_path) / config.engram_vocab_map_path
    preprocessor = reference.EngramPreprocessor(
        engram_vocab_size=list(config.engram_vocab_size),
        max_ngram_size=config.engram_max_ngram_size,
        n_head_per_ngram=config.engram_n_head_per_ngram,
        layer_ids=list(config.engram_layer_ids),
        tokenizer_name_or_path=None,
        pad_id=config.engram_pad_id,
        seed=config.engram_seed,
        vocab_map_path=str(map_path),
        layer_head_vocab_sizes=config.engram_layer_head_vocab_sizes,
    )
    hasher = HYV3EngramHash(config, model_path).to(args.device)

    sequences = _input_ids(model_path)
    reference_hashes = [
        preprocessor(torch.tensor(sequence).unsqueeze(0)) for sequence in sequences
    ]
    scenarios = {
        "prefill": [0 for _ in sequences],
        "chunked": [min(args.chunk_start, len(seq) - 1) for seq in sequences],
        "decode": [max(0, len(seq) - 1) for seq in sequences],
    }
    results: dict[str, Any] = {}
    for scenario, starts in scenarios.items():
        flat_ids: list[int] = []
        flat_positions: list[int] = []
        lookback: list[list[int]] = []
        query_start_loc = [0]
        expected: dict[int, list[torch.Tensor]] = {
            int(layer_id): [] for layer_id in config.engram_layer_ids
        }
        for sequence, hashes, start in zip(sequences, reference_hashes, starts):
            chunk = sequence[start:]
            flat_ids.extend(chunk)
            flat_positions.extend(range(start, len(sequence)))
            lookback.append(
                [
                    sequence[start - offset - 1] if start - offset - 1 >= 0 else -1
                    for offset in range(config.engram_max_ngram_size - 1)
                ]
            )
            query_start_loc.append(query_start_loc[-1] + len(chunk))
            for layer_id in expected:
                expected[layer_id].append(hashes[layer_id][:, start:].squeeze(0))

        actual = hasher(
            torch.tensor(flat_ids, dtype=torch.int64, device=args.device),
            torch.tensor(flat_positions, dtype=torch.int64, device=args.device),
            torch.tensor(lookback, dtype=torch.int32, device=args.device),
            torch.tensor(query_start_loc, dtype=torch.int32, device=args.device),
        )
        for layer_id, parts in expected.items():
            torch.testing.assert_close(
                actual[layer_id].cpu(), torch.cat(parts), rtol=0, atol=0
            )
        results[scenario] = {
            "tokens": len(flat_ids),
            "requests": len(sequences),
            "layers": list(expected),
        }

    print(json.dumps({"status": "PASS", "hash": results}, ensure_ascii=False))


def validate_generate(args: argparse.Namespace) -> None:
    if args.disable_engram:
        os.environ["VLLM_HYV3_ENGRAM_DISABLE"] = "1"
    else:
        os.environ.pop("VLLM_HYV3_ENGRAM_DISABLE", None)
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        max_num_seqs=len(_prompts()),
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_log_stats=True,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
    )
    outputs = llm.generate(_prompts(), sampling)
    result = {
        "status": "PASS",
        "engram": "OFF" if args.disable_engram else "ON",
        "outputs": [
            {
                "prompt": output.prompt,
                "token_ids": list(output.outputs[0].token_ids),
                "text": output.outputs[0].text,
                "finish_reason": output.outputs[0].finish_reason,
            }
            for output in outputs
        ],
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    hash_parser = subparsers.add_parser("hash")
    hash_parser.add_argument("--model", required=True)
    hash_parser.add_argument("--device", default="cuda")
    hash_parser.add_argument("--chunk-start", type=int, default=3)
    hash_parser.set_defaults(func=validate_hash)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--model", required=True)
    generate_parser.add_argument("--disable-engram", action="store_true")
    generate_parser.add_argument("--max-model-len", type=int, default=2048)
    generate_parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    generate_parser.add_argument("--max-tokens", type=int, default=32)
    generate_parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    generate_parser.add_argument("--output")
    generate_parser.set_defaults(func=validate_generate)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
