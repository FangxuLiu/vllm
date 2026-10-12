# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram layers and n-gram hashing for HYV3."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

ENGRAM_NO_HASH_ID = -1


def _flatten_head_vocab_sizes(config, layer_id: int) -> list[int]:
    values = getattr(config, "engram_layer_head_vocab_sizes", None)
    if values is None:
        raise ValueError(
            "HYV3 Engram requires `engram_layer_head_vocab_sizes` in config.json."
        )
    entry = values.get(layer_id, values.get(str(layer_id)))
    if entry is None:
        raise ValueError(f"Missing Engram head vocabulary sizes for layer {layer_id}.")
    if entry and isinstance(entry[0], (list, tuple)):
        flat = [int(size) for group in entry for size in group]
    else:
        flat = [int(size) for size in entry]
    expected = (int(config.engram_max_ngram_size) - 1) * int(
        config.engram_n_head_per_ngram
    )
    if len(flat) != expected:
        raise ValueError(
            f"Layer {layer_id} has {len(flat)} Engram hash heads; expected {expected}."
        )
    return flat


def _resolve_vocab_map_path(model_path: str, configured_path: str | None) -> str:
    if configured_path:
        path = os.path.expanduser(configured_path)
        if not os.path.isabs(path):
            path = os.path.join(model_path, path)
    else:
        path = os.path.join(model_path, "engram_vocab_map.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "HYV3 Engram requires a frozen vocabulary map, but it was not found "
            f"at {path}."
        )
    return path


def _load_frozen_token_map(path: str, seed: int) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as data:
        lookup = np.asarray(data["lookup_table"], dtype=np.int64)
        metadata = json.loads(bytes(data["metadata"]).decode("utf-8"))

    source_len = int(metadata.get("source_tokenizer_len", -1))
    if source_len != lookup.shape[0]:
        raise ValueError(
            "Corrupt Engram vocabulary map: source_tokenizer_len="
            f"{source_len}, lookup rows={lookup.shape[0]}."
        )
    frozen_seed = metadata.get("engram_seed")
    if frozen_seed is not None and int(frozen_seed) != seed:
        raise ValueError(
            f"Engram seed mismatch: map uses {frozen_seed}, config uses {seed}."
        )
    stored_sha = metadata.get("lookup_sha256")
    actual_sha = hashlib.sha256(
        np.ascontiguousarray(lookup, dtype=np.int64).tobytes()
    ).hexdigest()
    if stored_sha is not None and stored_sha != actual_sha:
        raise ValueError("Engram vocabulary map failed its SHA256 integrity check.")
    return lookup, metadata


class HYV3EngramHash(nn.Module):
    """GPU n-gram hashing using the runner-provided request lookback window."""

    def __init__(self, config, model_path: str) -> None:
        super().__init__()
        self.layer_ids = tuple(int(x) for x in config.engram_layer_ids)
        self.max_ngram_size = int(config.engram_max_ngram_size)
        self.n_heads = int(config.engram_n_head_per_ngram)
        self.pad_token_id = int(config.engram_pad_id)
        self.seed = int(getattr(config, "engram_seed", 0))
        self.lookback_depth = self.max_ngram_size - 1

        vocab_map_path = _resolve_vocab_map_path(
            model_path, getattr(config, "engram_vocab_map_path", None)
        )
        token_map, metadata = _load_frozen_token_map(vocab_map_path, self.seed)
        if not 0 <= self.pad_token_id < token_map.shape[0]:
            raise ValueError(
                f"Engram pad token {self.pad_token_id} is outside the frozen map."
            )
        self.compressed_vocab_size = int(
            metadata.get("compressed_vocab_size", int(token_map.max()) + 1)
        )
        self.pad_id = int(token_map[self.pad_token_id])

        head_vocab_sizes = [
            _flatten_head_vocab_sizes(config, layer_id) for layer_id in self.layer_ids
        ]
        multipliers = []
        max_long = np.iinfo(np.int64).max
        half_bound = max(1, (max_long // self.compressed_vocab_size) // 2)
        for layer_id in self.layer_ids:
            rng = np.random.default_rng(self.seed + 10007 * layer_id)
            values = rng.integers(
                low=0,
                high=half_bound,
                size=(self.max_ngram_size,),
                dtype=np.int64,
            )
            multipliers.append(values * 2 + 1)

        self.register_buffer(
            "token_map",
            torch.from_numpy(token_map.astype(np.int32, copy=False)),
            persistent=False,
        )
        self.register_buffer(
            "multipliers",
            torch.from_numpy(np.stack(multipliers)),
            persistent=False,
        )
        self.register_buffer(
            "head_vocab_sizes",
            torch.tensor(head_vocab_sizes, dtype=torch.int64),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        lookback_token_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        num_reqs = query_start_loc.numel() - 1
        if lookback_token_ids.shape != (num_reqs, self.lookback_depth):
            raise ValueError(
                "HYV3 Engram lookback shape mismatch: got "
                f"{tuple(lookback_token_ids.shape)}, expected "
                f"({num_reqs}, {self.lookback_depth})."
            )

        num_tokens = input_ids.shape[0]
        token_indices = torch.arange(num_tokens, device=input_ids.device)
        actual_tokens = token_indices < query_start_loc[-1]
        req_ids = torch.searchsorted(query_start_loc[1:], token_indices, right=True)
        req_ids = req_ids.clamp_max(num_reqs - 1)
        chunk_starts = query_start_loc.index_select(0, req_ids)
        local_positions = token_indices - chunk_starts

        compressed: list[torch.Tensor] = []
        for shift in range(self.max_ngram_size):
            in_chunk = local_positions >= shift
            current_indices = (token_indices - shift).clamp_min(0)
            current_tokens = input_ids.index_select(0, current_indices)

            lookback_cols = shift - local_positions - 1
            safe_cols = lookback_cols.clamp(0, self.lookback_depth - 1)
            historical_tokens = lookback_token_ids[req_ids, safe_cols]
            has_history = (~in_chunk) & (historical_tokens >= 0)
            source_tokens = torch.where(
                in_chunk,
                current_tokens,
                torch.where(
                    has_history,
                    historical_tokens,
                    historical_tokens.new_full((), self.pad_token_id),
                ),
            ).long()
            safe_tokens = source_tokens.clamp_min(0)
            compressed.append(self.token_map[safe_tokens].long())

        outputs: dict[int, torch.Tensor] = {}
        for layer_index, layer_id in enumerate(self.layer_ids):
            rolling = compressed[0] * self.multipliers[layer_index, 0]
            columns = []
            for shift in range(1, self.max_ngram_size):
                rolling = torch.bitwise_xor(
                    rolling,
                    compressed[shift] * self.multipliers[layer_index, shift],
                )
                primes = self.head_vocab_sizes[
                    layer_index,
                    (shift - 1) * self.n_heads : shift * self.n_heads,
                ]
                columns.append(torch.remainder(rolling[:, None], primes[None, :]))
            hashes = torch.cat(columns, dim=-1)
            hashes = hashes.masked_fill(
                ((input_ids == self.pad_token_id) | ~actual_tokens).unsqueeze(-1),
                ENGRAM_NO_HASH_ID,
            )
            outputs[layer_id] = hashes
        return outputs


class HYV3EngramMultiHeadEmbedding(nn.Module):
    def __init__(
        self,
        head_vocab_sizes: Sequence[int],
        embedding_dim: int,
        use_fp8_qdq: bool,
    ) -> None:
        super().__init__()
        self.num_heads = len(head_vocab_sizes)
        self.embedding_dim = int(embedding_dim)
        self.use_fp8_qdq = bool(use_fp8_qdq)
        self.head_vocab_sizes = tuple(int(x) for x in head_vocab_sizes)
        offsets = [0]
        for size in self.head_vocab_sizes[:-1]:
            offsets.append(offsets[-1] + size)
        self.register_buffer(
            "offsets", torch.tensor(offsets, dtype=torch.int64), persistent=False
        )
        self.embedding = nn.Embedding(sum(self.head_vocab_sizes), self.embedding_dim)

    @staticmethod
    def _fp8_fake_quantize(values: torch.Tensor) -> torch.Tensor:
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("HYV3 Engram FP8 QDQ requires float8_e4m3fn support.")
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        scale = values.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / fp8_max
        quantized = (values / scale).clamp(-fp8_max, fp8_max)
        return quantized.to(torch.float8_e4m3fn).to(values.dtype) * scale

    def forward(self, hash_ids: torch.Tensor) -> torch.Tensor:
        no_hash = hash_ids == ENGRAM_NO_HASH_ID
        physical_ids = hash_ids.clamp_min(0) + self.offsets
        output = self.embedding(physical_ids)
        output = output.masked_fill(no_hash.unsqueeze(-1), 0)
        if self.use_fp8_qdq:
            output = self._fp8_fake_quantize(output)
            output = output.masked_fill(no_hash.unsqueeze(-1), 0)
        return output


class HYV3EngramLayer(nn.Module):
    """HYV3 Engram residual matching the exported Transformers implementation."""

    def __init__(self, config, layer_id: int, prefix: str = "") -> None:
        super().__init__()
        del prefix
        self.hidden_size = int(config.hidden_size)
        self.total_heads = (int(config.engram_max_ngram_size) - 1) * int(
            config.engram_n_head_per_ngram
        )
        per_head_dim = int(config.engram_n_embed_per_ngram) // int(
            config.engram_n_head_per_ngram
        )
        head_vocab_sizes = _flatten_head_vocab_sizes(config, layer_id)
        self.multi_head_embedding = HYV3EngramMultiHeadEmbedding(
            head_vocab_sizes,
            per_head_dim,
            bool(getattr(config, "engram_fp8_embedding", False)),
        )
        self.lookup_per_head_norms = nn.ModuleList(
            [nn.RMSNorm(per_head_dim) for _ in range(self.total_heads)]
        )
        engram_hidden_size = self.total_heads * per_head_dim
        self.value_proj = nn.Linear(engram_hidden_size, self.hidden_size)
        self.key_projs = nn.ModuleList(
            [nn.Linear(engram_hidden_size, self.hidden_size)]
        )
        self.norm_key = nn.ModuleList([nn.RMSNorm(self.hidden_size)])
        self.norm_query = nn.ModuleList([nn.RMSNorm(self.hidden_size)])
        self.unified_rmsnorm = nn.RMSNorm(self.hidden_size, elementwise_affine=False)
        self.unified_pre_gamma = nn.Parameter(torch.ones(self.hidden_size))
        self.unified_per_hc_scales = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.hidden_size))]
        )
        self.zero_per_hc_scales = bool(
            getattr(config, "engram_zero_and_freeze_per_hc_scales", False)
        )

    def forward(
        self, hidden_states: torch.Tensor, hash_ids: torch.Tensor
    ) -> torch.Tensor:
        embeddings = self.multi_head_embedding(hash_ids).to(hidden_states.dtype)
        embeddings = torch.stack(
            [
                norm(embeddings[:, head])
                for head, norm in enumerate(self.lookup_per_head_norms)
            ],
            dim=1,
        ).flatten(start_dim=-2)

        key = self.norm_key[0](self.key_projs[0](embeddings))
        query = self.norm_query[0](hidden_states)
        device_type = hidden_states.device.type
        context = (
            torch.autocast(device_type=device_type, enabled=False)
            if device_type in ("cpu", "cuda")
            else contextlib.nullcontext()
        )
        with context:
            raw_gate = (key * query).sum(dim=-1)
        gate = raw_gate * (1.0 / math.sqrt(self.hidden_size))
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gate = gate.sigmoid().unsqueeze(-1)

        value = self.unified_rmsnorm(self.value_proj(embeddings))
        main_signal = self.unified_pre_gamma * value
        output = gate * main_signal
        branch = self.unified_per_hc_scales[0] * main_signal
        if self.zero_per_hc_scales:
            branch = branch * 0.0
        output = output + branch
        no_engram = (hash_ids == ENGRAM_NO_HASH_ID).all(dim=-1)
        return output.masked_fill(no_engram.unsqueeze(-1), 0)
