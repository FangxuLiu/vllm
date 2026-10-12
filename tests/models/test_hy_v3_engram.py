# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
import math
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from vllm.model_executor.models.hy_v3_engram import (
    ENGRAM_NO_HASH_ID,
    HYV3EngramHash,
    HYV3EngramLayer,
)


def _write_vocab_map(path, lookup: np.ndarray, seed: int = 0) -> None:
    lookup = np.asarray(lookup, dtype=np.int64)
    metadata = {
        "source_tokenizer_len": int(lookup.shape[0]),
        "compressed_vocab_size": int(lookup.max()) + 1,
        "engram_seed": seed,
        "lookup_sha256": hashlib.sha256(lookup.tobytes()).hexdigest(),
    }
    np.savez(
        path,
        lookup_table=lookup.astype(np.int32),
        metadata=np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8),
    )


def _config(vocab_map_name: str, fp8: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=8,
        engram_enable=True,
        engram_layer_ids=[2],
        engram_vocab_size=[11, 17],
        engram_max_ngram_size=3,
        engram_n_embed_per_ngram=8,
        engram_n_head_per_ngram=2,
        engram_pad_id=0,
        engram_seed=0,
        engram_vocab_map_path=vocab_map_name,
        engram_layer_head_vocab_sizes={"2": [[11, 13], [17, 19]]},
        engram_fp8_embedding=fp8,
        engram_zero_and_freeze_per_hc_scales=True,
    )


def _oracle_hashes(
    sequences: list[list[int]],
    config: SimpleNamespace,
    lookup: np.ndarray,
) -> torch.Tensor:
    compressed_vocab = int(lookup.max()) + 1
    bound = max(1, (np.iinfo(np.int64).max // compressed_vocab) // 2)
    rng = np.random.default_rng(10007 * 2)
    multipliers = (
        rng.integers(0, bound, size=(config.engram_max_ngram_size,), dtype=np.int64) * 2
        + 1
    )
    pad = int(lookup[config.engram_pad_id])
    head_sizes = config.engram_layer_head_vocab_sizes["2"]
    rows = []
    for sequence in sequences:
        compressed = lookup[np.asarray(sequence, dtype=np.int64)]
        token_hashes = []
        for position in range(len(sequence)):
            columns = []
            rolling = np.int64(0)
            for shift in range(config.engram_max_ngram_size):
                source = compressed[position - shift] if position >= shift else pad
                rolling = np.bitwise_xor(rolling, source * multipliers[shift])
                if shift > 0:
                    columns.extend(
                        int(rolling % prime) for prime in head_sizes[shift - 1]
                    )
            token_hashes.append(columns)
        rows.append(torch.tensor(token_hashes, dtype=torch.int64))
    return torch.cat(rows, dim=0)


def test_hyv3_engram_hash_uses_request_lookback(tmp_path):
    lookup = np.arange(32, dtype=np.int64)
    vocab_map = tmp_path / "vocab_map.npz"
    _write_vocab_map(vocab_map, lookup)
    config = _config(vocab_map.name)
    hasher = HYV3EngramHash(config, str(tmp_path))

    query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32)
    input_ids = torch.tensor([4, 5, 8, 0], dtype=torch.int64)
    positions = torch.tensor([2, 3, 2, 0], dtype=torch.int64)
    lookback = torch.tensor([[3, 2], [7, 6]], dtype=torch.int32)

    actual = hasher(input_ids, positions, lookback, query_start_loc)[2]
    expected = _oracle_hashes([[2, 3, 4, 5], [6, 7, 8]], config, lookup)
    expected = torch.cat(
        [expected[2:4], expected[-1:], expected.new_full((1, 4), ENGRAM_NO_HASH_ID)]
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_hyv3_engram_no_hash_produces_zero_residual():
    config = _config("unused.npz")
    layer = HYV3EngramLayer(config, layer_id=2)
    hidden = torch.randn(3, config.hidden_size)
    hashes = torch.full((3, 4), ENGRAM_NO_HASH_ID, dtype=torch.int64)

    residual = layer(hidden, hashes)
    assert torch.count_nonzero(residual) == 0


def test_hyv3_engram_residual_matches_reference():
    config = _config("unused.npz")
    torch.manual_seed(123)
    layer = HYV3EngramLayer(config, layer_id=2).eval()
    hidden = torch.randn(5, config.hidden_size)
    hashes = torch.stack(
        [
            torch.randint(0, 11, (5,)),
            torch.randint(0, 13, (5,)),
            torch.randint(0, 17, (5,)),
            torch.randint(0, 19, (5,)),
        ],
        dim=-1,
    )

    actual = layer(hidden, hashes)
    embedding = layer.multi_head_embedding
    values = embedding.embedding(hashes + embedding.offsets)
    values = torch.stack(
        [
            F.rms_norm(
                values[:, head],
                (4,),
                layer.lookup_per_head_norms[head].weight,
                layer.lookup_per_head_norms[head].eps,
            )
            for head in range(4)
        ],
        dim=1,
    ).flatten(-2)
    key = F.linear(values, layer.key_projs[0].weight, layer.key_projs[0].bias)
    key = F.rms_norm(key, (8,), layer.norm_key[0].weight, layer.norm_key[0].eps)
    query = F.rms_norm(
        hidden, (8,), layer.norm_query[0].weight, layer.norm_query[0].eps
    )
    gate = (key * query).sum(-1) / math.sqrt(8)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gate = gate.sigmoid().unsqueeze(-1)
    value = F.linear(values, layer.value_proj.weight, layer.value_proj.bias)
    value = F.rms_norm(value, (8,), None, layer.unified_rmsnorm.eps)
    signal = layer.unified_pre_gamma * value
    expected = gate * signal + layer.unified_per_hc_scales[0] * signal
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
