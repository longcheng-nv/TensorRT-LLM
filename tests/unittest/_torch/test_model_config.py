import json
import struct
import types

import pytest
import torch

from tensorrt_llm._torch.model_config import _DEEPSEEK_V4_ROUTED_EXPERT_WEIGHT, ModelConfig
from tensorrt_llm._torch.pyexecutor.model_loader import validate_and_set_kv_cache_quant
from tensorrt_llm.llmapi.llm_args import (
    DeepSeekSparseAttentionConfig,
    DeepSeekV4SparseAttentionConfig,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


def make_pretrained_config(
    *,
    num_attention_heads: int = 16,
    num_key_value_heads=8,
    head_dim: int | None = None,
    num_hidden_layers: int = 1,
    vocab_size: int = 3000,
):
    # A minimal config object that provides the attributes used by
    # ModelConfig.get_bindings_model_config().
    hidden_size = head_dim * num_attention_heads
    intermediate_size = hidden_size * 4

    return types.SimpleNamespace(
        architectures=["DummyArchitecture"],
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        num_hidden_layers=num_hidden_layers,
        vocab_size=vocab_size,
        torch_dtype=torch.float16,
    )


@pytest.mark.parametrize(
    "num_key_value_heads",
    [
        pytest.param(8, id="kv_heads_scalar"),
        pytest.param([8, 20], id="kv_heads_per_layer_varied"),
    ],
)
@pytest.mark.parametrize("enable_attention_dp", [False, True])
@pytest.mark.parametrize(
    "mapping_kwargs",
    [
        # Same tp/cp sizes, but different ways of setting attention TP:
        # - No explicit attn_tp_size: Mapping infers it.
        # - Explicit attn_tp_size: Mapping uses the provided value.
        dict(world_size=8, tp_size=4, cp_size=2),
        dict(world_size=4, tp_size=2, cp_size=2, attn_tp_size=4),
    ],
)
def test_get_bindings_model_config_attention_dp_attn_tp_override(
    enable_attention_dp, mapping_kwargs, num_key_value_heads
):
    mapping = Mapping(enable_attention_dp=enable_attention_dp, **mapping_kwargs)
    cfg = make_pretrained_config(
        # Keep values consistent:
        # hidden_size = num_attention_heads * head_dim.
        num_attention_heads=16,
        head_dim=4,
        num_key_value_heads=num_key_value_heads,
        num_hidden_layers=2,
    )
    model_config = ModelConfig(pretrained_config=cfg, mapping=mapping)

    tokens_per_block = 32
    bindings_cfg = model_config.get_bindings_model_config(tokens_per_block=tokens_per_block)

    # bindings hidden_size is sharded by attn_tp_size and attn_cp_size.
    attn_tp_size = mapping.attn_tp_size if not mapping.enable_attention_dp else 1
    attn_cp_size = mapping.attn_cp_size

    def ceil_div(a, b):
        return (a + b - 1) // b

    assert bindings_cfg.num_heads == ceil_div(cfg.num_attention_heads, attn_tp_size * attn_cp_size)
    # bindings hidden_size is sharded by attn_tp_size.
    assert bindings_cfg.hidden_size == ceil_div(cfg.hidden_size, attn_tp_size)
    if isinstance(cfg.num_key_value_heads, (list, tuple)):
        expected_num_kv_heads_per_layer = [
            ceil_div(kv, attn_tp_size * attn_cp_size) for kv in cfg.num_key_value_heads
        ]
        assert list(bindings_cfg.num_kv_heads_per_layer) == expected_num_kv_heads_per_layer
        assert bindings_cfg.num_kv_heads(0) == expected_num_kv_heads_per_layer[0]
    else:
        assert bindings_cfg.num_kv_heads(0) == ceil_div(
            cfg.num_key_value_heads, attn_tp_size * attn_cp_size
        )

    # tp_size-dependent value (uses mapping.tp_size, not attn_tp_size).
    assert bindings_cfg.mlp_hidden_size == ceil_div(cfg.intermediate_size, mapping.tp_size)
    assert bindings_cfg.tokens_per_block == tokens_per_block


def _make_model_config_with_kv_quant(kv_cache_quant_algo):
    return ModelConfig(quant_config=QuantConfig(kv_cache_quant_algo=kv_cache_quant_algo))


def test_validate_and_set_kv_cache_quant_auto_uses_checkpoint():
    model_config = _make_model_config_with_kv_quant(QuantAlgo.FP8)
    validate_and_set_kv_cache_quant(model_config, "auto")
    assert model_config.quant_config.kv_cache_quant_algo == QuantAlgo.FP8


def test_validate_and_set_kv_cache_quant_explicit_dtype_overrides():
    model_config = _make_model_config_with_kv_quant(QuantAlgo.FP8)
    validate_and_set_kv_cache_quant(model_config, "nvfp4")
    assert model_config.quant_config.kv_cache_quant_algo == QuantAlgo.NVFP4


def test_validate_and_set_kv_cache_quant_rejects_invalid_dtype():
    model_config = _make_model_config_with_kv_quant(QuantAlgo.FP8)
    with pytest.raises(ValueError, match="Accepted types are"):
        validate_and_set_kv_cache_quant(model_config, "invalid_dtype")


def _write_safetensors_header(checkpoint_dir, tensor_dtype, tensor_shape):
    shard_name = "model-00001-of-00001.safetensors"
    header = {
        _DEEPSEEK_V4_ROUTED_EXPERT_WEIGHT: {
            "dtype": tensor_dtype,
            "shape": tensor_shape,
            "data_offsets": [0, 0],
        }
    }
    encoded_header = json.dumps(header).encode("utf-8")

    with open(checkpoint_dir / shard_name, "wb") as f:
        f.write(struct.pack("<Q", len(encoded_header)))
        f.write(encoded_header)

    with open(checkpoint_dir / "model.safetensors.index.json", "w") as f:
        json.dump({"weight_map": {_DEEPSEEK_V4_ROUTED_EXPERT_WEIGHT: shard_name}}, f)


@pytest.mark.parametrize(
    "tensor_dtype,tensor_shape,expected_layout,expected_is_base",
    [
        pytest.param("I8", [2048, 2048], "mxfp4", False, id="mxfp4"),
        pytest.param("U8", [2048, 2048], "nvfp4", False, id="nvfp4"),
        pytest.param("F8_E4M3", [2048, 4096], None, True, id="base-fp8"),
    ],
)
def test_deepseek_v4_base_checkpoint_detection(
    tmp_path, tensor_dtype, tensor_shape, expected_layout, expected_is_base
):
    _write_safetensors_header(tmp_path, tensor_dtype, tensor_shape)

    assert ModelConfig._detect_deepseek_v4_routed_moe_layout(str(tmp_path)) == expected_layout
    assert ModelConfig._is_deepseek_v4_base_checkpoint(str(tmp_path)) is expected_is_base


# ---------------------------------------------------------------------------
# index_topk merge between a user-supplied sparse_attention_config and the
# checkpoint's pretrained_config.
#
# The production merge lives in the nested
# ``update_sparse_attention_indexer_config`` helper inside
# ``ModelConfig.from_pretrained`` (tensorrt_llm/_torch/model_config.py): the
# DeepSeek-V4 path (~L697-700) and the DeepSeek-V3.2 / GLM path (~L769-772).
# That helper is not importable and exercising it end to end needs a real
# checkpoint + quant config, so we lock its *contract* here against the real
# config classes. ``_resolve_index_topk`` is a byte-for-byte mirror of the
# production expression; if the production merge changes, update this mirror.
# ---------------------------------------------------------------------------


def _resolve_index_topk(sparse_attention_config, pretrained_config):
    # Mirror of model_config.py: an explicitly-set index_topk wins, otherwise
    # the checkpoint value is used. A plain ``a or b`` is WRONG here because
    # the V4 subclass default (512) is truthy and would shadow the checkpoint.
    if "index_topk" in sparse_attention_config.model_fields_set:
        return sparse_attention_config.index_topk
    return pretrained_config.index_topk


def _pretrained_config_with_index_topk(index_topk):
    # The merge only reads .index_topk; index_n_heads / index_head_dim use a
    # separate `or` path and are irrelevant to this contract.
    return types.SimpleNamespace(index_topk=index_topk, index_n_heads=None, index_head_dim=None)


def test_v4_pro_index_topk_falls_back_to_checkpoint():
    """V4 Pro regression: a user who enables GVR without setting index_topk
    must inherit the checkpoint's 1024, not the truthy 512 subclass default."""
    # fp8 keeps construction GPU-SM-agnostic (V4 defaults to fp4 -> SM>=100).
    sac = DeepSeekV4SparseAttentionConfig(enable_heuristic_topk=True, indexer_k_dtype="fp8")
    pc = _pretrained_config_with_index_topk(1024)

    assert "index_topk" not in sac.model_fields_set
    assert _resolve_index_topk(sac, pc) == 1024
    # The original `or`-based merge produced the silent-halving bug:
    assert (sac.index_topk or pc.index_topk) == 512


def test_v4_explicit_index_topk_wins_over_checkpoint():
    """An explicit user value is honored even when it differs from the
    checkpoint."""
    sac = DeepSeekV4SparseAttentionConfig(index_topk=512, indexer_k_dtype="fp8")
    pc = _pretrained_config_with_index_topk(1024)

    assert "index_topk" in sac.model_fields_set
    assert _resolve_index_topk(sac, pc) == 512


def test_v4_flash_index_topk_unaffected():
    """V4 Flash checkpoints carry index_topk=512, which equals the subclass
    default, so the fix is a no-op for Flash."""
    sac = DeepSeekV4SparseAttentionConfig(enable_heuristic_topk=True, indexer_k_dtype="fp8")
    pc = _pretrained_config_with_index_topk(512)

    assert _resolve_index_topk(sac, pc) == 512


def test_v32_index_topk_falls_back_to_checkpoint():
    """V3.2 / GLM path: the base config default is None, so the checkpoint
    value flows through. (The old `or` already worked here precisely because
    None is falsy -- this documents that V3.2 is unaffected by the bug.)"""
    sac = DeepSeekSparseAttentionConfig(enable_heuristic_topk=True)
    pc = _pretrained_config_with_index_topk(1024)

    assert sac.index_topk is None
    assert "index_topk" not in sac.model_fields_set
    assert _resolve_index_topk(sac, pc) == 1024
    # Old `or` happened to give the same answer for V3.2 (None is falsy):
    assert (sac.index_topk or pc.index_topk) == 1024


def test_model_fields_set_root_cause():
    """Pin the exact Pydantic mechanism the fix relies on: the V4 subclass's
    truthy 512 default is NOT in model_fields_set unless the user sets it."""
    default_cfg = DeepSeekV4SparseAttentionConfig(indexer_k_dtype="fp8")
    assert default_cfg.index_topk == 512
    assert "index_topk" not in default_cfg.model_fields_set

    explicit_cfg = DeepSeekV4SparseAttentionConfig(index_topk=512, indexer_k_dtype="fp8")
    assert explicit_cfg.index_topk == 512
    assert "index_topk" in explicit_cfg.model_fields_set
