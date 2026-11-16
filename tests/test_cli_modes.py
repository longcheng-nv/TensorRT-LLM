"""CLI decode-mode defaults: chat / serve / bench run the fast (lossy) preset unless --lossless; an explicit
--sub-rank / --spec-k always wins; --fast and --lossless are mutually exclusive."""
import pytest

from deepinfer_mac.__main__ import build_parser, decode_mode
from deepinfer_mac.model.model import MoE

_KNOBS = ("SPEC_K", "SPEC_K2", "SUB_RANK", "SUB_GAP", "SKIP_MISS_BELOW", "PREFILL_TOPK", "LOOKAHEAD", "LOOKAHEAD_K")
_MISSING = object()


@pytest.fixture(autouse=True)
def _restore_moe_knobs():
    keep = {k: MoE.__dict__.get(k, _MISSING) for k in _KNOBS}
    yield
    for k, v in keep.items():
        if v is _MISSING:
            if k in MoE.__dict__:
                delattr(MoE, k)
        else:
            setattr(MoE, k, v)


def _mode(cmd="serve", *argv):
    desc = decode_mode(build_parser().parse_args([cmd, *argv]))
    return MoE.SUB_RANK, MoE.SPEC_K, desc


@pytest.mark.parametrize("cmd", ["serve", "chat", "bench"])
def test_default_is_the_fast_preset(cmd):
    sub, spec, desc = _mode(cmd)
    assert (sub, spec) == (2, 2)
    assert "LOSSY" in desc and "--lossless" in desc


def test_fast_flag_equals_default():
    assert _mode("serve", "--fast")[:2] == (2, 2)


def test_lossless_is_exact_routing():
    sub, spec, desc = _mode("serve", "--lossless")
    assert (sub, spec) == (None, 4)
    assert desc.startswith("decode mode: lossless")


def test_explicit_knobs_win():
    assert _mode("serve", "--sub-rank", "3")[:2] == (3, 2)
    assert _mode("serve", "--spec-k", "4")[:2] == (2, 4)
    assert _mode("serve", "--lossless", "--spec-k", "3")[:2] == (None, 3)
    sub, _, desc = _mode("serve", "--lossless", "--sub-rank", "4")
    assert sub == 4 and "LOSSY" in desc


def test_fast_and_lossless_are_exclusive():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["serve", "--fast", "--lossless"])
