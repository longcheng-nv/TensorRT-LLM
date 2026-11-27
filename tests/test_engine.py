"""End-to-end engine on a tiny on-disk checkpoint: SSD expert streaming, engram table, prefix
reuse, chunked prefill, bounded replay, structured output."""

import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_parity as T  # noqa: E402  (skips the module when the reference is unavailable)
import tiny_ckpt  # noqa: E402

from deepinfer_mac.engine import LLM, SamplingParams  # noqa: E402


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    rm, mm = T.build()
    # 8-bit experts: near-lossless, so disk vs memory checks the plumbing, not quantization noise
    root = tiny_ckpt.write(rm, rm_args(rm), tmp_path_factory.mktemp("tiny"), T.TOKENIZER, bits=(8, 8, 8))
    llm = LLM(root, expert_cache_gb=0.01, bounded_replay=False, prefill_chunk=16)
    return mm, llm


def rm_args(rm):
    return T.tiny_args()


def _cos(a, b):
    a, b = np.ravel(a).astype(np.float64), np.ravel(b).astype(np.float64)
    return a @ b / np.linalg.norm(a) / np.linalg.norm(b)


def test_disk_matches_memory(setup):
    mm, llm = setup
    toks = np.random.default_rng(3).integers(100, 120000, size=40).tolist()
    a = mm.forward(mm.new_state(), toks)
    st, b, _, _ = llm._prefill(toks, want_hidden=False)
    c = _cos(np.array(a), np.array(b))
    print("disk vs memory (8-bit experts, 4-bit engram, chunked):", c, llm.weights.experts.stats)
    # catches wiring errors (wrong experts/rows give ~0.3); finer differences here are fp8 activation
    # rounding flips amplified by the tiny random model -- strict parity lives in test_parity and
    # scripts/real_check.py (real weights, >= 0.99997)
    assert c > 0.95


def test_prefix_reuse_and_generate(setup):
    """Reuse must equal building the same state by hand. (It need not equal a one-shot prefill:
    decode and prefill differ in the last ulps, and this tiny random router -- top-2 of 8 --
    has near-ties that such ulps flip.)"""
    _, llm = setup
    llm.reset()
    p = np.random.default_rng(4).integers(100, 120000, size=30).tolist()
    sp = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
    out1 = [o.token_id for o in llm.stream(p, sp)]
    assert llm.last_stats["reused_tokens"] == 0
    p2 = p + out1 + [500, 600]
    out2 = [o.token_id for o in llm.stream(p2, sp)]
    assert llm.last_stats["reused_tokens"] == len(p) + len(out1) - 1  # the final token is never fed
    m = llm.model
    st = m.new_state()
    m.forward(st, p[:-1])  # prefill = prompt body, then its last token (the snapshot point)
    m.forward(st, p[-1:])
    for t in out1[:-1]:
        m.forward(st, [t])
    rest = p2[len(st.tokens):]
    m.forward(st, rest[:-1])
    logits = m.forward(st, rest[-1:])
    manual = [int(np.argmax(np.array(logits[-1])))]
    print("reuse", out2[:1], "manual", manual, llm.last_stats)
    assert out2[:1] == manual


def test_bounded_replay_runs(setup):
    _, llm = setup
    llm.reset()
    llm.bounded_replay, llm.keep_tokens = True, 12
    p = np.random.default_rng(5).integers(100, 120000, size=60).tolist()
    out = [o.token_id for o in llm.stream(p, SamplingParams(temperature=0, max_tokens=4, ignore_eos=True))]
    llm.bounded_replay = False
    assert len(out) == 4


def test_structured_regex(setup):
    _, llm = setup
    llm.reset()
    sp = SamplingParams(temperature=1.0, max_tokens=12, seed=0, so_args={"schema_type": "regex", "schema": r"[0-9]{3}-[a-z]{2}"})
    text = "".join(o.text for o in llm.stream([0, 1000, 2000], sp))
    print("constrained:", repr(text))
    assert re.fullmatch(r"[0-9]{3}-[a-z]{2}(<｜end▁of▁sentence｜>)?", text)


def test_byte_saver_knobs(setup):
    """PREFILL_TOPK / SKIP_MISS_BELOW / LOOKAHEAD_K run end to end and stay close to the exact
    model (tiny model: top-2 of 8, so effects are large; this guards plumbing, not quality)."""
    from deepinfer_mac.model.model import MoE

    _, llm = setup
    p = np.random.default_rng(11).integers(100, 120000, size=40).tolist()
    sp = SamplingParams(temperature=0, max_tokens=6, ignore_eos=True)
    llm.reset()
    base = [o.token_id for o in llm.stream(p, sp)]
    saved = (MoE.PREFILL_TOPK, MoE.PREFILL_KEEP, MoE.SKIP_MISS_BELOW, MoE.LOOKAHEAD, MoE.LOOKAHEAD_K)
    try:
        MoE.PREFILL_TOPK, MoE.PREFILL_KEEP = 1, 8
        MoE.SKIP_MISS_BELOW = 0.45
        MoE.LOOKAHEAD, MoE.LOOKAHEAD_K = True, 1
        llm.reset()
        out = [o.token_id for o in llm.stream(p, sp)]
    finally:
        MoE.PREFILL_TOPK, MoE.PREFILL_KEEP, MoE.SKIP_MISS_BELOW, MoE.LOOKAHEAD, MoE.LOOKAHEAD_K = saved
    print("exact", base, "with byte savers", out)
    assert len(out) == 6


def test_snapshot_reuse_when_history_diverges(setup):
    """Thinking-mode follow-ups drop the previous reasoning from the prompt, so the new prompt
    diverges right after the old prompt's last token. The snapshot (one token before the old
    prompt's end) must be reused, and restoring it -- after generation moved the state on -- must
    reproduce exactly the state built by hand with the same chunking (rings, compressor tails,
    engram history all rolled back)."""
    _, llm = setup
    llm.reset()
    p = np.random.default_rng(12).integers(100, 120000, size=30).tolist()
    sp = SamplingParams(temperature=0, max_tokens=5, ignore_eos=True)
    list(llm.stream(p, sp))
    p2 = p[:-1] + [777, 888, 999]  # shares only p[:-1] with the previous prompt
    st, logits, _, reused = llm._prefill(p2, want_hidden=False)
    assert reused == len(p) - 1
    m = llm.model
    ref = m.new_state()
    m.forward(ref, p[:-1])  # same chunking: old body, new body, new last token
    m.forward(ref, [777, 888])
    ref_logits = m.forward(ref, [999])
    a, b = np.array(logits[-1]), np.array(ref_logits[-1])
    print("snapshot restore vs hand-built: max abs diff", float(np.abs(a - b).max()))
    assert np.allclose(a, b, atol=1e-4) and int(a.argmax()) == int(b.argmax())


def test_shared_prefix_anchor(setup):
    """A new conversation that starts with an earlier one's system prompt (everything up to the first
    <｜User｜>) restores the anchor state kept for it and prefills only the rest. The anchor is made the second
    time the prefix starts a conversation; the call that makes it reports no reuse. Restoring must give exactly
    the state built by hand with the same chunking -- also the second time, after decoding wrote into the
    restored state (the anchor's rings are copied on restore)."""
    _, llm = setup
    llm.reset()
    llm.ANCHOR_MIN = 8
    try:
        rng = np.random.default_rng(21)
        user = llm._user_id
        assert user is not None
        prefix = rng.integers(100, 120000, size=20).tolist() + [user]
        sp = SamplingParams(temperature=0, max_tokens=6, ignore_eos=True)
        a = prefix + rng.integers(100, 120000, size=7).tolist()
        list(llm.stream(a, sp))
        assert llm.last_stats["reused_tokens"] == 0 and not llm._anchors  # seen once: no anchor yet
        a2 = prefix + rng.integers(100, 120000, size=5).tolist()
        list(llm.stream(a2, sp))
        assert llm.last_stats["reused_tokens"] == 0  # this call made the anchor
        assert len(llm._anchors) == 1 and llm._anchors[0][1] == prefix
        m = llm.model
        for seed in (22, 23):
            b = prefix + np.random.default_rng(seed).integers(100, 120000, size=6).tolist()
            assert llm.cached_prefix_len(b) == len(prefix)
            st, logits, _, reused = llm._prefill(b, want_hidden=False)
            assert reused == len(prefix)
            ref = m.new_state()
            m.forward(ref, prefix)  # same chunking: anchor, rest of the body, last token
            m.forward(ref, b[len(prefix):-1])
            ref_logits = m.forward(ref, b[-1:])
            x, y = np.array(logits[-1]), np.array(ref_logits[-1])
            print(f"anchor restore #{seed - 21} vs hand-built: max abs diff", float(np.abs(x - y).max()))
            assert np.allclose(x, y, atol=1e-4) and int(x.argmax()) == int(y.argmax())
            llm._state = st
            list(llm.stream(b + [5, 6], sp))  # decode moves the restored state on before the next restore
    finally:
        del llm.ANCHOR_MIN
        llm.reset()


def test_chunked_bounded_prefill_matches_hand_built(setup):
    """_feed's chunking with bounded replay: earlier chunks skip the decoder, the last one (at least keep_tokens)
    replays it over its tail -- the same as forwarding those chunks by hand."""
    _, llm = setup
    llm.reset()
    saved = llm.bounded_replay, llm.keep_tokens, llm.prefill_chunk
    llm.bounded_replay, llm.keep_tokens, llm.prefill_chunk = True, 8, 20
    try:
        p = np.random.default_rng(31).integers(100, 120000, size=61).tolist()
        st, logits, _, reused = llm._prefill(p, want_hidden=False)
        assert reused == 0
        m = llm.model
        ref = m.new_state()
        body = p[:-1]  # 60 tokens -> chunks of 20, 20, 20 (the last keeps >= 8 for the decoder)
        m.forward(ref, body[:20], replay_from=20)
        m.forward(ref, body[20:40], replay_from=20)
        m.forward(ref, body[40:], replay_from=12)
        ref_logits = m.forward(ref, p[-1:])
        x, y = np.array(logits[-1]), np.array(ref_logits[-1])
        print("chunked bounded prefill vs hand-built: max abs diff", float(np.abs(x - y).max()))
        assert np.allclose(x, y, atol=1e-4) and int(x.argmax()) == int(y.argmax())
    finally:
        llm.bounded_replay, llm.keep_tokens, llm.prefill_chunk = saved
        llm.reset()
