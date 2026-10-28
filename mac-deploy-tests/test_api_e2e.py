"""End-to-end API tests for the local DeepSeek-V4.1-Flash deployment served by DeepInfer-Mac.

The server speaks the DeepSeek API contract via deepseek-recipe (OpenAI-, Responses- and
Anthropic-compatible surfaces). Contract facts this harness relies on (verified against
deepinfer_mac/server.py and deepseek_recipe 0.2.0):
  * model name: "deepseek-flash" (also alias "deepseek-v4-flash"); /v1/models lists it.
  * thinking (reasoning) mode is ON by default and FORCES temperature 1.0; a request without
    max_tokens defaults to 65536 tokens. Every deterministic test therefore sends
    {"thinking": {"type": "disabled"}, "max_tokens": N, "temperature": 0}.
  * reasoning_effort must be a string ("low" | "high" | "max"); never an int.
  * structured output: response_format {type: json_object} / {type: json_schema, json_schema: {...}}
    are mapped to xgrammar; with thinking on, the grammar starts after </think>.
  * usage carries prompt_cache_hit_tokens (DeepSeek API style) for prefix reuse.
  * one sequence at a time behind a global lock; SSD-streamed decode is slow (~1-4 tok/s),
    so per-request timeouts are generous and max_tokens is kept small.

Run:  DSV41_BASE_URL=http://127.0.0.1:8000 .venv/bin/python -m pytest -q -s test_api_e2e.py
Env:  DSV41_BASE_URL (default http://127.0.0.1:8000), DSV41_MODEL (default from /v1/models),
      DSV41_TIMEOUT seconds (default 3600), DSV41_MAX_TOKENS (default 48), DSV41_SKIP_SLOW=1 to skip
      the long-context tests.
Each request's latency / usage / text head is appended to results.jsonl for the final report.
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path

import httpx
import pytest

BASE = os.environ.get("DSV41_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = float(os.environ.get("DSV41_TIMEOUT", "3600"))
MAXTOK = int(os.environ.get("DSV41_MAX_TOKENS", "48"))
SKIP_SLOW = os.environ.get("DSV41_SKIP_SLOW") == "1"
RESULTS = Path(__file__).with_name("results.jsonl")
REF = Path("/Users/loncheng/Models/DeepSeek-V4.1-Flash-1017-ref")
NO_THINK = {"thinking": {"type": "disabled"}}


def _record(name: str, **kw):
    kw.update(test=name, ts=time.time())
    with RESULTS.open("a") as f:
        f.write(json.dumps(kw, ensure_ascii=False) + "\n")


@pytest.fixture(scope="session")
def client():
    with httpx.Client(base_url=BASE, timeout=httpx.Timeout(TIMEOUT, connect=30), trust_env=False) as c:
        yield c


@pytest.fixture(scope="session")
def model(client) -> str:
    if os.environ.get("DSV41_MODEL"):
        return os.environ["DSV41_MODEL"]
    r = client.get("/v1/models")
    r.raise_for_status()
    return r.json()["data"][0]["id"]


def chat(client, model, messages, *, max_tokens=MAXTOK, temperature=0.0, stream=False, thinking=False,
         extra=None, name="chat"):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if not thinking:
        body.update(NO_THINK)
    if extra:
        body.update(extra)
    t0 = time.perf_counter()
    if not stream:
        r = client.post("/v1/chat/completions", json=body)
        dt = time.perf_counter() - t0
        assert r.status_code == 200, f"{r.status_code}: {r.text[:800]}"
        js = r.json()
        msg = js["choices"][0]["message"]
        text = msg.get("content") or ""
        usage = js.get("usage") or {}
        _record(name, latency_s=round(dt, 2), completion_tokens=usage.get("completion_tokens"),
                prompt_tokens=usage.get("prompt_tokens"), cache_hit=usage.get("prompt_cache_hit_tokens"),
                finish=js["choices"][0].get("finish_reason"), text=text[:300],
                reasoning_head=(msg.get("reasoning_content") or "")[:120])
        return text, js
    body["stream"] = True
    chunks, first, events = [], None, []
    with client.stream("POST", "/v1/chat/completions", json=body) as r:
        assert r.status_code == 200, f"{r.status_code}: {r.read()[:800]!r}"
        for line in r.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                events.append("DONE")
                break
            if first is None:
                first = time.perf_counter() - t0
            js = json.loads(payload)
            events.append(js)
            for ch in js.get("choices") or []:
                delta = ch.get("delta") or {}
                if delta.get("content"):
                    chunks.append(delta["content"])
    dt = time.perf_counter() - t0
    text = "".join(chunks)
    _record(name, latency_s=round(dt, 2), ttft_s=round(first or 0, 2), n_chunks=len(chunks), text=text[:300],
            done=("DONE" in events))
    return text, {"chunks": chunks, "events": events}


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s).strip(".。,").lower()


# ---------------------------------------------------------------- contract tests

def test_models_endpoint(client):
    r = client.get("/v1/models")
    assert r.status_code == 200, r.text[:300]
    js = r.json()
    assert js.get("object") == "list" and js.get("data"), js
    assert "deepseek-flash" in [m["id"] for m in js["data"]]
    _record("models", ids=[m["id"] for m in js["data"]])


def test_stats_endpoint(client):
    r = client.get("/stats")
    assert r.status_code == 200, r.text[:300]
    js = r.json()
    assert "expert_cache_GB" in js, js
    _record("stats", **{k: v for k, v in js.items() if isinstance(v, (int, float, str))})


def test_basic_completion_shape(client, model):
    text, js = chat(client, model, [{"role": "user", "content": "Reply with exactly the word: pong"}], max_tokens=8, name="shape")
    assert js["object"] == "chat.completion"
    assert js["model"], js
    ch = js["choices"][0]
    assert ch["message"]["role"] == "assistant"
    assert ch["finish_reason"] in ("stop", "length")
    assert js["usage"]["completion_tokens"] > 0 and js["usage"]["prompt_tokens"] > 0
    assert "pong" in text.lower(), text


def test_prompt_tokens_match_reference_encoder(client, model):
    """usage.prompt_tokens must equal the token count of the reference-rendered prompt (+ generation header)."""
    import sys
    sys.path.insert(0, str(REF / "encoding"))
    from encoding import encode_messages  # type: ignore  # noqa: E402  (official reference encoder)
    from tokenizers import Tokenizer  # noqa: E402
    tok = Tokenizer.from_file(str(REF / "tokenizer.json"))
    messages = [{"role": "user", "content": "Reply with exactly the word: pong"}]
    rendered = encode_messages(messages, thinking_mode="chat")
    ref_ids = tok.encode(rendered, add_special_tokens=False).ids
    _, js = chat(client, model, messages, max_tokens=4, name="prompt_tokens")
    got = js["usage"]["prompt_tokens"]
    # engine prepends BOS (id 0) explicitly; allow +/-2 for header/BOS accounting differences
    assert abs(got - len(ref_ids)) <= 2, (got, len(ref_ids), rendered[-120:])
    _record("prompt_tokens_detail", server=got, reference=len(ref_ids))


def test_streaming_matches_nonstream(client, model):
    msgs = [{"role": "user", "content": "Count from 1 to 5 separated by commas, nothing else."}]
    t_ns, _ = chat(client, model, msgs, max_tokens=24, name="stream_ref")
    t_s, meta = chat(client, model, msgs, max_tokens=24, stream=True, name="stream")
    assert meta["events"] and meta["events"][-1] == "DONE", "stream did not end with data: [DONE]"
    assert len(meta["chunks"]) >= 2, "streaming returned a single content chunk"
    assert norm(t_s) == norm(t_ns), (t_s, t_ns)
    assert "1,2,3,4,5" in norm(t_s)


def test_greedy_determinism_with_logprobs(client, model):
    msgs = [{"role": "user", "content": "Write one sentence about the ocean."}]
    extra = {"logprobs": True, "top_logprobs": 2}
    a, ja = chat(client, model, msgs, max_tokens=24, extra=extra, name="determinism_a")
    b, jb = chat(client, model, msgs, max_tokens=24, extra=extra, name="determinism_b")
    assert a == b, f"greedy decode not deterministic:\n{a!r}\n{b!r}"
    la = ja["choices"][0].get("logprobs"); lb = jb["choices"][0].get("logprobs")
    assert la and lb and la.get("content"), "logprobs missing from response"
    toks_a = [(c["token"], round(c["logprob"], 4)) for c in la["content"]]
    toks_b = [(c["token"], round(c["logprob"], 4)) for c in lb["content"]]
    assert toks_a == toks_b, "token logprobs differ between identical greedy runs"
    assert all(c["logprob"] <= 0 for c in la["content"])


def test_seeded_sampling_reproducible(client, model):
    msgs = [{"role": "user", "content": "Name a colour."}]
    extra = {"seed": 7}
    a, _ = chat(client, model, msgs, max_tokens=12, temperature=0.7, extra=extra, name="seed_a")
    b, _ = chat(client, model, msgs, max_tokens=12, temperature=0.7, extra=extra, name="seed_b")
    assert a == b, f"seeded sampling not reproducible: {a!r} vs {b!r}"


def test_max_tokens_respected(client, model):
    _, js = chat(client, model, [{"role": "user", "content": "Tell me a long story about a dragon."}], max_tokens=8, name="max_tokens")
    assert js["usage"]["completion_tokens"] <= 8
    assert js["choices"][0]["finish_reason"] == "length"


def test_system_prompt_and_multiturn(client, model):
    msgs = [
        {"role": "system", "content": "You are a terse assistant. Always answer in UPPERCASE."},
        {"role": "user", "content": "What is the capital of France? One word."},
    ]
    t1, _ = chat(client, model, msgs, max_tokens=8, name="multiturn_1")
    assert "PARIS" in t1.upper(), t1
    msgs += [{"role": "assistant", "content": t1}, {"role": "user", "content": "And of Japan? One word."}]
    t2, js2 = chat(client, model, msgs, max_tokens=8, name="multiturn_2")
    assert "TOKYO" in t2.upper(), t2
    letters = re.sub(r"[^A-Za-z]", "", t2)
    assert letters == letters.upper(), f"system instruction (uppercase) not honoured: {t2!r}"
    # the second turn extends the first: the engine should report a prefix-cache hit
    hit = (js2.get("usage") or {}).get("prompt_cache_hit_tokens")
    _record("prefix_reuse_multiturn", cache_hit=hit, prompt=js2["usage"]["prompt_tokens"])
    assert hit is None or hit > 0


def test_unknown_model_rejected(client):
    r = client.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4, **NO_THINK})
    assert r.status_code == 400, (r.status_code, r.text[:300])


def test_malformed_body_rejected(client, model):
    r = client.post("/v1/chat/completions", json={"model": model, "messages": "not-a-list", "max_tokens": 4})
    assert r.status_code in (400, 422), (r.status_code, r.text[:300])


# ---------------------------------------------------------------- functional correctness oracles

@pytest.mark.parametrize("q,expect", [
    ("What is 17 * 23? Answer with the number only.", "391"),           # README transcript answer
    ("中国的首都是哪里？只回答城市名。", "北京"),                              # official inference/examples/example.txt
    ("Reverse the string 'deepseek'. Answer with just the reversed string.", "keespeed"),
    ("Which is larger, 9.11 or 9.9? Answer with just the number.", "9.9"),
])
def test_exact_answers(client, model, q, expect):
    text, _ = chat(client, model, [{"role": "user", "content": q}], max_tokens=12, name=f"oracle:{expect}")
    assert norm(expect) in norm(text), f"expected {expect!r} in {text!r}"


def test_primes_under_100(client, model):
    """Official example prompt: 列出100以内的所有素数 -> exactly the 25 primes."""
    text, _ = chat(client, model, [{"role": "user", "content": "列出100以内的所有素数，只输出用逗号分隔的数字。"}], max_tokens=96, name="primes")
    nums = sorted({int(x) for x in re.findall(r"\d+", text)})
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    assert nums == primes, (nums, text)


def test_code_generation_executes(client, model):
    prompt = ("Write a Python function `fib(n)` returning the n-th Fibonacci number with fib(0)=0, fib(1)=1. "
              "Output only the code in a single ```python block, no explanation.")
    text, _ = chat(client, model, [{"role": "user", "content": prompt}], max_tokens=120, name="codegen")
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    code = m.group(1) if m else text
    ns: dict = {}
    exec(compile(code, "<model>", "exec"), ns)  # noqa: S102 - deliberately executing model code in-process for the oracle
    assert [ns["fib"](i) for i in range(10)] == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]


# ---------------------------------------------------------------- structured output (xgrammar)

CITY_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "population_millions": {"type": "number"}},
    "required": ["city", "population_millions"],
    "additionalProperties": False,
}


def test_json_schema_no_thinking(client, model):
    extra = {"response_format": {"type": "json_schema", "json_schema": {"name": "city", "schema": CITY_SCHEMA, "strict": True}}}
    text, js = chat(client, model, [{"role": "user", "content": "Give the city Tokyo and its population in millions as JSON."}],
                    max_tokens=48, extra=extra, name="json_schema")
    obj = json.loads(text)
    assert set(obj) == {"city", "population_millions"}, obj
    assert obj["city"].lower().startswith("tokyo") and 5 < float(obj["population_millions"]) < 45
    assert js["choices"][0]["finish_reason"] == "stop"


def test_json_object_mode(client, model):
    extra = {"response_format": {"type": "json_object"}}
    text, _ = chat(client, model, [{"role": "user", "content": "Return a JSON object with keys a=1 and b=2. Output JSON only."}],
                   max_tokens=32, extra=extra, name="json_object")
    obj = json.loads(text)
    assert obj.get("a") == 1 and obj.get("b") == 2, obj


def test_json_schema_with_thinking_trigger(client, model):
    """Thinking on: the grammar must only start after </think> (trigger_token_ids path)."""
    extra = {"response_format": {"type": "json_schema", "json_schema": {"name": "city", "schema": CITY_SCHEMA, "strict": True}},
             "reasoning_effort": "low"}
    text, js = chat(client, model, [{"role": "user", "content": "Give the city Tokyo and its population in millions as JSON."}],
                    max_tokens=400, thinking=True, extra=extra, name="json_schema_thinking")
    obj = json.loads(text)
    assert obj["city"].lower().startswith("tokyo"), obj
    msg = js["choices"][0]["message"]
    _record("json_schema_thinking_detail", has_reasoning=bool(msg.get("reasoning_content")), finish=js["choices"][0]["finish_reason"])


# ---------------------------------------------------------------- thinking mode

def test_thinking_mode_round_trip(client, model):
    extra = {"reasoning_effort": "low"}
    text, js = chat(client, model, [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
                    max_tokens=300, thinking=True, extra=extra, name="thinking")
    msg = js["choices"][0]["message"]
    reasoning = msg.get("reasoning_content")
    assert reasoning, "thinking mode returned no reasoning_content"
    assert "<think>" not in (text or "") and "</think>" not in (text or ""), "think tags leaked into content"
    assert "4" in text, (text, reasoning[:200])


# ---------------------------------------------------------------- tool calling (DSML)

def test_tool_call_round_trip(client, model):
    tools = [{"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    }}]
    msgs = [{"role": "user", "content": "Use the get_weather tool to check the weather in Paris now. You must call the tool."}]
    text, js = chat(client, model, msgs, max_tokens=96, extra={"tools": tools}, name="tool_call")
    ch = js["choices"][0]
    calls = ch["message"].get("tool_calls") or []
    assert calls, f"model did not emit a tool call: {text!r}"
    assert ch["finish_reason"] == "tool_calls", ch["finish_reason"]
    fn = calls[0]["function"]
    assert fn["name"] == "get_weather"
    args = json.loads(fn["arguments"])
    assert "paris" in str(args.get("city", "")).lower(), args
    msgs += [ch["message"], {"role": "tool", "tool_call_id": calls[0]["id"], "content": json.dumps({"temp_c": 18, "sky": "cloudy"})}]
    text2, js2 = chat(client, model, msgs, max_tokens=48, extra={"tools": tools}, name="tool_result")
    assert "18" in text2 or "cloud" in text2.lower(), text2


# ---------------------------------------------------------------- long context / prefix reuse

@pytest.mark.skipif(SKIP_SLOW, reason="DSV41_SKIP_SLOW=1")
@pytest.mark.parametrize("n_lines", [700, 1400])   # ~6K and ~12K tokens: crosses the 4096 prefill chunk boundary once / twice
def test_long_prompt_needle_and_prefix_reuse(client, model, n_lines):
    filler = "\n".join(f"line {i}: the quick brown fox jumps over the lazy dog near the river bank." for i in range(n_lines))
    needle = "The secret code is 7731-ZETA."
    half = len(filler) // 2
    prompt = f"{filler[:half]}\n{needle}\n{filler[half:]}\n\nWhat is the secret code? Answer with just the code."
    msgs = [{"role": "user", "content": prompt}]
    t1, j1 = chat(client, model, msgs, max_tokens=12, name=f"needle_{n_lines}_a")
    assert j1["usage"]["prompt_tokens"] > 4096 * (1 if n_lines == 700 else 2), j1["usage"]
    assert "7731-zeta" in norm(t1), t1
    t2, j2 = chat(client, model, msgs, max_tokens=12, name=f"needle_{n_lines}_b")
    assert t2 == t1, "repeat of identical long prompt changed the answer (prefix reuse / Engram history bug?)"
    hit = (j2.get("usage") or {}).get("prompt_cache_hit_tokens")
    _record(f"needle_{n_lines}_cache", cache_hit=hit, prompt=j2["usage"]["prompt_tokens"])
    # after an unrelated prompt the answer must still be identical
    chat(client, model, [{"role": "user", "content": "Say hello."}], max_tokens=6, name="unrelated")
    t3, _ = chat(client, model, msgs, max_tokens=12, name=f"needle_{n_lines}_c")
    assert t3 == t1


# ---------------------------------------------------------------- vision

def test_vision_official_example(client, model):
    img = REF / "inference/examples/images/carrots.jpeg"
    if not img.exists():
        pytest.skip("official example image missing")
    b64 = base64.b64encode(img.read_bytes()).decode()
    msgs = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": "What vegetable is shown and what colour is it? Answer in a few words."},
    ]}]
    text, _ = chat(client, model, msgs, max_tokens=24, name="vision_carrots")
    assert "carrot" in text.lower() and "orange" in text.lower(), text


def test_vision_loopback_url_rejected(client, model):
    msgs = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "http://127.0.0.1:8000/v1/models"}},
        {"type": "text", "text": "Describe."},
    ]}]
    r = client.post("/v1/chat/completions", json={"model": model, "messages": msgs, "max_tokens": 8, **NO_THINK})
    assert r.status_code == 400, (r.status_code, r.text[:300])


# ---------------------------------------------------------------- other API surfaces

def test_anthropic_messages_endpoint(client, model):
    r = client.post("/v1/messages", json={
        "model": model, "max_tokens": 8, "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": "Reply with exactly the word: pong"}],
    }, headers={"anthropic-version": "2023-06-01"})
    assert r.status_code == 200, r.text[:400]
    js = r.json()
    assert js["type"] == "message" and js["role"] == "assistant"
    text = "".join(b.get("text", "") for b in js["content"] if b.get("type") == "text")
    assert "pong" in text.lower(), js
    _record("anthropic_messages", text=text[:100], usage=js.get("usage"))


def test_responses_endpoint(client, model):
    # Responses API runs in reasoning mode; give it room for the reasoning item plus the answer
    r = client.post("/v1/responses", json={
        "model": model, "max_output_tokens": 200, "input": "Reply with exactly the word: pong",
        "reasoning": {"effort": "low"},
    })
    assert r.status_code == 200, r.text[:400]
    js = r.json()
    assert js.get("object") == "response", js
    assert js.get("status") == "completed", js.get("status")
    assert any(item.get("type") == "reasoning" for item in js.get("output", [])), "no reasoning item in Responses output"
    texts = []
    for item in js.get("output", []):
        for c in item.get("content", []) or []:
            if c.get("type") == "output_text":
                texts.append(c.get("text", ""))
    assert "pong" in "".join(texts).lower(), js
    _record("responses", text="".join(texts)[:100])
