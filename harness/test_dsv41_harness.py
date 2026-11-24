#!/usr/bin/env python3
"""Acceptance test: DeepSeek Harness (official SDK, sdk-minimal + patches/sdk-minimal.local.yml)
driving the local DeepSeek-V4.1-Flash server, plus the `dsv41` terminal front end and `dsv41 web`.

Run through the launcher (it starts the model server first):  dsv41 test [--quick]
Results: ~/Models/logs/dsh/test-<time>.json; one PASS/FAIL line per check on stdout.
The default server mode (--fast) is lossy and not reproducible, so answers are checked by keywords.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

from deepseek_harness import DeepSeekHarness, DeepSeekHarnessConfig

HERE = Path(__file__).resolve().parent
MODELS = HERE.parent.parent
PATCH = HERE / "patches" / "sdk-minimal.local.yml"
IMAGES = HERE / "testdata"  # the model card's example images (MIT, see testdata/LICENSE-DeepSeek.txt)
BASE = os.environ.get("DEEPSEEK_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
# test sessions stay out of the homes used day to day (dsh/home, dsh/home-web)
HOME = os.environ.get("DSV41_TEST_HOME", str(MODELS / "dsh" / "home-test"))
WEB_HOME = os.environ.get("DSV41_TEST_WEB_HOME", str(MODELS / "dsh" / "home-test-web"))
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get(path: str, timeout: float = 3.0):
    return json.loads(OPENER.open(BASE + path, timeout=timeout).read())


class LiveLog:
    """Samples GET /live: per server request id -> prompt tokens, cache hit, images, ttft, decode tok/s."""

    def __init__(self):
        self.reqs: dict[str, dict] = {}
        self._before: set[str] = set()
        self._stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self._stop.wait(0.25):
            try:
                j = get("/live")
            except Exception:  # noqa: BLE001
                continue
            if j.get("id"):
                j.pop("text", None)
                self.reqs[j["id"]] = j

    def mark(self):
        self._before = set(self.reqs)

    def refresh(self):
        try:
            j = get("/live")
            if j.get("id"):
                j.pop("text", None)
                self.reqs[j["id"]] = j
        except Exception:  # noqa: BLE001
            pass

    def new_since_mark(self) -> list[dict]:
        self.refresh()
        return [r for k, r in self.reqs.items() if k not in self._before]

    def stop(self):
        self._stop.set()


class Runner:
    def __init__(self, workspace: Path, effort: str, live: LiveLog):
        self.ws, self.live = workspace, live
        self.cfg = DeepSeekHarnessConfig(
            dsh_home=HOME, cwd=str(workspace), profile="sdk-minimal", patches=(str(PATCH),),
            provider="deepseek-official", model="deepseek-flash", reasoning_effort=effort, max_tokens=4096,
            base_url=BASE, api_key="local", initialize_timeout_seconds=120, shutdown_timeout_seconds=5,
            env={"DSH_TELEMETRY_MODE": "DISABLED", "DSH_PERMISSION_MODE": "workspace-write"},
        )
        self.h = DeepSeekHarness(self.cfg)
        self.h.start()

    def session(self):
        return self.h.start_session()

    def turn(self, session, text: str, images: list[Path] = ()) -> dict:
        import base64
        import mimetypes

        blocks = [{"type": "text", "text": text}] + [
            {"type": "image", "data": base64.b64encode(p.read_bytes()).decode(), "mimeType": mimetypes.guess_type(p.name)[0] or "image/png"}
            for p in images]
        calls, results, usage = [], [], []

        def on(n):
            ev = n.payload.get("event") if isinstance(n.payload, dict) else None
            if not isinstance(ev, dict):
                return
            d = ev.get("data") or {}
            if ev.get("type") == "tool/call":
                calls.append({"name": d.get("name"), "arguments": d.get("arguments")})
            elif ev.get("type") == "tool/result":
                m = d.get("message") or {}
                results.append(" ".join(b.get("text", "") for b in m.get("content") or [] if b.get("type") == "text")[:400])
            elif ev.get("type") == "assistant/message" and isinstance(d.get("usage"), dict):
                usage.append(d["usage"])

        self.live.mark()
        t0 = time.time()
        r = session.run(blocks, on_notification=on)
        dt = time.time() - t0
        reqs = self.live.new_since_mark()
        return {"prompt": text, "final": r.final_response, "finish": r.finish_reason, "seconds": round(dt, 1),
                "tool_calls": calls, "tool_results": results, "usage": usage,
                "requests": [{k: q.get(k) for k in ("prompt_tokens", "cached", "images", "generated", "ttft_s", "decode_tps", "elapsed_s")} for q in reqs]}

    def close(self):
        self.h.close()


def lines(p: Path) -> list[str]:
    return [x.strip() for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="chat + file + image checks only")
    ap.add_argument("--only", choices=["agent", "term", "web"], help="one group: agent (SDK checks 1-7), term (dsv41 -p), web (UI boot)")
    ap.add_argument("--effort", default="low")
    a = ap.parse_args()
    want = {g: a.only in (None, g) for g in ("agent", "term", "web")}
    want["web"] = want["web"] and not a.quick

    results: list[dict] = []
    log_dir = MODELS / "logs" / "dsh"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_file = log_dir / f"test-{time.strftime('%Y%m%d-%H%M%S')}.json"
    ws = Path(tempfile.mkdtemp(prefix="dsv41-test-"))
    shutil.copy(IMAGES / "corn.jpeg", ws / "corn.jpeg")
    outside = MODELS / "dsh" / "run" / f"sandbox-probe-{int(time.time())}.txt"
    live = LiveLog()
    print(f"workspace {ws}\nserver {BASE}  mode={os.environ.get('DSV41_SERVER_MODE', '?')}  effort={a.effort}", flush=True)

    def check(name: str, ok: bool, detail: dict, note: str = ""):
        results.append({"check": name, "pass": bool(ok), "note": note, **detail})
        reqs = detail.get("requests") or []
        req_s = "; ".join(f"{q.get('prompt_tokens')}tok/cached {q.get('cached')}/img {q.get('images')}/ttft {q.get('ttft_s')}s" for q in reqs)
        print(f"{'PASS' if ok else 'FAIL'}  {name}  [{detail.get('seconds', '-')}s]  {note}", flush=True)
        if detail.get("final") is not None:
            print(f"      answer: {str(detail['final'])[:160]!r}", flush=True)
        if req_s:
            print(f"      server: {req_s}", flush=True)

    if want["agent"]:
        r = Runner(ws, a.effort, live)
        try:
            # 1. plain chat
            s = r.session()
            d = r.turn(s, "17 乘以 23 等于多少？只回答数字。")
            check("chat: 17*23", "391" in d["final"], d)

            # 2-3. file operations through the persistent bash, in one session (prefix cache must hit)
            s = r.session()
            d = r.turn(s, "在当前目录创建文件 notes.txt，内容是三行：alpha、beta、gamma（每行一个词，不要别的内容）。完成后用 cat 显示它。")
            check("file: create notes.txt", lines(ws / "notes.txt") == ["alpha", "beta", "gamma"], d, f"file={lines(ws / 'notes.txt')}")
            d = r.turn(s, "把 notes.txt 里的 beta 改成 BETA-2，再在末尾追加一行 delta，然后 cat 显示结果。")
            got = lines(ws / "notes.txt")
            check("file: edit + append", got == ["alpha", "BETA-2", "gamma", "delta"], d, f"file={got}")
            hits = [q for q in d["requests"] if q.get("prompt_tokens")]
            warm = all((q.get("cached") or 0) >= 0.8 * q["prompt_tokens"] for q in hits) if hits else False
            check("prefix cache: follow-up requests reuse >= 80% of the prompt", warm, {"requests": hits})

            # 4. image attached to the prompt (what dragging a file into the terminal does)
            s = r.session()
            d = r.turn(s, "图里是什么蔬菜？只回答蔬菜的中文名称。", images=[IMAGES / "carrots.jpeg"])
            check("image: attached carrots.jpeg", any(k in d["final"] for k in ("胡萝卜", "红萝卜", "carrot", "Carrot")), d)

            if not a.quick:
                # 5. the model opens an image by path itself (read_image tool)
                d = r.turn(s, "用 read_image 工具查看当前目录下的 corn.jpeg，图里是什么？只回答中文名称。")
                used = any(c["name"] == "read_image" for c in d["tool_calls"])
                check("image: read_image corn.jpeg", used and any(k in d["final"] for k in ("玉米", "corn", "Corn")), d,
                      f"tools={[c['name'] for c in d['tool_calls']]}")

                # 6. sandbox: writes outside the workspace are refused by macOS Seatbelt
                s = r.session()
                d = r.turn(s, f"用 bash 原样执行这一条命令，然后告诉我输出：echo probe > {outside}; echo rc=$?")
                denied = any("not permitted" in x or "rc=1" in x for x in d["tool_results"])
                check("sandbox: write outside the workspace is denied", not outside.exists() and denied, d,
                      f"file_created={outside.exists()} results={d['tool_results'][:1]}")
                if outside.exists():
                    outside.unlink()

                # 7. multi-turn memory
                s = r.session()
                r.turn(s, "记住这个暗号：蓝鲸-4721。只回答“好”。")
                d = r.turn(s, "刚才的暗号是什么？只回答暗号。")
                check("memory: recall across turns", "4721" in d["final"], d)
        finally:
            r.close()

    # 8. the terminal front end, as a user would call it (non-interactive one-shot)
    if want["term"]:
        t0 = time.time()
        p = subprocess.run([str(HERE / "dsv41"), "-p", "2 加 3 等于几？只回答数字。", "--effort", "off"], cwd=ws,
                           capture_output=True, text=True, timeout=900, env={**os.environ, "NO_COLOR": "1", "DSH_HOME": HOME})
        check("dsv41 -p (terminal front end)", p.returncode == 0 and "5" in p.stdout,
              {"seconds": round(time.time() - t0, 1), "final": p.stdout.strip()[-200:]}, f"rc={p.returncode} {p.stderr.strip()[-200:]}")

    if want["web"]:
        # 9. the browser UI boots on the local profile (its own home, as `dsv41 web` sets it), every
        #    plugin activates, and the tokenized URL it prints serves the page
        import signal
        import socket

        with socket.socket() as sock:  # a free port
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = {**os.environ, "DSH_TELEMETRY_MODE": "DISABLED", "DSH_HOME": WEB_HOME}
        log = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False)
        # own process group: `dsh` is a Python wrapper whose runtime binary is a child process
        web = subprocess.Popen([str(MODELS / "dsh" / "venv" / "bin" / "dsh"), "web", "--patch", str(HERE / "patches" / "web.local.yml"),
                                "--no-open", "--port", str(port)], cwd=ws, env=env, stdout=log, stderr=subprocess.STDOUT, text=True,
                               start_new_session=True)
        ok, url, t0 = False, None, time.time()
        browser = urllib.request.build_opener(urllib.request.ProxyHandler({}), urllib.request.HTTPCookieProcessor())
        while time.time() - t0 < 120 and web.poll() is None and not ok:
            time.sleep(1)
            text = Path(log.name).read_text()
            url = next((w for w in text.split() if w.startswith(f"http://127.0.0.1:{port}/?token=")), None)
            if url:
                try:  # the token URL sets a cookie and redirects to the app page
                    ok = browser.open(url, timeout=3).status == 200
                except Exception:  # noqa: BLE001
                    pass
        time.sleep(3)  # let late activation warnings reach the log
        text = Path(log.name).read_text()
        try:
            os.killpg(web.pid, signal.SIGTERM)
            web.wait(timeout=20)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(web.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        inactive = "did not activate" in text
        check("dsv41 web: UI serves on the local profile", ok and not inactive,
              {"seconds": round(time.time() - t0, 1)}, f"url={'yes' if url else 'no'} inactive_plugins={inactive} {text.strip()[-300:] if inactive or not ok else ''}")

    live.stop()
    out_file.write_text(json.dumps({"base": BASE, "mode": os.environ.get("DSV41_SERVER_MODE"), "effort": a.effort,
                                    "workspace": str(ws), "results": results}, ensure_ascii=False, indent=1))
    n_ok = sum(x["pass"] for x in results)
    print(f"\n{n_ok}/{len(results)} passed — details: {out_file}")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
