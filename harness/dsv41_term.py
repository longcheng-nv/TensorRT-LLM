#!/usr/bin/env python3
"""dsv41 terminal agent: DeepSeek Harness on the local DeepSeek-V4.1-Flash server.

The agent is DeepSeek's own Harness (official Python SDK, `sdk-minimal` profile = the "DSH Minimal"
scaffold of the model card: one-line persona, a persistent bash, read_image, working_directory),
overlaid by patches/sdk-minimal.local.yml for the local DeepInfer-Mac server. This file is only the
terminal front end: line input, dragged-in files, live output, approvals, sessions.

  * Type a request. Drag files from Finder into the Terminal window: image paths are attached as
    images (the model sees them), every path is also kept in the text so the agent can use it.
  * /paste attaches the image on the clipboard (e.g. a Ctrl+Shift+Cmd+4 screenshot).
  * The agent works in the directory dsv41 was started from: the macOS sandbox confines its writes
    to that directory and /tmp (reads are unrestricted). --full-access lifts the confinement.
  * Ctrl+C stops the current request (the harness restarts and the session is restored).

Run it through the `dsv41` launcher, which starts the model server first.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATCH = HERE / "patches" / "sdk-minimal.local.yml"
DEFAULT_HOME = HERE.parent.parent / "dsh" / "home"  # /Users/<you>/Models/dsh/home
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
EFFORTS = ("off", "low", "high", "max")

# ------------------------------------------------------------------------------ terminal output

TTY = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if TTY else s


DIM, GRAY, CYAN, YELLOW, RED, GREEN, BOLD = "2", "90", "36", "33", "31", "32", "1"
io_lock = threading.RLock()


def out(s: str = "", end: str = "\n") -> None:
    with io_lock:
        sys.stdout.write(s + end)
        sys.stdout.flush()


def clear_status() -> None:
    if TTY:
        with io_lock:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()


def width() -> int:
    return max(40, min(shutil.get_terminal_size((100, 20)).columns, 160))


# ------------------------------------------------------------------------------ server helpers


def http_json(url: str, timeout: float = 3.0):
    req = urllib.request.Request(url, headers={"accept": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # localhost: never via a proxy
    with opener.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


class Live:
    """Shows the model's output while a step runs, by polling the server's GET /live (the harness
    SDK only delivers whole model steps). Reasoning is gray, the answer plain, DSML tool-call markup
    hidden (the tool call is printed from the harness event instead). One server request = one
    model step; all terminal writes happen under io_lock."""

    MARKERS = ("</think>", "<｜DSML｜")

    def __init__(self, base: str, thinking: bool, enabled: bool = True):
        self.base, self.thinking, self.enabled = base.rstrip("/"), thinking, enabled
        self.done: set[str] = set()
        self.req: str | None = None
        self._reset(None)
        self.line_open = False  # a text line without its newline is on screen
        self.status_shown, self.last_status = False, ""  # the self-overwriting status line is on screen
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _reset(self, rid: str | None) -> None:
        self.req, self.offset, self.pending, self.shown, self.answer_started = rid, 0, "", False, False
        self.mode = "reason" if self.thinking else "answer"

    # -- polling thread
    def start(self) -> None:
        if not self.enabled:
            return
        try:  # whatever /live shows before our request is sent belongs to someone else (or the past)
            prev = http_json(f"{self.base}/live?offset=1000000000", timeout=2.0).get("id")
            if prev:
                self.done.add(prev)
        except (urllib.error.URLError, OSError, ValueError):
            pass
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        with io_lock:
            self._settle()

    def _loop(self) -> None:
        while not self._stop.wait(0.35):
            try:
                self.poll()
            except (urllib.error.URLError, OSError, ValueError):
                pass

    def _fetch(self):
        try:
            return http_json(f"{self.base}/live?offset={self.offset}", timeout=2.0)
        except urllib.error.HTTPError as e:
            if e.code == 404:  # a server without /live: whole-step output only
                self.enabled = False
            raise

    def poll(self, final: bool = False) -> None:
        """Consume new text of the request on the engine. `final`: the step ended; flush its text."""
        with io_lock:
            js = self._fetch()
            rid = js.get("id")
            if rid is None or rid in self.done:
                return
            if rid != self.req:
                if final:  # the step's request already left /live (a newer one runs)
                    return
                self._settle()
                self._reset(rid)
                js = self._fetch()
            phase = js.get("phase")
            if phase == "prefill":
                img = f"，{js['images']} 张图" if js.get("images") else ""
                self._status(f"⏳ 读入提示 {js.get('prompt_tokens', 0):,} tok（缓存命中 {js.get('cached') or 0:,}{img}）… {js.get('elapsed_s', 0):.0f}s")
                return
            text = js.get("text") or ""
            self.offset += len(text)
            self._feed(text, flush=final or phase == "done")
            if final:
                self.done.add(rid)
                self._settle()
            elif self.mode == "hidden" and phase == "decode":
                self._status(f"… 生成工具调用（{js.get('generated', 0)} tok）")

    def step_finished(self) -> bool:
        """Called when the harness reports a finished model step: flush it; True when its text was
        (at least partly) shown live, so the caller prints only the tool calls."""
        if not self.enabled or self.req is None or self.req in self.done:
            return False
        try:
            self.poll(final=True)
        except (urllib.error.URLError, OSError, ValueError):
            pass
        with io_lock:
            if self.req is not None:
                self.done.add(self.req)
            self._settle()
        return self.shown

    # -- rendering (caller holds io_lock)
    def _feed(self, text: str, flush: bool) -> None:
        buf = self.pending + text
        self.pending = ""
        while buf and self.mode != "hidden":
            hits = [(buf.find(m), m) for m in self.MARKERS if m in buf]
            if hits:
                i, m = min(hits)
                self._emit(buf[:i])
                buf = buf[i + len(m):]
                if m == "</think>":
                    self.mode = "answer"
                    self._settle()
                else:
                    self.mode = "hidden"
                continue
            if not flush:  # hold back a tail that may be the start of a marker
                keep = max((k for m in self.MARKERS for k in range(1, len(m)) if buf.endswith(m[:k])), default=0)
                if keep:
                    self.pending, buf = buf[-keep:], buf[:-keep]
            self._emit(buf)
            return

    def _emit(self, s: str) -> None:
        if (self.mode == "answer" and not self.answer_started) or (self.mode == "reason" and not self.shown):
            s = s.lstrip("\n")  # no blank lines at the start of the reasoning or of the answer
        if not s:
            return
        self._clear_status()
        sys.stdout.write(c(GRAY, s) if self.mode == "reason" else s)
        sys.stdout.flush()
        self.shown = True
        self.answer_started = self.answer_started or self.mode == "answer"
        self.line_open = not s.endswith("\n")

    def _status(self, s: str) -> None:
        if not TTY or (self.status_shown and s == self.last_status):
            return
        if self.line_open:
            sys.stdout.write("\n")
            self.line_open = False
        sys.stdout.write("\r\033[K" + c(GRAY, s))
        sys.stdout.flush()
        self.status_shown, self.last_status = True, s

    def _clear_status(self) -> None:
        if self.status_shown:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
            self.status_shown, self.last_status = False, ""

    def _settle(self) -> None:
        """Leave the cursor at the start of an empty line."""
        self._clear_status()
        if self.line_open:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self.line_open = False


# ------------------------------------------------------------------------------ input parsing

_PATH = re.compile(r"""'([^']+)'|"([^"]+)"|((?:~|/)(?:\\.|[^\s\\'"])+)""")


def find_paths(line: str) -> list[tuple[str, str]]:
    """(raw text, resolved absolute path) for every existing file or directory named in `line`,
    in the forms Terminal / iTerm2 / VS Code insert on drag-and-drop: `\\ `-escaped or quoted."""
    found = []
    for m in _PATH.finditer(line):
        raw = m.group(0)
        cand = m.group(1) or m.group(2) or re.sub(r"\\(.)", r"\1", m.group(3))
        p = os.path.abspath(os.path.expanduser(cand))
        if os.path.exists(p):
            found.append((raw, p))
    return found


def image_block(path: str) -> dict:
    p = Path(path)
    data = p.read_bytes()
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    if p.suffix.lower() in (".heic", ".heif", ".tif", ".tiff", ".bmp"):  # re-encode with macOS sips
        with tempfile.TemporaryDirectory() as d:
            dst = Path(d) / (p.stem + ".jpg")
            subprocess.run(["sips", "-s", "format", "jpeg", str(p), "--out", str(dst)], check=True, capture_output=True)
            data, mime = dst.read_bytes(), "image/jpeg"
    return {"type": "image", "data": base64.b64encode(data).decode(), "mimeType": mime}


def clipboard_image() -> str | None:
    """Save the clipboard image (PNG) to a temp file; None when the clipboard holds no image."""
    dst = Path(tempfile.gettempdir()) / f"dsv41-clipboard-{int(time.time())}.png"
    script = (
        'set f to POSIX file "%s"\n'
        "try\n set d to (the clipboard as «class PNGf»)\non error\n return \"none\"\nend try\n"
        "set h to open for access f with write permission\nset eof of h to 0\nwrite d to h\nclose access h\nreturn \"ok\""
    ) % dst
    r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    return str(dst) if r.stdout.strip() == "ok" and dst.exists() else None


# ------------------------------------------------------------------------------ @ mentions
# DeepSeek Harness's own grammar: `@path` at the start of input or after whitespace (an e-mail
# address does not count), `@"path with spaces"`, relative paths from the workspace root. dsh
# sends a mention as plain text and leaves reading to the model; here the file goes along with the
# prompt instead (one model step less at ~10 tok/s): text files inline, images attached,
# directories as a short tree. `@path:10-40` sends only those lines.

MENTION = re.compile(r'(?<!\S)@(?:"([^"\n]+)"|((?:\\.|[^\s"\\])+))')
MENTION_MAX_BYTES = int(os.environ.get("DSV41_MENTION_MAX_BYTES", 48_000))  # ~12K tokens of code
BINARY_EXT = {".pdf", ".zip", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar", ".dmg", ".pkg", ".app", ".so", ".dylib",
              ".o", ".a", ".bin", ".safetensors", ".npy", ".npz", ".pt", ".pth", ".onnx", ".mp3", ".mp4", ".mov",
              ".wav", ".docx", ".xlsx", ".pptx", ".key", ".numbers", ".pages", ".sqlite", ".db", ".jar", ".class"}
IGNORE_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
               ".tox", "build", "dist", ".idea", ".vscode", ".DS_Store", ".next", "target"}


def est_tokens(s: str) -> int:
    cjk = sum(1 for ch in s if ord(ch) >= 0x2E80)
    return int((len(s) - cjk) / 3.6 + cjk * 0.8) + 1


def resolve_mention(token: str, cwd: str) -> tuple[str | None, tuple[int, int] | None, str]:
    """-> (absolute existing path or None, line range, trailing text the token swallowed)."""
    def find(t: str):
        p = os.path.abspath(os.path.join(cwd, os.path.expanduser(t)))
        if os.path.exists(p):
            return p, None
        m = re.fullmatch(r"(.+?):(\d+)(?:-(\d+))?", t)
        if m:
            p = os.path.abspath(os.path.join(cwd, os.path.expanduser(m.group(1))))
            if os.path.isfile(p):
                a = int(m.group(2))
                return p, (a, int(m.group(3) or a))
        return None, None

    # Chinese text often runs straight on after the path ("看 @notes.txt，第一行是什么"): take the
    # longest prefix that exists, cutting only where a non-path character follows (so a typo such
    # as @notes.txtx stays unresolved instead of quietly matching notes.txt)
    for i in range(len(token), 0, -1):
        if i < len(token) and re.match(r"[A-Za-z0-9_.~/\-]", token[i]):
            continue
        p, rng = find(token[:i])
        if p:
            return p, rng, token[i:]
    return None, None, ""


def dir_tree(path: str, depth: int = 2, limit: int = 200) -> tuple[str, int]:
    """Indented listing of `path`, `depth` levels deep, directories first, hidden and build/VCS
    directories skipped. -> (listing, entries seen); scanning stops after 5000 entries."""
    out: list[str] = []
    seen = [0]

    def walk(d: str, level: int) -> None:
        try:
            entries = sorted(os.scandir(d), key=lambda e: (not e.is_dir(follow_symlinks=False), e.name.lower()))
        except OSError:
            return
        for e in entries:
            if e.name.startswith(".") or e.name in IGNORE_DIRS or seen[0] > 5000:
                continue
            seen[0] += 1
            is_dir = e.is_dir(follow_symlinks=False)
            if len(out) < limit:
                out.append("  " * level + e.name + ("/" if is_dir else ""))
            if is_dir and level + 1 < depth:
                walk(e.path, level + 1)

    walk(path, 0)
    more = seen[0] - len(out)
    return "\n".join(out) + (f"\n… 另有 {more} 项未列出" if more > 0 else ""), seen[0]


def read_text(path: str, rng: tuple[int, int] | None) -> tuple[str | None, int, bool]:
    """-> (text or None when binary, total lines, truncated)."""
    if Path(path).suffix.lower() in BINARY_EXT:
        return None, 0, False
    with open(path, "rb") as f:
        raw = f.read(MENTION_MAX_BYTES * 8 if rng else MENTION_MAX_BYTES + 1)
    if b"\0" in raw[:8192]:
        return None, 0, False
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        if e.start < len(raw) - 4:  # a cut multi-byte char at the end is fine, anything earlier is binary
            return None, 0, False
        text = raw[: e.start].decode("utf-8")
    lines = text.splitlines()
    if rng:
        a, b = max(1, rng[0]), max(rng[0], rng[1])
        return "\n".join(lines[a - 1 : b]), len(lines), False
    truncated = os.path.getsize(path) > MENTION_MAX_BYTES
    if truncated:
        text = text[:MENTION_MAX_BYTES].rsplit("\n", 1)[0]
    return text, len(lines), truncated


def expand_mentions(line: str, cwd: str) -> tuple[str, list[dict], list[str], list[str]]:
    """-> (prompt text, extra text blocks with file contents, image paths, one note per mention)."""
    blocks, images, notes = [], [], []
    out, last = [], 0
    for m in MENTION.finditer(line):
        token = m.group(1) if m.group(1) is not None else re.sub(r"\\(.)", r"\1", m.group(2))
        path, rng, tail = resolve_mention(token, cwd)
        out.append(line[last : m.start()])
        last = m.end()
        if path is None:
            out.append(m.group(0))
            notes.append(c(GRAY, f"？未找到 @{token}，按原文发送"))
            continue
        shown = os.path.relpath(path, cwd) if path == cwd or path.startswith(cwd + os.sep) else path
        if os.path.isdir(path):
            shown = shown.rstrip("/") + "/"
        mention = f'@"{shown}"' if " " in shown else f"@{shown}"
        if rng:
            mention += f":{rng[0]}-{rng[1]}"
        out.append(mention + tail)
        name = os.path.basename(path.rstrip("/")) or path
        if os.path.isdir(path):
            tree, n = dir_tree(path)
            blocks.append({"type": "text", "text": f'<directory path="{path}">\n{tree}\n</directory>'})
            notes.append(f"📁 {name}/（{n} 项，约 {est_tokens(tree):,} tok）")
        elif Path(path).suffix.lower() in IMAGE_EXT:
            if path not in images:
                images.append(path)
            notes.append(f"🖼 {name}")
        else:
            text, n_lines, truncated = read_text(path, rng)
            if text is None:
                notes.append(f"📦 {name}（二进制，只传路径，agent 可自行处理）")
                continue
            attr = f' lines="{rng[0]}-{rng[1]}"' if rng else ""
            more = "\n[文件较大，只附上了开头部分；需要其余内容请用 sed 分段读取]" if truncated else ""
            blocks.append({"type": "text", "text": f'<file path="{path}"{attr}>\n{text}\n</file>{more}'})
            span = f"第 {rng[0]}-{rng[1]} 行" if rng else f"{n_lines} 行"
            notes.append(f"📄 {name}（{span}，约 {est_tokens(text):,} tok{'，已截断' if truncated else ''}）")
    out.append(line[last:])
    return "".join(out), blocks, images, notes


def build_input(line: str, extra_images: list[str] | None = None, cwd: str | None = None) -> tuple[list[dict], list[str]]:
    """-> (content blocks, notes to show). @ mentions first (see above); then dragged-in paths:
    images among them are attached, every path stays in the text, unescaped and quoted."""
    cwd = cwd or os.getcwd()
    text, file_blocks, images, notes = expand_mentions(line, cwd)
    images = list(extra_images or []) + [p for p in images if p not in (extra_images or [])]
    for raw, path in find_paths(text):
        if raw.startswith("@"):
            continue
        if raw != path:  # escaped / quoted / ~ forms become one plain quoted path
            text = text.replace(raw, f'"{path}"', 1)
        if Path(path).is_file() and Path(path).suffix.lower() in IMAGE_EXT and path not in images:
            images.append(path)
            notes.append(f"🖼 {Path(path).name}")
    for p in extra_images or []:
        notes.append(f"🖼 {Path(p).name}")
    text = text.strip() or ("请看这张图片。" if images else "请看附上的内容。")
    blocks = [{"type": "text", "text": text}] + file_blocks + [image_block(p) for p in images]
    return blocks, notes


class Completer:
    """Tab completion of @ mentions (and slash commands) for readline/libedit: `@src/ma<Tab>` completes
    the path, `@main<Tab>` searches the workspace by name (git-tracked files when in a repo)."""

    COMMANDS = ["/help", "/new", "/sessions", "/resume", "/effort", "/img", "/paste", "/stats", "/exit"]

    def __init__(self, cwd: str):
        self.cwd, self.index, self.t_index, self.matches = cwd, [], 0.0, []

    def complete(self, text: str, state: int):
        if state == 0:
            try:
                self.matches = self._matches(text)
            except Exception:  # noqa: BLE001 - completion must never break input
                self.matches = []
        return self.matches[state] if state < len(self.matches) else None

    def _matches(self, text: str) -> list[str]:
        if text.startswith("/") and not text.startswith("//"):
            hits = [x for x in self.COMMANDS if x.startswith(text)]
            if hits:
                return hits
        if not text.startswith("@"):
            return []
        frag = text[1:].lstrip('"')
        if frag.startswith(("~", "/", ".")) or "/" in frag:
            return self._paths(frag)
        return self._search(frag)

    @staticmethod
    def _mention(rel: str, is_dir: bool) -> str:
        rel = rel + ("/" if is_dir and not rel.endswith("/") else "")
        return f'@"{rel}' + ("" if is_dir else '"') if " " in rel else f"@{rel}"

    def _paths(self, frag: str) -> list[str]:
        head, part = os.path.split(frag)
        base = os.path.join(self.cwd, os.path.expanduser(head)) if head else self.cwd
        try:
            names = sorted(os.listdir(base))
        except OSError:
            return []
        out = []
        for n in names:
            if n.startswith(".") and not part.startswith("."):
                continue
            if n.lower().startswith(part.lower()):
                out.append(self._mention(os.path.join(head, n) if head else n, os.path.isdir(os.path.join(base, n))))
        return out[:200]

    def _files(self) -> list[str]:
        if self.index and time.time() - self.t_index < 30:
            return self.index
        files: list[str] = []
        try:
            r = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=self.cwd, capture_output=True,
                               text=True, timeout=3)
            if r.returncode == 0:
                files = r.stdout.splitlines()[:50_000]
        except (OSError, subprocess.SubprocessError):
            pass
        if not files:
            t0 = time.time()
            for root, dirs, fs in os.walk(self.cwd):
                dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]
                rel = os.path.relpath(root, self.cwd)
                files += [f if rel == "." else os.path.join(rel, f) for f in fs if not f.startswith(".")]
                if len(files) > 20_000 or time.time() - t0 > 0.5:
                    break
        dirs = sorted({os.path.dirname(f) + "/" for f in files if os.path.dirname(f)})
        self.index, self.t_index = files + dirs, time.time()
        return self.index

    def _search(self, frag: str) -> list[str]:
        if not frag:
            return self._paths("")
        q = frag.lower()
        ranked = []
        for f in self._files():
            name = os.path.basename(f.rstrip("/")).lower()
            if name.startswith(q):
                rank = 0
            elif q in name:
                rank = 1
            elif q in f.lower():
                rank = 2
            else:
                continue
            ranked.append((rank, len(f), f))
        return [self._mention(f.rstrip("/"), f.endswith("/")) for _, _, f in sorted(ranked)[:40]]


# ------------------------------------------------------------------------------ the session


class Term:
    def __init__(self, a):
        self.a = a
        self.base = a.base_url.rstrip("/")
        self.effort = a.effort
        self.home = Path(a.home)
        self.cwd = str(Path.cwd().resolve())
        self.h = None
        self.session = None
        self.live = Live(self.base, thinking=self.effort != "off", enabled=TTY)
        self.approver: threading.Thread | None = None

    # -- harness lifecycle
    def _config(self):
        from deepseek_harness import DeepSeekHarnessConfig

        env = {
            "DSH_TELEMETRY_MODE": "DISABLED",
            "DSV41_CONTEXT": str(self.a.context),
            "DSV41_MAX_TOKENS": str(self.a.max_tokens),
            "DSH_PERMISSION_MODE": self.a.permission,
        }
        return DeepSeekHarnessConfig(
            dsh_home=str(self.home), cwd=self.cwd, profile="sdk-minimal", patches=(str(PATCH),),
            provider="deepseek-official", model="deepseek-flash",
            reasoning_effort=self.effort, max_tokens=self.a.max_tokens,
            base_url=self.base, api_key=os.environ.get("DEEPSEEK_API_KEY") or "local",
            approval_requests=True, env=env, initialize_timeout_seconds=120, shutdown_timeout_seconds=5,
        )

    def start(self, resume: str | None = None) -> None:
        from deepseek_harness import DeepSeekHarness

        self.h = DeepSeekHarness(self._config())
        self.h.start()
        self.approver = threading.Thread(target=self._approvals, args=(self.h,), daemon=True)
        self.approver.start()
        self.session = self.h.resume_session(resume) if resume else self.h.start_session()
        self.live = Live(self.base, thinking=self.effort != "off", enabled=TTY)

    def restart(self, effort: str | None = None) -> None:
        sid = self.session.id if self.session else None
        self.close()
        if effort:
            self.effort = effort
        started = self._session_exists(sid)
        self.start(resume=sid if started else None)
        if not started and sid:  # nothing persisted yet: keep the id for the next prompt
            self.session = self.h.start_session(sid)

    def close(self) -> None:
        if self.h is not None:
            try:
                self.h.close()
            except Exception:  # noqa: BLE001 - shutting down anyway
                pass
            self.h = None

    def _approvals(self, h) -> None:
        """Answer the runtime's approval requests from the terminal (fail closed on anything else)."""
        while True:
            try:
                req = h.client.next_request()
            except Exception:  # noqa: BLE001 - runtime closed
                return
            try:
                if req.method != "approval/request":
                    h.client.respond_error(req.id, code=-32601, message="unsupported by dsv41 terminal")
                    continue
                p = req.payload if isinstance(req.payload, dict) else {}
                with io_lock:
                    clear_status()
                    out(c(YELLOW, f"\n⚠ 需要授权：{p.get('toolName', '?')} — {p.get('reason') or ''}"))
                    detail = json.dumps({k: v for k, v in p.items() if k not in ('toolName', 'reason')}, ensure_ascii=False)
                    out(c(YELLOW, "  " + detail[:600]))
                    ans = input(c(YELLOW, "  允许这一次？[y/N] ")).strip().lower()
                h.client.respond(req.id, {"outcome": "allowed-once" if ans in ("y", "yes", "是") else "rejected"})
            except Exception:  # noqa: BLE001 - never leave a request hanging open
                try:
                    h.client.respond(req.id, {"outcome": "rejected"})
                except Exception:  # noqa: BLE001
                    pass

    # -- sessions
    def _sessions(self) -> list[tuple[float, str, str]]:
        """(mtime, id, title) of this workspace's sessions, newest first."""
        rows = []
        for f in (self.home / "sessions").glob("*/session-*/session.v*.jsonl"):
            try:
                with f.open() as fh:
                    head = json.loads(fh.readline())
                    if head.get("cwd") != self.cwd:
                        continue
                    title = ""
                    for line in fh:
                        if '"session/title"' in line:
                            title = json.loads(line).get("data", {}).get("title", title)
                rows.append((f.stat().st_mtime, head["id"], title))
            except (OSError, ValueError, KeyError):
                continue
        return sorted(rows, reverse=True)

    def _session_exists(self, sid: str | None) -> bool:
        return bool(sid) and any(s[1] == sid for s in self._sessions())

    # -- one prompt
    def ask(self, blocks: list[dict]) -> str | None:
        t0 = time.time()
        steps = {"n": 0}
        final = {"reason": None}

        def on(n):
            ev = n.payload.get("event") if isinstance(n.payload, dict) else None
            if n.method != "session.event" or not isinstance(ev, dict) or n.payload.get("sessionId") != self.session.id:
                return
            kind, data = ev.get("type"), ev.get("data") or {}
            if kind == "assistant/message":
                steps["n"] += 1
                shown = self.live.step_finished()
                self._show_message(data.get("message") or {}, shown)
            elif kind == "tool/result":
                self._show_result(data.get("message") or {})
            elif kind == "llm/retry":
                out(c(YELLOW, f"↻ 模型请求重试：{json.dumps(data, ensure_ascii=False)[:200]}"))
            elif kind == "turn/end":
                final["reason"] = (data.get("reason") or {}).get("kind")
                if final["reason"] not in ("completed", None):
                    out(c(RED, f"■ 本轮结束：{json.dumps(data.get('reason'), ensure_ascii=False)[:400]}"))

        self.live.start()
        try:
            r = self.session.run(blocks, on_notification=on)
        finally:
            self.live.stop()
        took = time.time() - t0
        out(c(GRAY, f"── {steps['n']} 步 · {took:.0f}s · {r.finish_reason}"))
        return r.final_response

    def _show_message(self, msg: dict, shown_live: bool) -> None:
        with io_lock:
            clear_status()
            for b in msg.get("content") or []:
                t = b.get("type")
                if t == "reasoning" and not shown_live and b.get("text"):
                    out(c(GRAY, b["text"].strip()))
                elif t == "text" and not shown_live and b.get("text"):
                    out(b["text"].strip())
                elif t == "tool-call":
                    try:
                        args = json.loads(b.get("arguments") or "{}")
                    except ValueError:
                        args = {"arguments": b.get("arguments")}
                    if b.get("name") == "bash" and "command" in args:
                        body = args["command"]
                    else:
                        body = json.dumps(args, ensure_ascii=False)
                    lines = body.splitlines() or [""]
                    out(c(CYAN, f"⚙ {b.get('name')}: ") + c(CYAN, lines[0][: width() - 12]))
                    for ln in lines[1:8]:
                        out(c(CYAN, "    " + ln[: width() - 6]))
                    if len(lines) > 8:
                        out(c(CYAN, f"    … 另有 {len(lines) - 8} 行"))

    def _show_result(self, msg: dict) -> None:
        texts, n_img = [], 0
        for b in msg.get("content") or []:
            if b.get("type") == "text":
                texts.append(b.get("text") or "")
            elif b.get("type") == "image":
                n_img += 1
        lines = "\n".join(texts).rstrip().splitlines()
        color = RED if msg.get("isError") else GRAY
        with io_lock:
            clear_status()
            for ln in lines[:10]:
                out(c(color, "  │ " + ln[: width() - 6]))
            if len(lines) > 10:
                out(c(color, f"  │ … 另有 {len(lines) - 10} 行"))
            if n_img:
                out(c(color, f"  │ [{n_img} 张图片已交给模型]"))

    # -- REPL
    def banner(self) -> None:
        mode = os.environ.get("DSV41_SERVER_MODE", "?")
        perm = {"workspace-write": "只能写当前目录和 /tmp", "read-only": "只读", "danger-full-access": "完全访问（不受沙箱限制）"}
        out(c(BOLD, "DeepSeek-V4.1-Flash · DeepSeek Harness（minimal）· 本地 DeepInfer-Mac"))
        out(c(GRAY, f"  服务 {self.base}（{mode}）· 推理强度 {self.effort} · 上下文 {self.a.context:,} · 单步输出上限 {self.a.max_tokens:,}"))
        out(c(GRAY, f"  工作目录 {self.cwd} · 权限：{perm.get(self.a.permission, self.a.permission)}"))
        out(c(GRAY, f"  会话 {self.session.id}"))
        out(c(GRAY, "  @文件 引用（Tab 补全）· 拖入文件 = 附上路径（图片直接给模型看）· /paste 贴剪贴板图片 · /help · Ctrl+C 中断 · Ctrl+D 退出"))

    HELP = """引用本地文件（@，Tab 补全）：
  @notes.txt 解释一下      文本文件：内容随问题一起发给模型（超过约 48 KB 只附开头）
  @src/main.py:40-80 …     只附这几行
  @"my file.md" …          路径有空格时加引号（Tab 补全会自动加）
  @src/ 有哪些模块？        目录：附上两层目录树
  @截图.png 这是什么？      图片：直接给模型看
  @ma<Tab>                 按文件名在工作目录里查找；@src/<Tab> 按路径补全
  相对路径从工作目录算起；也可以写绝对路径或 ~/…；邮箱里的 @ 不算。
  拖入文件（不加 @）只把路径交给 agent，由它自己去读；拖入的图片会直接附上。
命令：
  /help                 本帮助
  /new                  开始新会话
  /sessions             列出本目录的历史会话
  /resume [id|序号]     恢复会话（默认最近一个）
  /effort off|low|high|max   推理强度（off = 不思考，最快；high/max 更强更慢）
  /img <路径> [问题]     附上图片（路径里有空格时可加引号）
  /paste [问题]          附上剪贴板里的图片（截图用 Ctrl+Shift+Cmd+4）
  /stats                服务端上一次请求的统计
  /exit                 退出（或 Ctrl+D）
多行输入：行尾加 \\ 续行；直接粘贴多行文本会合并成一条。"""

    def command(self, line: str) -> bool:
        """Handle a slash command; False when it is not one of ours (then it goes to the model)."""
        cmd, _, rest = line.partition(" ")
        rest = rest.strip()
        if cmd in ("/exit", "/quit"):
            raise EOFError
        if cmd == "/help":
            out(self.HELP)
        elif cmd == "/new":
            self.session = self.h.start_session()
            out(c(GREEN, f"新会话 {self.session.id}"))
        elif cmd == "/sessions":
            rows = self._sessions()[:20]
            if not rows:
                out("本目录还没有保存的会话。")
            for i, (mt, sid, title) in enumerate(rows, 1):
                cur = " ←当前" if self.session and sid == self.session.id else ""
                out(f"  {i:2d}. {time.strftime('%m-%d %H:%M', time.localtime(mt))}  {sid}  {title[:40]}{cur}")
        elif cmd == "/resume":
            rows = self._sessions()
            if self.session:
                rows = [r for r in rows if r[1] != self.session.id] or rows
            if not rows:
                out("本目录没有可恢复的会话。")
                return True
            sid = rows[0][1]
            if rest.isdigit() and 1 <= int(rest) <= len(self._sessions()):
                sid = self._sessions()[int(rest) - 1][1]
            elif rest:
                sid = rest
            self.close()
            self.start(resume=sid)
            out(c(GREEN, f"已恢复会话 {sid}"))
        elif cmd == "/effort":
            if rest not in EFFORTS:
                out(f"用法：/effort {'|'.join(EFFORTS)}（当前 {self.effort}）")
            else:
                self.restart(effort=rest)
                out(c(GREEN, f"推理强度 = {rest}（会话 {self.session.id} 已保留）"))
        elif cmd == "/stats":
            for path in ("/stats", "/live"):
                try:
                    js = http_json(self.base + path)
                    js.pop("text", None)
                    out(f"{path}: {json.dumps(js, ensure_ascii=False)}")
                except Exception as e:  # noqa: BLE001
                    out(f"{path}: {e}")
        elif cmd in ("/img", "/paste"):
            if cmd == "/paste":
                img = clipboard_image()
                if not img:
                    out(c(RED, "剪贴板里没有图片。"))
                    return True
                self.send(rest, [img])
            else:
                paths = find_paths(rest)
                if not paths:
                    out(c(RED, "找不到文件。用法：/img <路径> [问题]"))
                    return True
                raw, path = paths[0]
                self.send(rest.replace(raw, "", 1), [path])
        else:
            return False
        return True

    def send(self, text: str, images: list[str] | None = None) -> None:
        blocks, notes = build_input(text, images, cwd=self.cwd)
        for n in notes:
            out(c(GRAY, "  " + n))
        try:
            self.ask(blocks)
        except KeyboardInterrupt:
            self.live.stop()
            out(c(YELLOW, "\n⏹ 已中断。重启 harness 并恢复会话…"))
            self.restart()
            out(c(GRAY, "（服务端会在当前这一步收尾后再处理下一个请求）"))
        except Exception as e:  # noqa: BLE001 - keep the REPL alive; the session is persisted
            self.live.stop()
            out(c(RED, f"\n✗ harness 出错：{type(e).__name__}: {str(e)[:500]}"))
            out(c(YELLOW, "  重启 harness 并恢复会话…"))
            self.restart()

    def read(self) -> str:
        prompt = "\001\033[1;36m\002dsv41 ❯\001\033[0m\002 " if TTY else "dsv41> "
        line = input(prompt)
        lines = [line]
        while lines[-1].endswith("\\"):
            lines[-1] = lines[-1][:-1]
            lines.append(input("… "))
        if sys.stdin.isatty():  # a multi-line paste arrives at once: keep it as one prompt
            while select.select([sys.stdin], [], [], 0.05)[0]:
                lines.append(input(""))
        return "\n".join(lines).strip()

    def repl(self) -> None:
        self.banner()
        while True:
            try:
                line = self.read()
            except KeyboardInterrupt:
                out("")
                continue
            except EOFError:
                out("")
                return
            if not line:
                continue
            try:
                if line.startswith("/") and self.command(line):
                    continue
            except EOFError:
                return
            except KeyboardInterrupt:
                out("")
                continue
            except Exception as e:  # noqa: BLE001 - a failed command must not end the REPL
                out(c(RED, f"✗ {type(e).__name__}: {str(e)[:500]}"))
                if self.h is None:
                    self.start()
                continue
            self.send(line)


def server_ok(base: str) -> bool:
    try:
        return any(m.get("id") == "deepseek-flash" for m in http_json(base.rstrip("/") + "/v1/models", timeout=3).get("data", []))
    except Exception:  # noqa: BLE001
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dsv41", description="DeepSeek Harness terminal agent on the local DeepSeek-V4.1-Flash")
    ap.add_argument("-p", "--prompt", action="append", help="run this prompt (repeatable) and exit")
    ap.add_argument("--image", action="append", default=[], help="attach an image to the first --prompt")
    ap.add_argument("--resume", nargs="?", const="last", help="resume a session of this directory (id, or the last one)")
    ap.add_argument("--effort", choices=EFFORTS, default=os.environ.get("DSV41_EFFORT", "low"))
    ap.add_argument("--context", type=int, default=int(os.environ.get("DSV41_CONTEXT", 65536)))
    ap.add_argument("--max-tokens", type=int, default=int(os.environ.get("DSV41_MAX_TOKENS", 16384)))
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--full-access", dest="permission", action="store_const", const="danger-full-access")
    g.add_argument("--read-only", dest="permission", action="store_const", const="read-only")
    ap.set_defaults(permission=os.environ.get("DSH_PERMISSION_MODE", "workspace-write"))
    ap.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--home", default=os.environ.get("DSH_HOME", str(DEFAULT_HOME)))
    a = ap.parse_args(argv)

    if not server_ok(a.base_url):
        print(f"本地模型服务 {a.base_url} 没有响应。先运行：dsv41 serve", file=sys.stderr)
        return 2
    Path(a.home).mkdir(parents=True, exist_ok=True)
    t = Term(a)
    resume = None
    if a.resume:
        rows = t._sessions()
        resume = (rows[0][1] if rows else None) if a.resume == "last" else a.resume
        if a.resume == "last" and not resume:
            print("本目录没有可恢复的会话，开始新会话。", file=sys.stderr)
    try:
        t.start(resume=resume)
    except Exception as e:  # noqa: BLE001
        print(f"DeepSeek Harness 启动失败：{e}", file=sys.stderr)
        return 3
    try:
        if a.prompt:
            for i, p in enumerate(a.prompt):
                blocks, notes = build_input(p, a.image if i == 0 else None, cwd=t.cwd)
                for n in notes:
                    out(c(GRAY, "  " + n))
                t.ask(blocks)
            return 0
        try:
            import readline  # line editing (CJK backspace, arrows), history and @ completion for input()

            hist = Path(a.home).parent / "repl_history"
            try:
                readline.read_history_file(hist)
            except OSError:
                pass
            readline.set_history_length(2000)
            completer = Completer(t.cwd)
            readline.set_completer(completer.complete)
            readline.set_completer_delims(" \t\n")
            if "libedit" in (readline.__doc__ or ""):  # macOS / uv Pythons ship libedit
                readline.parse_and_bind("bind ^I rl_complete")
            else:
                readline.parse_and_bind("tab: complete")
        except ImportError:
            hist = None
        try:
            t.repl()
        finally:
            if hist is not None:
                try:
                    readline.write_history_file(hist)
                except OSError:
                    pass
        return 0
    finally:
        t.close()


if __name__ == "__main__":
    sys.exit(main())
