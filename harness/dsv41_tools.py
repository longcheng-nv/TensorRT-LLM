#!/usr/bin/env python3
"""MCP tools for the dsv41 terminal agent (DeepSeek Harness on the local DeepSeek-V4.1-Flash).

  dsv41_tools.py search    server "web":      search — Bing, DuckDuckGo fallback, no API key
  dsv41_tools.py browser   server "browser":  open / read / screenshot / click / type /
                                              wait_for_login (the window closes with the session)

Mounted through DeepSeek Harness's own MCP client by patches/sdk-minimal.local.yml (tools appear
as mcp__web__search, mcp__browser__open, ...); the agent reads plain pages with dsh's web_fetch.

The browser is the installed Google Chrome on a dedicated profile (default ~/Models/dsh/
browser-profile, never the everyday Chrome profile), in a visible window: when a site needs a
login the window comes to the front, the user signs in there (passwords never pass through the
model — typing into password fields is refused), and the agent continues on the signed-in page.
Logins persist in that profile. DSV41_BROWSER_ACTIONS=0 drops click/type (read-only browsing).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from html import unescape
from pathlib import Path

from mcp.server.mcpserver import Image, MCPServer
from mcp.types import ToolAnnotations

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/131.0.0.0 Safari/537.36")
READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=True)
ACTS = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True)


# ------------------------------------------------------------------------------ search


def _get(url: str, timeout: float = 15.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # honours HTTP(S)_PROXY / NO_PROXY
        return r.read().decode(r.headers.get_content_charset() or "utf-8", "replace")


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", html))).strip()


def _unbing(url: str) -> str:
    """bing.com/ck/a?...&u=a1<base64url> -> the target URL."""
    if "bing.com/ck/a" not in url:
        return url
    u = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get("u", [""])[0]
    if u.startswith("a1"):
        b = u[2:]
        with contextlib.suppress(ValueError, UnicodeDecodeError):
            return base64.urlsafe_b64decode(b + "=" * (-len(b) % 4)).decode()
    return url


def _bing(query: str, n: int) -> list[tuple[str, str, str]]:
    html = _get("https://www.bing.com/search?" + urllib.parse.urlencode({"q": query, "count": str(min(30, n * 2))}))
    out = []
    for block in re.findall(r'<li class="b_algo[^"]*"[^>]*>(.*?)</li>', html, re.S):
        m = re.search(r'<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
        if not m:
            continue
        snip = re.search(r'<p[^>]*>(.*?)</p>', block, re.S) or re.search(r'class="b_caption[^"]*"[^>]*>(.*?)</div>', block, re.S)
        out.append((_text(m.group(2)), _unbing(unescape(m.group(1))), _text(snip.group(1)) if snip else ""))
    return out


def _duckduckgo(query: str, n: int) -> list[tuple[str, str, str]]:
    html = _get("https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query}))
    out = []
    for m in re.finditer(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>(.*?)(?=<a[^>]+class="result__a"|$)', html, re.S):
        href = unescape(m.group(1))
        q = urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("uddg")
        snip = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', m.group(3), re.S)
        out.append((_text(m.group(2)), q[0] if q else href, _text(snip.group(1)) if snip else ""))
    return out


def run_search(query: str, max_results: int = 8) -> str:
    n = max(1, min(int(max_results), 20))
    errors = []
    for name, engine in (("Bing", _bing), ("DuckDuckGo", _duckduckgo)):
        try:
            hits = [h for h in engine(query, n) if h[1].startswith("http")][:n]
        except Exception as e:  # noqa: BLE001 - try the next engine
            errors.append(f"{name}: {type(e).__name__}: {e}")
            continue
        if hits:
            lines = [f"搜索「{query}」（{name}，{len(hits)} 条）："]
            for i, (title, url, snip) in enumerate(hits, 1):
                lines.append(f"{i}. {title}\n   {url}" + (f"\n   {snip[:300]}" if snip else ""))
            lines.append("需要细节时用 web_fetch 读取相应网址；回答时引用来源网址。")
            return "\n".join(lines)
        errors.append(f"{name}: no results")
    return "搜索失败：" + "；".join(errors)


# ------------------------------------------------------------------------------ browser


PROFILE = Path(os.environ.get("DSV41_BROWSER_PROFILE") or Path.home() / "Models" / "dsh" / "browser-profile")
LOGIN_HINT = re.compile(r"login|log-in|signin|sign-in|sign_in|/auth|oauth|/sso|passport|account/log|cas/", re.I)
TEXT_JS = "() => (document.body ? document.body.innerText : '')"
LINKS_JS = """() => Array.from(document.querySelectorAll('a[href]'))
  .filter(a => a.offsetParent !== null && a.innerText.trim())
  .slice(0, 400).map(a => [a.innerText.trim().replace(/\\s+/g, ' ').slice(0, 80), a.href])"""


class Browser:
    def __init__(self) -> None:
        self.pw = None
        self.ctx = None
        self.page = None
        self.lock = asyncio.Lock()

    async def ensure(self):
        if self.ctx is not None and self.page is not None and not self.page.is_closed():
            return self.page
        if self.ctx is not None and self.ctx.pages:
            self.page = self.ctx.pages[-1]
            return self.page
        from playwright.async_api import async_playwright

        PROFILE.mkdir(parents=True, exist_ok=True)
        self.pw = await async_playwright().start()
        try:
            self.ctx = await self.pw.chromium.launch_persistent_context(
                str(PROFILE), channel="chrome", headless=False, viewport={"width": 1280, "height": 860},
                ignore_default_args=["--enable-automation"],
                args=["--no-first-run", "--no-default-browser-check", "--disable-blink-features=AutomationControlled"])
        except Exception as e:
            await self.pw.stop()
            self.pw = None
            raise RuntimeError(f"无法启动 Chrome（需要 /Applications/Google Chrome.app；若上次的 dsv41 浏览器窗口还开着，请先关掉它）：{e}") from e
        self.ctx.on("page", self._on_page)
        self.page = self.ctx.pages[0] if self.ctx.pages else await self.ctx.new_page()
        return self.page

    def _on_page(self, page) -> None:  # links that open a new tab: follow them
        self.page = page

    async def close(self) -> None:
        with contextlib.suppress(Exception):
            if self.ctx is not None:
                await self.ctx.close()
        with contextlib.suppress(Exception):
            if self.pw is not None:
                await self.pw.stop()
        self.ctx = self.pw = self.page = None

    async def login_required(self, page) -> bool:
        with contextlib.suppress(Exception):
            if await page.locator("input[type=password]").first.is_visible(timeout=500):
                return True
        return bool(LOGIN_HINT.search(page.url)) and await page.locator("form").count() > 0

    async def summary(self, page, head: str = "", chars: int = 3000) -> str:
        with contextlib.suppress(Exception):
            await page.wait_for_load_state("networkidle", timeout=4000)
        title = await page.title()
        text = re.sub(r"\n\s*\n+", "\n", await page.evaluate(TEXT_JS)).strip()
        links = await page.evaluate(LINKS_JS)
        parts = [head] if head else []
        parts += [f"标题：{title}", f"网址：{page.url}"]
        if await self.login_required(page):
            parts.append("⚠ 这是登录页。请告诉用户在弹出的 Chrome 窗口里自己登录（不要索要或代填密码），"
                         "然后调用 wait_for_login 等待登录完成。")
        more = f"（共 {len(text)} 字，用 read 的 start 参数翻页）" if len(text) > chars else ""
        parts.append(f"正文{more}：\n{text[:chars]}")
        if links:
            parts.append("页面链接（文字 → 网址，可用 click 点击文字）：\n" + "\n".join(f"- {t} → {u}" for t, u in links[:30]))
        return "\n".join(parts)

    async def locate(self, page, target: str, kinds: tuple[str, ...]):
        """First visible element for `target`: CSS/XPath selector, or accessible name / label /
        placeholder / visible text, in that order of the requested kinds."""
        tries = []
        if target.startswith(("css=", "xpath=", "#", ".", "[", "//")) or re.search(r"[\[\]=>]", target):
            tries.append(page.locator(target))
        for k in kinds:
            if k == "link":
                tries.append(page.get_by_role("link", name=target))
            elif k == "button":
                tries.append(page.get_by_role("button", name=target))
            elif k == "label":
                tries.append(page.get_by_label(target))
            elif k == "placeholder":
                tries.append(page.get_by_placeholder(target))
            elif k == "name":
                tries.append(page.locator(f"[name={target!r}]"))
            elif k == "text":
                tries.append(page.get_by_text(target))
        for loc in tries:
            with contextlib.suppress(Exception):
                n = await loc.count()
                for i in range(min(n, 10)):
                    el = loc.nth(i)
                    if await el.is_visible():
                        return el
        return None


BROWSER = Browser()


async def b_open(url: str) -> str:
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", url):
        url = "https://" + url
    async with BROWSER.lock:
        page = await BROWSER.ensure()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=45_000)
        except Exception as e:  # noqa: BLE001 - report, keep the page
            return f"打开失败：{type(e).__name__}: {str(e)[:300]}\n当前网址：{page.url}"
        if await BROWSER.login_required(page):
            with contextlib.suppress(Exception):
                await page.bring_to_front()
        return await BROWSER.summary(BROWSER.page or page)


async def b_read(start: int = 0, max_chars: int = 6000) -> str:
    async with BROWSER.lock:
        if BROWSER.page is None:
            return "浏览器还没有打开页面：先调用 open。"
        page = BROWSER.page
        text = re.sub(r"\n\s*\n+", "\n", await page.evaluate(TEXT_JS)).strip()
        start = max(0, int(start))
        end = start + max(500, min(int(max_chars), 20_000))
        return f"标题：{await page.title()}\n网址：{page.url}\n正文第 {start}-{min(end, len(text))} 字（共 {len(text)} 字）：\n{text[start:end]}"


async def b_screenshot(full_page: bool = False) -> Image:
    async with BROWSER.lock:
        if BROWSER.page is None:
            raise RuntimeError("浏览器还没有打开页面：先调用 open。")
        data = await BROWSER.page.screenshot(type="jpeg", quality=80, full_page=bool(full_page))
        return Image(data=data, format="jpeg")


async def b_click(target: str) -> str:
    async with BROWSER.lock:
        if BROWSER.page is None:
            return "浏览器还没有打开页面：先调用 open。"
        page = BROWSER.page
        el = await BROWSER.locate(page, target, ("link", "button", "text"))
        if el is None:
            return f"没有找到可见的「{target}」。可以先用 read 或 screenshot 看看页面上有什么。"
        before = page.url
        await el.click(timeout=10_000)
        await asyncio.sleep(1.0)
        page = BROWSER.page or page
        with contextlib.suppress(Exception):
            await page.wait_for_load_state("domcontentloaded", timeout=15_000)
        head = "已点击。" + ("（页面已跳转）" if page.url != before else "")
        return await BROWSER.summary(page, head)


async def b_type(target: str, text: str, submit: bool = False) -> str:
    async with BROWSER.lock:
        if BROWSER.page is None:
            return "浏览器还没有打开页面：先调用 open。"
        page = BROWSER.page
        el = await BROWSER.locate(page, target, ("label", "placeholder", "name"))
        if el is None:
            return f"没有找到输入框「{target}」。"
        if (await el.get_attribute("type") or "").lower() == "password":
            return "拒绝：密码由用户在浏览器窗口里自己输入（调用 wait_for_login 等待），不要经过模型。"
        await el.fill(text, timeout=10_000)
        if submit:
            await el.press("Enter")
            await asyncio.sleep(1.0)
            with contextlib.suppress(Exception):
                await (BROWSER.page or page).wait_for_load_state("domcontentloaded", timeout=15_000)
        return await BROWSER.summary(BROWSER.page or page, "已输入" + ("并提交。" if submit else "。"))


async def b_wait_for_login(timeout_seconds: int = 300) -> str:
    async with BROWSER.lock:
        if BROWSER.page is None:
            return "浏览器还没有打开页面：先调用 open。"
        page = BROWSER.page
        if not await BROWSER.login_required(page):
            return await BROWSER.summary(page, "当前页面不需要登录（或已登录）。")
        start_url = page.url
        with contextlib.suppress(Exception):
            await page.bring_to_front()
        deadline = time.time() + max(10, min(int(timeout_seconds), 600))
        while time.time() < deadline:
            await asyncio.sleep(2.0)
            page = BROWSER.page or page
            with contextlib.suppress(Exception):
                if page.url != start_url and not await BROWSER.login_required(page):
                    return await BROWSER.summary(page, "✓ 用户已完成登录。")
        return f"等待超时：用户还没有完成登录（当前网址 {page.url}）。可以问用户是否还需要更多时间。"


# ------------------------------------------------------------------------------ servers


def search_server() -> MCPServer:
    s = MCPServer("web", instructions="Web search without an API key (Bing, DuckDuckGo fallback).")

    @s.tool(name="search", annotations=READ_ONLY, description=(
        "Search the web (Bing, falling back to DuckDuckGo). Returns titles, URLs and snippets; read a "
        "page with web_fetch, or with the browser tools when it needs a login. Cite the URLs you use."))
    async def search(query: str, max_results: int = 8) -> str:
        return await asyncio.to_thread(run_search, query, max_results)

    return s


def browser_server() -> MCPServer:
    @contextlib.asynccontextmanager
    async def lifespan(_server):
        try:
            yield {}
        finally:
            await BROWSER.close()

    s = MCPServer("browser", lifespan=lifespan, instructions=(
        "Visible Chrome window, dedicated profile: for pages behind a login or needing JavaScript (plain "
        "pages: web_fetch). On a login page ask the user to sign in in the window, then call wait_for_login; "
        "never ask for or type passwords. Ask before irreversible submissions (payments, messages, deletions)."))

    s.tool(name="open", annotations=READ_ONLY, description=(
        "Open a URL in the Chrome window. Returns title, URL, text and links; flags login pages."))(b_open)
    s.tool(name="read", annotations=READ_ONLY, description="Text of the current tab from character `start`.")(b_read)
    s.tool(name="screenshot", annotations=READ_ONLY, description="Screenshot of the current tab, as an image.")(b_screenshot)
    s.tool(name="wait_for_login", annotations=READ_ONLY, description=(
        "Wait (up to timeout_seconds) until the user has signed in in the window; returns the signed-in page."))(b_wait_for_login)
    if os.environ.get("DSV41_BROWSER_ACTIONS", "1") != "0":
        s.tool(name="click", annotations=ACTS, description="Click a link/button by visible text or CSS selector.")(b_click)
        s.tool(name="type", annotations=ACTS, description=(
            "Fill an input (label, placeholder, name or CSS selector); submit=true presses Enter. Not for passwords."))(b_type)
    return s


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which == "search":
        search_server().run("stdio")
    elif which == "browser":
        browser_server().run("stdio")
    elif which == "try-search":  # manual check: dsv41_tools.py try-search <query>
        print(run_search(" ".join(sys.argv[2:]) or "DeepSeek Harness"))
    else:
        sys.exit("usage: dsv41_tools.py search|browser|try-search <query>")
