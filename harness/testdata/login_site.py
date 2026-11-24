"""A tiny local site with a login wall, for testing the browser tools without a person.

  /secret   the protected page ("THE-SECRET-IS 4242"); 302 to /login until logged in
  /login    a form with a password field; the page polls /status and moves on to /secret once
            logged in -- the way a real site continues after the user signs in
  /approve  test hook: marks the session logged in, as if the user had signed in in the window
  /color    a solid red page (screenshot check)

python login_site.py [port]  serves until killed; tests import start() instead.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOGIN_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>请登录 · Test Portal</title></head>
<body><h1>Test Portal 登录</h1><form method="post" action="/login">
<label>用户名 <input name="username" placeholder="用户名"></label><br>
<label>密码 <input name="password" type="password" placeholder="密码"></label><br>
<button type="submit">登录</button></form>
<script>setInterval(async () => { const r = await fetch('/status'); if ((await r.json()).logged_in) location.href = '/secret'; }, 1000);</script>
</body></html>"""
SECRET_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>内部页面 · Test Portal</title></head>
<body><h1>欢迎回来</h1><p>THE-SECRET-IS 4242</p><a href="/color">颜色页</a></body></html>"""
COLOR_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>颜色页</title></head>
<body style="margin:0;background:#e00000"><h1 style="color:#fff;font-size:120px">COLOR</h1></body></html>"""


class State:
    def __init__(self) -> None:
        self.logged_in = False
        self.paths: list[str] = []


def start(port: int = 0) -> tuple[str, State, ThreadingHTTPServer]:
    state = State()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, body: str = "", ctype: str = "text/html; charset=utf-8", headers: dict | None = None):
            data = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?")[0]
            state.paths.append(path)
            if path == "/secret":
                if state.logged_in:
                    self._send(200, SECRET_PAGE)
                else:
                    self._send(302, headers={"Location": "/login?next=/secret"})
            elif path == "/login":
                self._send(200, LOGIN_PAGE)
            elif path == "/status":
                self._send(200, json.dumps({"logged_in": state.logged_in}), "application/json")
            elif path == "/approve":
                state.logged_in = True
                self._send(200, "ok", "text/plain")
            elif path == "/color":
                self._send(200, COLOR_PAGE)
            else:
                self._send(404, "not found", "text/plain")

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            state.paths.append(self.path)
            state.logged_in = True
            self._send(302, headers={"Location": "/secret"})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}", state, server


if __name__ == "__main__":
    base, _, srv = start(int(sys.argv[1]) if len(sys.argv) > 1 else 8765)
    print(f"serving {base}/secret (login wall), {base}/approve, {base}/color")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        srv.shutdown()
