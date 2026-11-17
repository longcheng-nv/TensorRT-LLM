"""Embed the decode-timeline viewer (a complete page made by embed.py) into the full report as one section.

The viewer page is stored base64-encoded in a non-executing <script type="text/plain"> block and loaded into
a same-origin srcdoc iframe at runtime, so the report stays a single self-contained file and the viewer's CSS /
JS cannot clash with the report's. The iframe grows to the viewer's height. Idempotent: the section between
<!-- TIMELINE:BEGIN --> and <!-- TIMELINE:END --> is replaced on every run.

usage: embed_in_report.py VIEWER.html REPORT.html
"""
from __future__ import annotations

import base64
import re
import sys

EMBED_CSS = ('<style id="embedded-in-report">header.top,#intro,footer.foot{display:none!important}'
             'main.wrap,.wrap{max-width:none!important}'  # the report widens the frame to the browser width ...
             '#summary,#howto{width:100%;max-width:960px;margin-left:auto;margin-right:auto}'  # ... text stays in the column
             'main.wrap{padding-top:0!important}h2 .num{display:none}'
             'body{padding-left:0!important;padding-right:0!important;padding-bottom:8px!important}</style>')

SECTION = """<!-- TIMELINE:BEGIN -->
<section id="timeline" aria-labelledby="t-timeline">
  <h2 id="t-timeline"><span class="num">06</span>解码时间线（实测，类似 nsys）</h2>
  <p>真实引擎上逐 token 的事件时间线：mk-6（关掉 mk-7 的开关）与 mk-7 两版代码，默认 <code>--fast</code> 与 <code>--lossless</code> 各一组，每组预热 60 个 token 后连续记录 5 个，接电源。CPU 轨是主线程每层的各个阶段；SSD 轨来自原生 IO 层的追踪（每个 2 MB 块、所在 IO 线程、需求读 / 推测读）；GPU 轨是推断的：本机拿不到 GPU 时间戳，注意力块的结束时刻取路由同步返回的实测值，MoE、共享专家与 Engram 块的时长为估计。插桩有少量开销，且只有 5 个 token，单个 token 的时长会受当时缓存内容影响，比较两版时以第 05 节的同进程 A/B 为准。</p>
  <p class="muted" style="font-size:13px">滚轮缩放、拖动平移，选择单个 token 可放大；悬停或聚焦查看详情。同一份数据的独立页面在 <code>trace/DECODE_TRACE.html</code>；Perfetto 格式在 <code>trace/perfetto_fast.json</code>、<code>trace/perfetto_lossless.json</code>（mk-7）与 <code>trace/perfetto_mk6_fast.json</code>、<code>trace/perfetto_mk6_lossless.json</code>（可拖进 ui.perfetto.dev 或 Chrome 的 chrome://tracing）。采集脚本 <code>mac-deploy-tests/mk/trace_decode.py</code>。</p>
  <div id="tl-bleed" style="align-self:flex-start;max-width:none;width:100%">
    <iframe id="tl-frame" title="解码时间线查看器" style="display:block;width:100%;height:1500px;border:0;background:var(--bg)"></iframe>
  </div>
  <script type="text/plain" id="tl-src">__B64__</script>
  <script>
  (function () {
    "use strict";
    var f = document.getElementById("tl-frame"), src = document.getElementById("tl-src");
    var box = document.getElementById("tl-bleed"), sec = document.getElementById("timeline");
    if (!f || !src) return;
    // full-bleed: the timeline spans the browser width (minus the page gutters); the text column is untouched
    function bleed() {
      if (!box || !sec) return;
      var cs = getComputedStyle(document.body);
      var gl = parseFloat(cs.paddingLeft) || 16, gr = parseFloat(cs.paddingRight) || 16;
      var vw = document.documentElement.clientWidth;  // excludes a classic scrollbar
      var r = sec.getBoundingClientRect();
      var w = Math.max(r.width, vw - gl - gr);
      box.style.width = w + "px";
      box.style.marginLeft = Math.min(0, gl - r.left) + "px";
    }
    bleed();
    window.addEventListener("resize", bleed);
    try {
      var bin = atob(src.textContent.trim()), u8 = new Uint8Array(bin.length);
      for (var i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
      f.srcdoc = new TextDecoder("utf-8").decode(u8);
    } catch (e) {
      var p = document.createElement("p");
      p.textContent = "时间线加载失败：" + e;
      f.replaceWith(p);
      return;
    }
    function fit() {
      try {
        var d = f.contentDocument;
        if (d && d.documentElement) f.style.height = Math.max(400, d.documentElement.scrollHeight) + "px";
      } catch (e) { /* not same-origin: keep the fixed height */ }
    }
    f.addEventListener("load", function () {
      fit();
      try { new f.contentWindow.ResizeObserver(fit).observe(f.contentDocument.body); } catch (e) { setInterval(fit, 1000); }
    });
  })();
  </script>
</section>
<!-- TIMELINE:END -->"""


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    viewer, report = sys.argv[1], sys.argv[2]
    page = open(viewer, encoding="utf-8").read()
    assert "</head>" in page, "viewer page has no </head>"
    page = page.replace("</head>", EMBED_CSS + "</head>", 1)
    b64 = base64.b64encode(page.encode("utf-8")).decode("ascii")
    sec = SECTION.replace("__B64__", b64)
    rep = open(report, encoding="utf-8").read()
    if "<!-- TIMELINE:BEGIN -->" in rep:
        rep = re.sub(r"<!-- TIMELINE:BEGIN -->.*?<!-- TIMELINE:END -->", lambda m: sec, rep, count=1, flags=re.S)
    else:
        anchor = '<!-- ============================================================ 限制与后续 -->'
        assert anchor in rep, "insertion anchor not found in the report"
        rep = rep.replace(anchor, sec + "\n\n" + anchor, 1)
    open(report, "w", encoding="utf-8").write(rep)
    print(f"embedded {len(page) // 1024} KB viewer ({len(b64) // 1024} KB base64) into {report}")


if __name__ == "__main__":
    main()
