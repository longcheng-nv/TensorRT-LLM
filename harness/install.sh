#!/bin/bash
# One-command deployment of DeepSeek-V4.1-Flash (DeepInfer-Mac engine) + DeepSeek Harness (`dsv41`)
# on an Apple Silicon Mac:
#
#   curl -fsSL https://raw.githubusercontent.com/longcheng-nv/TensorRT-LLM/deepinfer-mac-dsv41-flash/harness/install.sh | bash
#   # or, from a checkout:  bash DeepInfer-Mac/harness/install.sh
#
# Every step is idempotent: re-run the same command after an interruption and it continues
# (the model conversion resumes where it stopped). Layout under $DSV41_ROOT (default ~/Models):
#   DeepInfer-Mac/   engine + harness code (branch deepinfer-mac-dsv41-flash)   .venv/ inside
#   dsv41-mac/       converted model (~338 GB, streamed + converted from the official FP8 weights)
#   dsh/venv/        official DeepSeek Harness SDK (bundled macOS runtime, no Node.js)
#   dsh/home/        harness sessions        ~/.local/bin/dsv41 -> DeepInfer-Mac/harness/dsv41
#   dsh/tools-venv/  web search + login-capable browser tools (MCP; uses the installed Google Chrome)
#
# Environment:
#   DSV41_ROOT=~/Models        install root            DSV41_WORKERS=6     converter workers
#   DSV41_SKIP_MODEL=1         code + environments only (no 338 GB model)
#   DSV41_SKIP_TEST=1          no acceptance test at the end
#   DSV41_USE_PROXY=1          keep HTTP(S)_PROXY for the model download (default: direct, the
#                              proxy here throttles to ~9 MB/s)
#   DSV41_ALLOW_BATTERY=1      convert on battery (closing the lid then stops it)
#   DSV41_REPO / DSV41_BRANCH  code source (default the GitHub fork / deepinfer-mac-dsv41-flash)
#   DSV41_NO_LINK=1            do not create ~/.local/bin/dsv41 or touch ~/.zshrc
set -euo pipefail

main() {
ROOT="${DSV41_ROOT:-$HOME/Models}"
REPO="${DSV41_REPO:-https://github.com/longcheng-nv/TensorRT-LLM.git}"
BRANCH="${DSV41_BRANCH:-deepinfer-mac-dsv41-flash}"
ENGINE="$ROOT/DeepInfer-Mac"
MODEL="$ROOT/dsv41-mac"
DSH_ROOT="$ROOT/dsh"
LOGS="$ROOT/logs"
DSH_SDK="deepseek-harness-sdk==0.2.2rc2"

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()   { printf '    \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '    \033[33m!\033[0m %s\n' "$*" >&2; }
die()  { printf '\n\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }
direct() {   # run a download without the shell proxy (unless DSV41_USE_PROXY=1)
  if [ -n "${DSV41_USE_PROXY:-}" ]; then "$@"; else env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u ALL_PROXY -u all_proxy "$@"; fi
}

# ------------------------------------------------------------------------------ 1. preflight
step "1/7 检查这台 Mac"
[ "$(uname -s)" = Darwin ] || die "只支持 macOS"
[ "$(uname -m)" = arm64 ] || die "需要 Apple Silicon（arm64），当前 $(uname -m)"
macos="$(sw_vers -productVersion)"; [ "${macos%%.*}" -ge 14 ] || die "需要 macOS 14 或更新（当前 ${macos}）"
mem_gb=$(( $(sysctl -n hw.memsize) / 1073741824 ))
[ "$mem_gb" -ge 32 ] || die "内存 ${mem_gb} GB 太小（至少 32 GB；本方案在 48 GB 上验证）"
[ "$mem_gb" -ge 48 ] || warn "内存 ${mem_gb} GB：能跑，但专家缓存会很小、解码更慢（验证机为 48 GB）"
ok "macOS ${macos} · $(sysctl -n machdep.cpu.brand_string) · ${mem_gb} GB"
if ! xcode-select -p >/dev/null 2>&1; then
  xcode-select --install >/dev/null 2>&1 || true
  die "需要 Xcode 命令行工具（git、clang；引擎会在首次运行时编译一个小的 C 模块）。已弹出安装窗口，装完再运行本命令。"
fi
ok "Xcode 命令行工具：$(xcode-select -p)"
mkdir -p "$ROOT" "$LOGS"

# ------------------------------------------------------------------------------ 2. uv
step "2/7 Python 工具链（uv）"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    brew install uv </dev/null
  else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  fi
fi
command -v uv >/dev/null 2>&1 || die "uv 安装失败"
ok "$(uv --version)"

# ------------------------------------------------------------------------------ 3. code
step "3/7 引擎与 harness 代码 → ${ENGINE}"
if [ -d "$ENGINE/.git" ]; then
  ok "已存在（$(git -C "$ENGINE" rev-parse --abbrev-ref HEAD) @ $(git -C "$ENGINE" log -1 --format='%h %cd' --date=short)），保持不动"
  [ -f "$ENGINE/harness/dsv41" ] || die "$ENGINE 里没有 harness/（不是 ${BRANCH} 分支？）"
else
  direct git clone --branch "$BRANCH" --single-branch "$REPO" "$ENGINE" </dev/null
  ok "已克隆 ${BRANCH}"
fi
HARNESS="$ENGINE/harness"

# ------------------------------------------------------------------------------ 4. engine env
step "4/7 引擎 Python 环境 → ${ENGINE}/.venv"
if "$ENGINE/.venv/bin/python" -c "import mlx.core, deepseek_recipe, xgrammar, fastapi, deepinfer_mac" >/dev/null 2>&1; then
  ok "已就绪（mlx $("$ENGINE/.venv/bin/python" -c 'import mlx.core as m; print(m.__version__)')）"
else
  [ -x "$ENGINE/.venv/bin/python" ] || (cd "$ENGINE" && direct uv venv --python 3.12 .venv)
  (cd "$ENGINE" && direct uv pip install --python .venv/bin/python -e . -c "$HARNESS/constraints-engine.txt")
  "$ENGINE/.venv/bin/python" -c "import mlx.core, deepseek_recipe, xgrammar, fastapi, deepinfer_mac" || die "引擎环境导入失败"
  ok "已安装（版本按 harness/constraints-engine.txt 锁定）"
fi

# ------------------------------------------------------------------------------ 5. harness env + command
step "5/7 DeepSeek Harness（${DSH_SDK}）→ ${DSH_ROOT}/venv"
mkdir -p "$DSH_ROOT/home"
dshv() { DSH_HOME="$DSH_ROOT/home" "$DSH_ROOT/venv/bin/dsh" --version 2>/dev/null; }   # needs an explicit home
if dshv >/dev/null; then
  ok "已就绪（dsh $(dshv)）"
else
  [ -x "$DSH_ROOT/venv/bin/python" ] || direct uv venv --python 3.12 "$DSH_ROOT/venv"
  direct uv pip install --python "$DSH_ROOT/venv/bin/python" --prerelease=allow "$DSH_SDK" -c "$HARNESS/constraints-dsh.txt"
  dshv >/dev/null || die "DeepSeek Harness 安装失败"
  ok "已安装（dsh $(dshv)）"
fi
# web search + login-capable browser for the agent (MCP servers in harness/dsv41_tools.py)
tools_ok() { "$DSH_ROOT/tools-venv/bin/python" -c "import mcp.server.mcpserver, playwright.async_api" >/dev/null 2>&1; }
if tools_ok; then
  ok "联网工具环境已就绪（${DSH_ROOT}/tools-venv）"
else
  [ -x "$DSH_ROOT/tools-venv/bin/python" ] || direct uv venv --python 3.12 "$DSH_ROOT/tools-venv"
  direct uv pip install --python "$DSH_ROOT/tools-venv/bin/python" mcp playwright -c "$HARNESS/constraints-tools.txt"
  tools_ok || die "联网工具环境安装失败"
  ok "联网工具环境已安装（搜索 + 浏览器，版本按 constraints-tools.txt 锁定）"
fi
[ -d "/Applications/Google Chrome.app" ] || warn "没有找到 Google Chrome：搜索和 web_fetch 照常可用，浏览器（登录）工具需要先安装 Chrome"
if [ -z "${DSV41_NO_LINK:-}" ]; then
  mkdir -p "$HOME/.local/bin"
  ln -sfn "$HARNESS/dsv41" "$HOME/.local/bin/dsv41"
  if ! grep -qs 'HOME/.local/bin' "$HOME/.zshrc"; then
    printf '\n# dsv41 (DeepSeek-V4.1-Flash + DeepSeek Harness)\nexport PATH="$HOME/.local/bin:$PATH"\n' >> "$HOME/.zshrc"
    warn "已把 ~/.local/bin 加进 ~/.zshrc（新开终端生效）"
  fi
  ok "命令：~/.local/bin/dsv41 → ${HARNESS}/dsv41"
fi

# ------------------------------------------------------------------------------ 6. model
step "6/7 模型（DeepSeek-V4.1-Flash → ${MODEL}）"
doctor() { (cd "$ENGINE" && .venv/bin/python -m deepinfer_mac doctor --model "$MODEL" 2>&1) || true; }
if doctor | grep -q "conversion: complete"; then
  ok "转换已完成"
elif [ -n "${DSV41_SKIP_MODEL:-}" ]; then
  warn "跳过模型（DSV41_SKIP_MODEL=1）；之后去掉该变量再运行本命令即可续上"
else
  if [ ! -d "$MODEL" ]; then
    free_gb=$(df -g "$ROOT" | awk 'NR==2 {print $4}')
    [ "$free_gb" -ge 360 ] || die "${ROOT} 所在磁盘只剩 ${free_gb} GB，模型需要约 340 GB（建议留 360 GB）"
  fi
  if pmset -g batt | grep -q "Battery Power" && [ -z "${DSV41_ALLOW_BATTERY:-}" ]; then
    die "正在用电池：转换要几个小时，合盖或睡眠会中断。请接上电源并保持开盖（或 DSV41_ALLOW_BATTERY=1）"
  fi
  case "$(ipconfig getifaddr en0 2>/dev/null || true)" in
    172.20.10.*) die "当前连的是 iPhone 热点（约 0.4 MB/s），要下载约 510 GB 的源权重，请换正常网络" ;;
  esac
  if pgrep -f "deepinfer_mac.convert.remote" >/dev/null; then
    die "已有一个转换进程在跑（pid $(pgrep -f deepinfer_mac.convert.remote | head -1)），等它结束或先停掉"
  fi
  warn "开始/继续转换：从 Hugging Face 流式读取官方 FP8 权重（约 510 GB，不落盘），写出约 338 GB 的本地格式。"
  warn "按 20–35 MB/s 的网络约需 5–8 小时；可随时 Ctrl+C，重新运行本命令会断点续传。日志：${LOGS}/convert.log"
  if ! (cd "$ENGINE" && direct caffeinate -dims .venv/bin/python -u -m deepinfer_mac.convert.remote --out "$MODEL" \
          --workers "${DSV41_WORKERS:-6}" </dev/null 2>&1 | tee -a "$LOGS/convert.log"); then
    warn "转换进程退出（中断或出错）"
  fi
  doctor | grep -q "conversion: complete" || die "转换还没完成（见 ${LOGS}/convert.log），重新运行本命令即可续上"
  ok "转换完成"
fi

# ------------------------------------------------------------------------------ 7. verify
step "7/7 验收"
if [ -n "${DSV41_SKIP_TEST:-}" ] || ! doctor | grep -q "conversion: complete"; then
  warn "跳过验收（模型未就绪或 DSV41_SKIP_TEST=1）"
else
  "$HARNESS/dsv41" test --quick </dev/null || die "快速验收没有全部通过（结果在 ${LOGS}/dsh/）"
fi

cat <<EOF

$(printf '\033[1;32m')部署完成。$(printf '\033[0m')在任意目录打开终端：
  dsv41                 终端 agent（当前目录；需要时自动在后台启动模型服务）
  dsv41 -p "问题"        问一句就退出
  dsv41 web             浏览器界面（默认 Minimal agent，可拖图片）
  dsv41 status | stop   查看 / 停止模型服务
  dsv41 test            完整验收（约 8 分钟）
说明：${HARNESS}/README.md
EOF
}

main "$@"
