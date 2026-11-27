# dsv41：在 Mac 终端里用 DeepSeek Harness 驱动本地 DeepSeek-V4.1-Flash

完整报告（架构、性能、测试、限制）：[`mac-deploy-tests/DSV41_MAC_REPORT.html`](../mac-deploy-tests/DSV41_MAC_REPORT.html)。

一条命令：`dsv41`。它会先确认本地模型服务在跑（没跑就在后台启动），然后在**当前目录**打开一个终端 agent。这个 agent 是 DeepSeek 官方的 DeepSeek Harness（`dsh`），用的是 `sdk-minimal` 配置，也就是模型卡里 "DSH Minimal" 那套脚手架：一句话人设，加持久 bash、`read_image`、`working_directory` 三个工具。

```
dsv41                     终端 agent（当前目录）；需要时自动启动模型服务
dsv41 -p "问题"            只问一句就退出（可重复 -p；--image 图片 附图）
dsv41 --resume            继续本目录最近一次会话
dsv41 web                 浏览器界面（默认 Minimal agent，可拖图片）；会自动打开浏览器，
                          地址是终端里打印的 http://127.0.0.1:3080/?token=…（需要带 token 访问）
dsv41 serve [--lossless] [--cache-gb N]   只启动模型服务（后台）
dsv41 stop | restart | status | log
dsv41 test [--quick]      整套验收测试（15 项，含搜索、web_fetch、登录流程、截图），结果写到 ~/Models/logs/dsh/
```

## 一键部署（新 Mac）

```sh
curl -fsSL https://raw.githubusercontent.com/longcheng-nv/TensorRT-LLM/deepinfer-mac-dsv41-flash/harness/install.sh | bash
```

装在 `~/Models` 下（`DSV41_ROOT` 可改），整个过程：

1. **预检：** Apple Silicon、macOS 14+、内存 ≥ 32 GB（验证机是 48 GB）、Xcode 命令行工具。
2. **装 uv。**
3. **代码：** 克隆本分支到 `DeepInfer-Mac`。
4. **引擎环境：** 版本按 `harness/constraints-engine.txt` 锁定。
5. **DeepSeek Harness 环境：** 版本按 `constraints-dsh.txt` 锁定，然后链接 `~/.local/bin/dsv41`。
6. **转换模型：** 从 Hugging Face 流式读取官方 FP8 权重（约 510 GB，不落盘），写出约 338 GB，按 20–35 MB/s 的网络约需 5–8 小时。要求接电源、不在 iPhone 热点上，默认绕过 shell 代理。
7. **验收：** `dsv41 test --quick`。

注意：
- **可以反复运行。** 已完成的步骤会跳过，模型转换断点续传，中断后再跑同一条命令即可。
- **磁盘空间。** 至少需要约 360 GB。
- **只装代码和环境。** 用 `DSV41_SKIP_MODEL=1` 跳过模型（比如先装好，晚上接电源再转模型）。
- **其他环境变量**（代理、电池、工作线程数等）见 `install.sh` 开头。

## 在终端里怎么用

- **直接输入问题。** 回答是边生成边显示的：灰字是思考，白字是回答，青色 `⚙ bash: …` 是 agent 执行的命令，`│` 后面是命令输出。
- **拖文件进来。** 从 Finder 把文件拖进 Terminal 窗口，会插入路径（空格转义成 `\ `，或者带引号），`dsv41` 都能识别：
  - 图片（png/jpg/gif/webp/heic…）会作为图片附上，模型能直接看到；
  - 其他文件只把路径整理好留在文字里，agent 会用 bash 自己去读；
  - 同一行可以拖多个文件，再接着写问题。
- **用 `@` 引用文件。** 语法沿用 DeepSeek Harness 的：`@path`，路径有空格时写 `@"path"`，相对路径从工作目录算起。引用的内容会随问题一起发给模型，省掉 agent 再读一次文件的那一步（在约 10 tok/s 下大约省 5–10 s）。
  - **补全：** 按 Tab。`@ma<Tab>` 按文件名在工作目录里查找（在 git 仓库里时只找受管文件），`@src/<Tab>` 按路径补全。
  - **文本文件：** 内容内联，超过约 48 KB 只附开头（`DSV41_MENTION_MAX_BYTES` 可调）；`@file.py:40-80` 只附这几行。
  - **目录：** `@src/` 附上两层目录树。
  - **图片：** `@shot.png` 直接给模型看。
  - **二进制文件：** 只传路径。
  - **和拖入的区别：** 拖入文件时不加 `@`，只把路径交给 agent，由它自己决定读不读、读哪部分。
- **截图提问。** 用 `Ctrl+Shift+Cmd+4` 截图到剪贴板，再输入 `/paste 这个报错是什么意思？`。
- **联网。** 直接说"搜索一下…""读一下这个网址…"就行，agent 会自己选工具：
  - **搜索：** `mcp__web__search`，查 Bing，失败时退到 DuckDuckGo，不需要 API key，也不经过 DeepSeek 云端。
  - **读网页：** `web_fetch` 是 DeepSeek Harness 自带的工具，走本机匿名 HTTP，得到的是网页正文。
  - **需要登录的网站：** `mcp__browser__*`。agent 会弹出一个**专用的 Chrome 窗口**（配置在 `~/Models/dsh/browser-profile`，和你日常用的 Chrome 分开）。遇到登录页时终端会提示 🔐，你在这个窗口里自己登录（密码和两步验证都不经过模型，往密码框输入会被拒绝），agent 等你登录完成后接着读页面、点击、填表、截图（截图模型直接能看）。登录状态保存在这个配置里，下次不用重新登录；会话结束时窗口会关闭。
  - **安全约束：** 工具说明里要求 agent 在付款、发消息、删除这类不可逆的提交之前先问你。窗口是可见的，它做了什么你都能看到。只想让它看不想让它点，用 `DSV41_BROWSER_ACTIONS=0`。
  - **关掉浏览器工具：** `dsv41 --no-browser` 或 `/browser off`，每个新会话的提示少约 740 token。
  - **已知限制：** 部分网站（比如 Google 账号）会拒绝在自动化控制的浏览器里登录。
- **它能动哪些文件。**
  - 默认只能**写**启动时所在的目录和 `/tmp`，这是 macOS 沙箱（Seatbelt）强制的；**读**不受限制。
  - 想让它改别处的文件，就到那个目录里启动 `dsv41`；确实需要不受限制时用 `dsv41 --full-access`；只想让它看不想让它改，用 `--read-only`。
- **中断。** `Ctrl+C` 会中断当前请求：harness 重启，会话保留。服务端会把正在算的那一步收尾后再接下一个请求。
- **命令：** `/help` `/new` `/sessions` `/resume [序号|id]` `/effort off|low|high|max` `/img <路径> [问题]` `/paste [问题]` `/stats` `/exit`。
- **多行输入：** 行尾加 `\` 续行；直接粘贴多行文本会合并成一条。

### 推理强度（`--effort` / `/effort`）

| 强度 | 说明 |
|---|---|
| `off` | 不思考，最快 |
| `low`（默认） | 简短思考，交互使用合适 |
| `high` / `max` | 思考更充分，适合复杂任务。但在约 10 tok/s 下，几百到上千 token 的思考就是一两分钟 |

## 速度预期（M4 Pro 48 GB，默认 `--fast` 模式，专家缓存 8 GB）

| 场景 | 实测（`dsv41 test`，2026-11-24；新会话与带图为 2026-11-27） |
|---|---|
| 新会话第一轮（约 1,500 token 提示） | 服务启动后的前两个新会话首字约 27–37 s（第二个顺带保存 system 提示和工具定义的状态）；之后的新会话直接复用这段前缀，首字 11–13 s（原来约 30 s） |
| 同一会话后续轮次 | 提示里 95% 以上命中前缀缓存，首字 4–10 s |
| 生成 | 约 8–10 tok/s |
| 一次带 bash 的问答（出命令 → 执行 → 读结果再回答） | 约 25–45 s |
| 带图片的请求 | 复用 system 提示和工具定义，首字约 20–23 s（原来约 33 s）；但不复用会话里之前的内容，会话里有图之后每一步都要从用户消息开始重算。看完图建议 `/new` 开新会话 |

`--lossless` 模式输出确定、可复现，但只有约 4.5 tok/s。服务端模式在 `dsv41 serve --lossless` / `dsv41 restart --fast` 时切换。

## 配置了什么

| 位置 | 作用 |
|---|---|
| `~/Models/dsh/venv` | 官方 `deepseek-harness-sdk==0.2.2rc2`（自带 macOS arm64 运行时与 `dsh` 命令，不需要 Node.js） |
| `~/Models/dsh/home` | 终端 agent 的 `DSH_HOME`：会话（`sessions/<目录>/session-*/`）、附件、profile |
| `~/Models/dsh/home-web` | `dsv41 web` 的 `DSH_HOME`。必须和终端分开：Web profile 用 zstd 存会话，遇到终端的 `.jsonl` 会话目录会启动失败 |
| `~/Models/dsh/run` | 模型服务的 pid、模式、日志（`server.log`） |
| `~/Models/dsh/tools-venv` | 联网工具（`dsv41_tools.py` 的 MCP 服务：搜索、浏览器；`mcp`、`playwright` 版本按 `constraints-tools.txt` 锁定） |
| `~/Models/dsh/browser-profile` | 浏览器工具的专用 Chrome 配置（登录状态保存在这里；删掉这个目录就等于全部退出登录） |
| `patches/sdk-minimal.local.yml` | 终端 agent 用的 overlay，见下 |
| `patches/web.local.yml` | `dsv41 web` 用的 overlay，见下 |
| `~/.local/bin/dsv41` | 指向本目录 `dsv41` 的软链接 |

`patches/sdk-minimal.local.yml` 在官方 `sdk-minimal` 之上改了这些：
- **本地端点。** 上下文窗口按 64K 规划，`DSV41_CONTEXT` 可改；超过就自动压缩。服务端上限是 256K，但冷 prefill 只有约 140 tok/s，64K 就要约 8 分钟。
- **输出上限。** 每步 16K token。
- **不断流。** 不设 5 分钟空闲超时，长时间 prefill 期间服务端没有输出，不应被当成断流。
- **关掉上传扩展。** 官方默认每个请求都附带整段会话日志（最多 8 MiB）和插件清单，本地服务用不到，所以关掉。
- **沙箱。** 官方 `sdk-minimal` 不加限制地执行 bash（它面向一次性容器）；这里改成 `workspace-write`。
- **审批。** 需要越权的操作会在终端里问 y/N。
- **联网。** 加上 dsh 自带的 `web_fetch`（只开抓取，走本地 HTTP 后端：dsh 内置的搜索后端都是 DeepSeek 云端 API）；用 dsh 的 MCP 客户端挂两个本地服务：`web`（搜索）和 `browser`（可登录的专用 Chrome，`DSV41_BROWSER=0` 时不挂）。
- **去掉 MCP 资源工具。** MCP 客户端会顺带加 3 个资源工具，这两个服务没有资源，所以关掉，省约 300 token。

`patches/web.local.yml` 改了这些：
- **端点和上下文。** 与上面相同的本地端点和上下文设置。
- **默认 agent。** 新会话默认用 Minimal agent；Standard、PTC 仍可在界面里选，但提示要大得多。
- **不额外请求标题。** 关掉 LLM 生成会话标题，它会多发一个请求，挤掉服务端唯一的前缀缓存。
- **遥测。** `DISABLED`。官方默认在你点评分或反馈时，会把会话前缀上传到 `dsh-otel-collector.deepseeksvc.com`。
- **上传扩展。** 同样关掉。

服务端（`deepinfer_mac/server.py`）只加了一个只读的 `GET /live` 接口，返回当前请求的阶段、prompt/缓存命中/已生成 token 数和已生成的文字。原因是官方 SDK 只在每一步结束时才给出完整消息，终端靠它做逐字显示。

## 手动安装 / 重装 harness 部分

`install.sh` 做的就是下面这些（引擎和模型已经就位时）：

```sh
cd ~/Models && mkdir -p dsh
env -u HTTP_PROXY -u HTTPS_PROXY uv venv --python 3.12 dsh/venv
env -u HTTP_PROXY -u HTTPS_PROXY uv pip install --python dsh/venv/bin/python --prerelease=allow \
    "deepseek-harness-sdk==0.2.2rc2" -c DeepInfer-Mac/harness/constraints-dsh.txt
ln -sfn ~/Models/DeepInfer-Mac/harness/dsv41 ~/.local/bin/dsv41
dsv41 test
```

DeepSeek Harness 还处在开发者预览阶段（官方说明会有不兼容变更，也没有做过安全审计），所以这里固定了版本。升级前先在临时 `DSH_HOME` 里跑一遍 `dsv41 test`。

## 已知限制

- **单请求、单会话缓存。** 服务端一次只处理一个请求，只保留一份会话缓存（另外最多保留两段共享的 system 前缀）。同时开两个 `dsv41`，或者终端和 Web 一起用，会互相排队并挤掉对方的会话缓存。
- **带图对话只复用 system 前缀。** 见上面的速度表。
- **不能取消正在 prefill 的请求。** `Ctrl+C` 只能断开 harness，服务端要等这一步算完。
- **`working_directory` 工具。** 它能把 agent 的当前目录切到工作区外，但写入仍然被沙箱拦在启动目录里（已测）。
