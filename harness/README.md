# dsv41：在 Mac 终端里用 DeepSeek Harness 驱动本地 DeepSeek-V4.1-Flash

一条命令：`dsv41`。它会先确认本地模型服务在跑（没跑就在后台启动），然后在**当前目录**打开一个终端 agent。这个 agent 是 DeepSeek 官方的 DeepSeek Harness（`dsh`），用的是 `sdk-minimal` 配置，也就是模型卡里 "DSH Minimal" 那套脚手架：一句话人设，加持久 bash、`read_image`、`working_directory` 三个工具。

```
dsv41                     终端 agent（当前目录）；需要时自动启动模型服务
dsv41 -p "问题"            只问一句就退出（可重复 -p；--image 图片 附图）
dsv41 --resume            继续本目录最近一次会话
dsv41 web                 浏览器界面（默认 Minimal agent，可拖图片）；会自动打开浏览器，
                          地址是终端里打印的 http://127.0.0.1:3080/?token=…（需要带 token 访问）
dsv41 serve [--lossless] [--cache-gb N]   只启动模型服务（后台）
dsv41 stop | restart | status | log
dsv41 test [--quick]      整套验收测试，结果写到 ~/Models/logs/dsh/
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
- **截图提问。** 用 `Ctrl+Shift+Cmd+4` 截图到剪贴板，再输入 `/paste 这个报错是什么意思？`。
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

| 场景 | 实测（2026-11-24，`dsv41 test`） |
|---|---|
| 新会话第一轮（约 710 token 提示） | 首字约 25 s：引擎只复用完全相同的前缀，每个新会话都要从头算一遍 system 提示和工具定义 |
| 同一会话后续轮次 | 提示里 95% 以上命中前缀缓存，首字 4–10 s |
| 生成 | 约 8–10 tok/s |
| 一次带 bash 的问答（出命令 → 执行 → 读结果再回答） | 约 25–45 s |
| 带图片的请求 | 不复用前缀缓存（引擎目前的限制），会话里有图之后每一步首字约 28 s。看完图建议 `/new` 开新会话 |

`--lossless` 模式输出确定、可复现，但只有约 4.5 tok/s。服务端模式在 `dsv41 serve --lossless` / `dsv41 restart --fast` 时切换。

## 配置了什么

| 位置 | 作用 |
|---|---|
| `~/Models/dsh/venv` | 官方 `deepseek-harness-sdk==0.2.2rc2`（自带 macOS arm64 运行时与 `dsh` 命令，不需要 Node.js） |
| `~/Models/dsh/home` | 终端 agent 的 `DSH_HOME`：会话（`sessions/<目录>/session-*/`）、附件、profile |
| `~/Models/dsh/home-web` | `dsv41 web` 的 `DSH_HOME`。必须和终端分开：Web profile 用 zstd 存会话，遇到终端的 `.jsonl` 会话目录会启动失败 |
| `~/Models/dsh/run` | 模型服务的 pid、模式、日志（`server.log`） |
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

- **单请求、单缓存槽。** 服务端一次只处理一个请求，只保留一份前缀缓存。同时开两个 `dsv41`，或者终端和 Web 一起用，会互相排队并挤掉对方的缓存。
- **带图对话不复用缓存。** 见上面的速度表。
- **不能取消正在 prefill 的请求。** `Ctrl+C` 只能断开 harness，服务端要等这一步算完。
- **`working_directory` 工具。** 它能把 agent 的当前目录切到工作区外，但写入仍然被沙箱拦在启动目录里（已测）。
