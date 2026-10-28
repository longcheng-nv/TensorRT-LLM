# HANDOFF — DeepSeek-V4.1-Flash 本地部署（M4 Pro 48 GB MacBook）

更新时间：2026-10-27 22:20（用户要求「先暂停」，所有进程已停止）
会话 ID：`60697234-ef74-4285-9fb6-da2f8011ce9b`（`claude --resume 60697234-ef74-4285-9fb6-da2f8011ce9b` 可恢复完整对话）

## 目标
下载 DeepSeek V4.1 Flash 与 DeepInfer 推理引擎，在本机 MacBook 上跑通推理，并完成测试闭环验证部署功能正确。

## 关键结论（已核实）
- 官方 `deepseek-ai/DeepInfer`（`/Users/loncheng/Models/DeepInfer`，commit 5537415）只支持 NVIDIA SM100 / 华为 Ascend 950，Linux + CUDA 13；Mac 上无法执行（platform 枚举只有 GPU/NPU，`_C` 扩展 CUDA-only）。本项目仅把它当预言机代码用（tokenizer 策略、配置解析、权重名映射、eval/score.py）。
- 实际推理引擎：社区移植 `haoruilee/DeepInfer-Mac`（`/Users/loncheng/Models/DeepInfer-Mac`，commit a4df33c），MLX + SSD 专家流式加载，2/3/4-bit 有损专家量化 + 4-bit Engram + 位精确 mxfp8 dense。
- 官方权重 `deepseek-ai/DeepSeek-V4.1-Flash-1017` 510 GB FP8 不落盘，由转换器范围读取后直接写成约 338 GiB 的本地格式到 `/Users/loncheng/Models/dsv41-mac`。
- 本地补丁：`DeepInfer-Mac/deepinfer_mac/convert/hf.py` 把 ENDPOINTS 改为两个 Hugging Face（ModelScope 的 223.5.5.5 DNS 在本网络超时），并把 revision 固定为 `1bae0e1088993b9b82de08f59de5be5a34e42170`（`git diff` 可见，勿还原）。
- 网络：shell 里有 privoxy 代理 `127.0.0.1:8118`，走代理只有约 9 MB/s；所有下载进程必须 `env -u HTTP_PROXY -u HTTPS_PROXY` 直连（办公 Wi-Fi 20–35 MB/s）。iPhone 热点（172.20.10.x）只有 0.4 MB/s，禁止在热点上跑。
- 电源：合盖 + 电池会睡眠，`caffeinate` 挡不住；转换期间必须接电源、开盖。
- 内存：用户其他应用占约 28 GB，推理时 `--cache-gb` 取 6–16，由 `after_convert.sh` 按当时可用内存自动计算。

## 当前进度（2026-10-28 更新）
| 项目 | 状态 |
|---|---|
| 转换 | **完成** 2748/2748（10-28 10:39），`doctor` = conversion complete / ready to run，模型 338 GB，磁盘剩余 166 GiB |
| 首轮验证 `verify_deployment.sh`（cache 9 GiB） | 11 PASS / 4 FAIL；4 个 FAIL 均已定性并处理（见下） |
| 实测速度 | 解码 2.4 tok/s（命中率 0.45–0.55，1.57 GB/token，SSD 等待 248 ms/token）；prefill 21–24 tok/s；14 token 首字 7.8 s |
| 服务 | `python -m deepinfer_mac serve --cache-gb 9 --port 8000` 常驻 |
| 补充验证 `followup.sh` | 全部 PASS：API 28/28、离线 5/5、结构化 6/6（含 thinking）、quant_damage N=2 0.989、开关对比见 `logs/tune/results.tsv` |

**后续分析（10-28）**：`PRUNING_ANALYSIS.md`（裁剪不可能达 20×：带宽下限 40 ms/token、99.5% 路由质量需 220/384 专家）；`ROOFLINE_OPT.md`（无损配置优化：缓存 18 GiB + SPEC_K 3，ISL=1024 解码 3.15 tok/s = roofline 68%，prefill ≈100%；服务现以 `--cache-gb 18 --spec-k 3` 运行）。

**最终报告：`dsv41-mac-tests/REPORT.md`。任务已完成；后续可选：向上游提 server.py 线程补丁 PR、尝试更大缓存或 `--moe-topk 4`。**

首轮 4 个 FAIL 的定性：
- `quantdamage`（6 层）：HF 匿名限速导致卡死 24 min，被终止；非正确性门禁，`followup.sh` 以 N=2 重试。
- `structured`：官方 3 例（chat）全过；我加的 thinking 扩展中 regex/EBNF 两例因 512 token 内推理未结束而未触发语法，`followup.sh` 用 3000 token 复跑。
- `offline`：5/5 语义正确（第 5 条答「胡萝卜」），检查器只认英文已修正。
- `api`：24/28；4 个失败中 2 个是真实 bug —— 多轮第二轮/工具调用第二轮 500：MLX 流对象线程绑定，服务端每请求新线程无法使用上一轮快照 → **已在 `deepinfer_mac/server.py` 打补丁：所有推理固定到单一常驻工作线程**（`git diff` 可见）；其余 2 个是连带的连接重置和我的 Responses 测试 token 上限太小。补丁后 4 例全部通过。

## 已通过的门禁（转换前）
1. `preflight_integrity.py`：48 分片 header 与 `tensor_index.json` 96,085 张量偏移全部一致，仓库 sha 与固定 revision 一致。
2. DeepInfer-Mac 无权重单测 33/33（含官方 PyTorch 参考的 tiny 模型 parity）、pyflakes 干净；官方 `encoding/test_encoding.py` 55/55。
3. `gate_quant_layouts.py`：生产布局 3/3/3、2/2/3、4/4/4、DSpark 在 slab/gather_qmm 路径 vs 去量化参考 cos ≥ 0.999996，slab 字节与文件一致。
4. `gate_official_oracles.py`：H1 tokenizer 策略 27/27、H2 配置字段 81/81、H3 权重覆盖 11/11（零遗漏）、H4 提示词渲染 23/23（fixture 3/4 为已记录的 API 契约差异）。
5. 阶段门真实权重：`scripts/real_check.py 5 --noquant` worst layer cosine 0.99998，Engram 哈希 id 一致；断点恢复测试通过（kill 后重启正确跳过已完成任务）。
6. `scripts/quant_damage.py 5` 未完成（与转换器争带宽被我停掉），转换完成后在 `verify_deployment.sh` 里重跑。

## 继续的步骤
1. 接电源、连正常 Wi-Fi、开盖。
2. 自动模式（推荐）：
   ```bash
   nohup /Users/loncheng/Models/dsv41-mac-tests/resume_when_ready.sh > /Users/loncheng/Models/logs/resume-guardian.out 2>&1 &
   ```
   守护脚本每 2 分钟检测「AC 供电 + 非热点 + 直连 HF ≥ 5 MB/s」，满足即 `run_full_convert.sh` 续跑；转换完成后 `after_convert.sh` 自动算 `CACHE_GB` 并运行 `verify_deployment.sh`（KEEP_SERVER=1，验证后保留 `127.0.0.1:8000` 服务）。完成标记：`logs/verify/verify-launched.marker`，结果：`logs/verify/summary.txt`。
3. 手动模式：`run_full_convert.sh` 续跑；`convert_progress.sh` 看进度；完成后 `CACHE_GB=10 verify_deployment.sh`。
4. 验证完成后：读 `logs/verify/summary.txt` 与各步 log，用一个 workflow（多视角审查 agent）对照日志核实每条结论，再写最终报告（建议发布为 artifact）。报告须明示：官方 DeepInfer 不能在 macOS 执行、原始 510 GB 权重未保留、部署为有损量化（质量数字是 sanity band 而非 API 精确对齐）。

## 验证套件（`verify_deployment.sh`，按序）
doctor → weightfree(33) → encoding(55) → oracles → quantgate → engram(`check_engram.py`) → realcheck(`real_check.py 6 --noquant`，要求 cos ≥ 0.9999) → quantdamage(`quant_damage.py 6`) → speed(`decode_speed.py` 实测 tok/s) → bench → cli(17*23→391) → cli_zh(北京) → structured(`structured_output_check.py --thinking`：官方 regex/JSON/EBNF 三例) → offline(`offline_prompts_check.py`：官方 5 条离线提示词含图片，decode_ppl) → api(启动 serve，`pytest test_api_e2e.py`：/v1/models、/stats、流式、贪心确定性+logprobs、seed 复现、max_tokens、多轮+前缀复用、错误码、精确答案、代码可执行、JSON schema（含 thinking 触发）、thinking、工具调用往返、6K/12K 长上下文 needle+前缀复用一致性、视觉官方样例、回环 URL 拒绝、/v1/messages、/v1/responses）。

## API 契约要点（deepseek-recipe）
模型名 `deepseek-flash`；默认开启 thinking 且强制 temperature 1.0、未给 max_tokens 时默认 65536 → 确定性测试必须带 `"thinking":{"type":"disabled"}` 和 `max_tokens`；`reasoning_effort` 用字符串 low/high/max；`response_format` json_object / json_schema 映射到 xgrammar；一次只处理一个请求（全局锁）。

## 文件索引
- 引擎与模型：`/Users/loncheng/Models/DeepInfer-Mac`（venv `.venv`，py3.12，mlx 0.32.4）、`/Users/loncheng/Models/dsv41-mac`
- 官方参考：`/Users/loncheng/Models/DeepSeek-V4.1-Flash-1017-ref`、`…-Flash-ref`；`~/dl/dsv41-ref/inference` 软链接供脚本硬编码路径使用
- 测试与脚本：`/Users/loncheng/Models/dsv41-mac-tests/`（本文件所在）
- 日志：`/Users/loncheng/Models/logs/`
- 审计原始输出：`/private/tmp/claude-502/-Users-loncheng-Models/60697234-ef74-4285-9fb6-da2f8011ce9b/scratchpad/audit_result.json`（临时目录，可能被清理）
- 备选路径（仅在主路径放弃且删除 dsv41-mac 后）：llama.cpp master（deepseek41，PR #31086）+ unsloth UD-IQ1_S 233 GB，与主路径磁盘互斥。
