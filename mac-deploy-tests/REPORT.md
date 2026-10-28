# DeepSeek-V4.1-Flash 本地部署与验证报告（MacBook Pro M4 Pro, 48 GB）

日期：2026-10-28 ｜ 引擎：haoruilee/DeepInfer-Mac（a4df33c + 2 个本地补丁）｜ 模型：deepseek-ai/DeepSeek-V4.1-Flash-1017 @ 1bae0e1

## 结论
部署功能正确，测试闭环已完成：全部正确性门禁通过，API 端到端 28/28 通过。可用性受限于速度（默认 2.4 tok/s，开 `--moe-topk 4` 为 3.35 tok/s），属于「可用的低吞吐本地后端」。

必须明示的三点：
1. 官方 DeepInfer 引擎（`/Users/loncheng/Models/DeepInfer`）仅支持 NVIDIA SM100 / Ascend 950，macOS 不可执行；本部署用它的纯 Python 部分做预言机，推理引擎是其社区 MLX 移植。
2. 原始 510 GB FP8 权重未保留，转换器流式读取后写成 338 GB 本地格式（2/3/4-bit 专家、4-bit Engram、位精确 mxfp8 dense）。
3. 部署是有损量化：layer 0/1 的 3/3/3-bit 专家相对原始 FP4 的隐藏态余弦 0.989；质量数字是 sanity band，不等于 API 精确对齐。

## 环境事实
- 48 GB 统一内存；模型任何量化都不能全驻内存，唯一路径是 SSD 专家流式加载（dense 8 GB 常驻，213 GB 专家 + 114 GB Engram 在 SSD）。
- 其他应用占约 28 GB，本次验证专家缓存取 9 GiB；关掉大应用可提到 20–24 GiB，预计 3–4 tok/s。
- 网络：办公 Wi-Fi 直连 HF 20–79 MB/s，shell 代理 9 MB/s，iPhone 热点 0.4 MB/s；合盖+电池会睡眠。

## 门禁结果
| 门禁 | 结果 | 证明什么 |
|---|---|---|
| preflight_integrity | 48 分片 96,085 张量偏移全部一致，sha 固定 | 转换读取的字节与官方权重一一对应 |
| DeepInfer-Mac 单测 / pyflakes | 33/33，干净 | MLX 实现 == 官方 PyTorch 参考（tiny 模型） |
| 官方 encoding 测试 | 55/55 | 提示词格式语义 |
| 官方预言机 H1–H4 | PASS（27/81/11/23 项） | tokenizer 策略、配置、权重覆盖零遗漏、渲染字节级一致 |
| 量化布局数值门禁 | ALL PASS（cos ≥ 0.999996） | 2/3/4-bit slab 路径正确 |
| check_engram | OK | 114 GB Engram 表偏移/布局正确 |
| real_check 6 层真实权重 | worst cosine 0.99997 | 真实权重上 MLX == 官方参考 |
| quant_damage（layer 0/1） | 0.98887 / 0.98904 | 量化损伤量级，与 README 0.989–0.995 一致 |
| decode_speed / bench | 2.40 tok/s，prefill 21–24 tok/s | 实测吞吐 |
| CLI 贪心探针 | 391 / 北京 | 功能正确 |
| 官方结构化输出 3 例（chat + thinking） | 6/6 | xgrammar、EOS、</think> 触发 |
| 官方离线提示词 5 条（含 1/3/5 图） | 5/5，decode_ppl 1.04–1.68 | 文本与视觉路径 |
| API 端到端 pytest | 28/28 | 契约、流式、确定性+logprobs、seed、多轮前缀复用、工具调用、JSON schema、thinking、6K/12K 长上下文一致性、视觉、/v1/messages、/v1/responses |

## 发现并修复的问题
- **服务端多轮 500**（真实 bug）：MLX 流对象绑定线程，服务端每请求新线程无法使用上一轮的 KV 快照。补丁：`deepinfer_mac/server.py` 把推理固定到单一常驻线程。建议向上游提 issue/PR。
- 转换器默认 ModelScope 优先且无 revision 固定：补丁 `convert/hf.py`（HF 优先 + sha 固定）。
- `real_check.py` / `quant_damage.py` 在 N=1 时崩溃（Engram 层列表为空）；N≥2 可用。
- `quant_damage.py 6` 触发 HF 匿名限速卡死；N=2 可完成。
- 我方测试脚本的两处误报（thinking token 上限、英文答案检查）已修正。

## 运行时开关实测（cache 9 GiB，256 token prompt / 32 token 生成，第二轮）
| 设置 | 解码 tok/s | 命中率 | 每 32 token 读取 | 探针 |
|---|---|---|---|---|
| 基线 | 2.42 | 0.44 | 126 GB | 正确 |
| `--moe-topk 4` | **3.35** | 0.48 | 90 GB | 正确 |
| `--skip-miss 0.08` | 2.40 | 0.45 | 126 GB | 正确 |
| `--prefill-topk 4` | 2.50 | 0.46 | 119 GB | 正确 |
| topk4 + skip-miss | 3.31 | 0.49 | 90 GB | 正确 |

建议：需要速度时用 `--moe-topk 4`（有损，每 token 少算 2 个专家，+38%）；`--skip-miss` 在本机无增益。更激进的方案（2/2/2 全局重转、REAP 专家裁剪、去 Engram）见对话记录，暂不作为基线。

## 如何使用
```bash
# 服务（当前已在运行）
cd /Users/loncheng/Models/DeepInfer-Mac && .venv/bin/python -m deepinfer_mac serve --model /Users/loncheng/Models/dsv41-mac --cache-gb 9 --port 8000   # 可加 --moe-topk 4
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-flash","messages":[{"role":"user","content":"你好"}],"max_tokens":64,"thinking":{"type":"disabled"}}'
# 复跑验证
CACHE_GB=9 /Users/loncheng/Models/dsv41-mac-tests/verify_deployment.sh
```
日志：`/Users/loncheng/Models/logs/verify/`（summary.txt、summary-followup.txt、各步 .log），开关对比：`/Users/loncheng/Models/logs/tune/results.tsv`。
