# op26 战役花费记录 (2026-07-09 ~ 2026-07-10)

优化 + 验证全周期的资源花费档案。汇总:**GPU ≈ 15 GPU-h,Claude token ≈ $108(确定归属口径)**。

## 1 · GPU 花费(nsys 战役 + 门禁)

| 节点 | 内容 | GPU-h |
|---|---|---|
| b200-038/036 | iter0-2 实现/smoke + 291×2 门禁 + op26a 首 4 批 | ~1.0 |
| b200-027 | op26a 60 批(GPU0-2 × 2h)+ op26b 全 81 批(GPU3-5 × 0.5h) | ~7.5 |
| b200-069 | worst 17 批补齐三轮:r0 三卡中途重切(0.4)+ r1 错臂 5 臂返工(8 卡 × 0.5h = 4.0)+ r2 正确臂对(8 卡 × 9min = 1.2) | ~5.6 |
| **合计** | | **≈ 15** |

其中错臂返工浪费 ~4 GPU-h(教训:单分片重发 env 必须整段照抄 TAKEOVER Step 1,发车后立即 grep `arms=` 头行核验)。

## 2 · Claude token 花费

方法:解析本机 session transcript(`~/.claude/projects/.../\*.jsonl`)中 API 返回的
usage 记录逐条累加;定价 = Fable 5 官方牌价 $10 input / $50 output /
$12.5 cache-write(实测全部 5-min TTL)/ $1.0 cache-read,单位每 MTok。

### 确定归属 op26 的 session

| Session | 日期/节点 | 内容 | input | output | cache-wr | cache-rd | 花费 |
|---|---|---|---|---|---|---|---|
| f72976b5 | 07-09 038/036 | 设计+实现+门禁 iter0-3 | 40K | 149K | 0.38M | 10.7M | $23.22 |
| 327729b8 | 07-09 027 | 战役接管(短) | 36K | 0.6K | 0.07M | 0.10M | $1.36 |
| 10234912 | 07-09 027 | 战役监控(短) | 40K | 5K | 0.07M | 0.61M | $2.19 |
| 840e34e8 | 07-09 027 | 027 主战役 + parse pass1 | 58K | 101K | 0.56M | 13.1M | $25.77 |
| 79dce2d2 | 07-10 069 | 缺口补齐 + 8 卡重切 + 错臂返工 + 报告 + 提交 | 45K | 223K | 1.43M | 26.5M | $55.94 |
| **合计** | | | | | | | **≈ $108** |

### 口径说明

- 07-09 00:00-06:20 另有 6 个 session 共 ~$317,与 op22rr 重测 / op25
  backfill 等并行战役混杂,无法按臂拆分,未计入;07-10 并行 session
  da924869($70)归属不明,未计入。全项目 07-09 起总额 $495.80 是上界。
- 成本结构:大头是 cache-read(长战役监控型 session 每次唤醒重读全上下文,
  10-50M tok/session 级),output 总量仅 ~0.5M tok(确定口径内)。
- GPU 卡时按 marker/日志时间戳推算;token 按 transcript 实测,非估算。

## 3 · 关联档案

- 结果与判决:`ITERATIONS.md` iter4(headline、锚漂移、QA 门)
- 报告数据:`op22_temporal_fixed_hr_bench/REPORT.html` §1-2 +
  `op22rr_op26_raw.csv`(5432 行本机原始值)
- 提交:206168c08e(战役收口)、3762c22b5c(标签+花费小结)
