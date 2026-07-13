# op26 iter7 — core 低 BS 每行延迟微优化(leader 尾段攻坚)

2026-07-13 · umbriel-b200-092(8×B200 健康,新机预检 smoke×2 绿 + gate 582/582)
· branch omni/op21-gvr-prod · 接 iter6c 判决的"② core 低 BS 每行延迟微优化预研"

## 0 · 预研判决(ncu full-set,3 负格代表 + radix 对照,GPU 锁基频 1.15GHz)

工具:`prof_lowbs_cell.py`(复用 sweep_op22rr.build_call,与 nsys 战役同约定)
→ `analyze_iter7_ncu.py`(SOL/stall 汇总)→ `analyze_iter7_segments.py`
(warp-stall 采样按执行过的 UCGABAR_WAIT 站点分段 ≈ 墙钟相位分解)。
报告根 = `results_iter7_prof/`(不入库)。

| cell (op26_r0mc) | dur(ncu) | barrier-stall | UCGABAR 占采样 |
|---|---|---|---|
| K1024 fp32 N131072 BS1 | 37.9µs | 54% | 40% |
| K2048 fp16 N131072 BS1 | 43.7µs | 57% | 42% |
| K1024 fp32 N65536 BS8 | 38.3µs | 58% | 46% |
| radix 同格 (对照) | 35.0µs | 58% | —(无单一热点) |

分段分解(三格一致,快路恰 4 次 cluster 同步):

| 段 | 内容 | 墙钟占比 |
|---|---|---|
| seg0 | P1 gather + P1b hist/rung + R0 M-ary 扫描 + cluster 归并 | 24-27% |
| seg1 | R0 判决 + R1 inline shot 区 | 0-3.5% |
| seg2 | P3 各 CTA 并行 collect | 11-14% |
| **seg3** | **leader 串行尾段:3×peer DSMEM 拉取 + 单 CTA vendored snap P4;其余 3 CTA 在最终 UCGABAR 干等(占 seg3 采样 70-73%)** | **57-61%** |

**结论:低 BS 缺口(fin 负格带 1.20-1.34×)的主体不是扫描带宽(DRAM
0.25-0.5%)、不是 R0 趟数,是 cluster 港 vendored epilogue 的 leader 串行
尾段。** BS=8 与 BS=1 时长几乎相同(38.3 vs 37.9µs)再证 latency-bound,
与 fin scaling 分析吻合。iter6c 曾估计的"scan 循环 ILP/向量宽度"证伪为
非主体(seg0 仅 1/4);证伪史里 "Cluster DSM @高 BS (Opt-B) 0.36-0.45×"
是把 DSM 用于扫描相位,与本杠杆(尾段并行化)无冲突。

## 1 · iter7 设计(优先级序,均挂编译 key 消融臂)

- **D3 `p4_rs`(先做,验证过的内核)**:把 1cta 臂已上硅的 op#7 exact
  rank-scatter P4(fp32 P4 1.11-2.12×,barrier 14→7)verbatim 港进
  GvrOp26R0ClusterKernel 的 leader P4,替换 vendored histogram_snap;
  dtype 调度先照 1cta 现行(fp32 开;16-bit 对照臂裁定)。与 p1b_cache
  移植同型(iter6 尾声先例:1cta 结论在 cluster 港不必复现,必须 mc 域
  独立 A/B)。
- **D1 `push_gather`**:P3 后 peers 把自己的 (key,val) 候选经
  st.shared::cluster **主动推**进 leader smem(base offset = 低 rank 峰
  count 前缀,P3 后已有 cluster 同步点可复用),替换 leader 的 3×串行
  DSMEM 拉取循环。
- **D2 cluster-cooperative P4(备选,D3/D1 不够再做)**:hist 建于
  leader smem 由 4 CTA cluster-atomic 共建(各自消化自己的候选,gather
  全免),scatter 各 CTA 直写 gmem;代价 = P4 内 barrier 变 UCGABAR,
  数据依赖迭代(snap/精修)×cluster 同步有 iter6c 全局同步税风险,
  须单格 nsys 裁定。

判据(单格 nsys,cold gm,臂对 = gvr_multicta_cutedsl + op26_r0mc(new)
+ op26_r0mc(base) + radix_cutedsl 同批):
- 负格代表 K1024 fp32 131072 BS1-8 vs radix ≥1.0(from ~0.84);
- mc 调度域全 dtype 无 <0.98 新洞(r0mcc A/B 判据同款);
- 赢了 → 54 批 mcab 网格复判 → 默认落 dispatch → 报告回填收编
  (连同 p1bc_mc 默认的 ~1-3.4% 一起)。

## 2 · 消融积压(iter7 主线后)

kC-diet(K512@1536 省 28KB smem)/ K2048 fp32 edge-aim R1 / qfracs=UH4/M3A
对照 —— 均二阶,单格 nsys 判决制。

## 3 · gotcha 继承

- nsys/ncu 一律 `env -u GITHUB_TOKEN -u HF_TOKEN`;*.ncu-rep/*.sqlite 永不入库(.gitignore 已含)。
- ncu 锁基频(1.15GHz),绝对时长勿与 nsys boost 数字直接比,比值/占比可用。
- 计时批独占 GPU;smoke/gate 可与之并行在他卡。
- 发车禁 `cmd1 && cmd2 &`;长跑 setsid;marker 幂等。
