# op26 iter6 — R0 h-空间梯准入:HLS-op25/op27 经验统一移植回 cuteDSL GVR 家族

2026-07-12 · umbriel-b200-049(8×B200 全空闲,全部用于并行验证)· branch omni/op21-gvr-prod

## 0 · 目标(用户 2026-07-12 重申)

提高 **GVR (cuteDSL) base** 与 **GVR multi-CTA (cuteDSL, PR#15198)** 的性能与
跨数据分布泛化性,使其对 **Radix (cuteDSL)** 与 **SGLang StreamingTopK** 在
N≥8K 的全部 seq_len×BS×分布网格上**胜 10%+**(N<8K corner case 允许输)。
约束:**不做跨算子 dispatch**(禁止 dispatch 到 radix 路径);GVR 类阈值方法
本身优化到 speed-of-light。GVR 家族内部的 per-cell 配置调度不在禁止之列
(op13/op18/op20/op26 惯例)。

## 1 · Scoreboard(iter5 收口态,op22rr 2718 格,cold 同刻度)

| 臂 | vs radix_cutedsl | vs sglang (fp32 K≤1024) |
|---|---|---|
| op26_1cta | **gm 0.919**,win10 = 1020/2718 | gm 1.237,win10 = 310/630 |
| op26_mc | **gm 1.005**,win10 = 994/2718 | gm 1.320,win10 = 272/630 |

主要缺口(N≥8K):
- **N≥262K 低 BS 全线塌方**(1cta 0.27-0.45;mc 16-bit 0.49-0.65):
  radix 每行 4-32 SM vs GVR 1(或 cluster≤4)—— 带宽结构差 × 趟数差。
- **K2048 16-bit @262K**:1cta 0.45(tie 几何 + 趟数)。
- **fp32 中段 vs SGLang**:8K-262K 成片 0.75-1.03(SGLang 单趟流式;GVR ≥2.5 趟)。
- **mc @N=8192 全 dtype 0.52-0.64**(cluster 税/相位链)。

## 2 · 证伪史约束(必须尊重,来源见各 bucket)

- in-loop 串行多阈值三次证伪(Opt-F、op8 k=4、op16)——但 **op18/op19/op20 的
  M-ary R0(第一趟多阈值)+ fused collect 是上硅赢过的**(op20 tier1 rival/x
  gm 1.345 @旧 synth 轴 event 计时);它们的短板 = [pmin,pmax] 几何铺法
  (place_mode 0-4)按旧分布逐格调参,**无 hit-rate 条件化 → op22rr real/worst
  轴泛化性未知(未回填)**。
- P1 模型化 seed(self-loop v1-v3)证伪;**但 HLS w3a 不是模型 seed,是
  prev-topK 值分布的分位数 rung**,已在 29,820 真实 Pro 转移上筛出静态快路率
  base3 0.310 → **w3a 0.957**(oracle 0.998),并在 op25/op27 上硅 ship。
- smem-resident(op15/op20 iter6)、P4 内部 refine、cluster>4、
  L2-persistence 等均死;小 N≤8K 相位链地板 = 结构墙(本战役明确豁免)。
- 趟数不是硅上时延完备代理(iter5 secant2 教训)→ 所有 host 筛选结论必须
  单格 nsys 消融确认后才进全网格。

## 3 · iter6 设计 — 三个已验证事实的统一

> (1) count_ge 多阈值一趟近免费(M=2 ×1.02,M=4 ×1.25-1.40,count_ge_multi_bench);
> (2) logits CCDF 尾部近指数 → 插值必须在 log 空间(op13/op21/op26 已上硅);
> (3) Pmean seed 无分布鲁棒性 → 256-bin 直方图 + 多阈值划分(HLS P1b ≡ 精确分位数)。

**新臂 `op26_r0`(GvrOp26R0Kernel)**,单 CTA,零 vendored 编辑:

1. **P1b(新)**:P1 之后对 K 个 prev-topK gather 值建 256-bin smem 直方图
   ([pmin,pmax] 值域;K≤2048,L2 热,成本可忽略)→ 按 h-空间 qfracs
   (w3a 0.92/0.45/0.048 系,host 筛选定终值)取 M 根 rung 阈值
   = prev-topK 值分布的分位数 → **分布自适应铺法,替代几何 fracs**。
2. **R0**:op18 `block_count_ge_multi`(同内存路径,M 静态寄存器计数器,
   缓存全部 M 列 per-thread counts)一趟测 M 根 rung;
   任一 count∈[kK,kCC](经典宽窗,比 HLS ms 窗宽得多 → 静态快路率应 >0.96)
   → **P3 零重扫**(op18 缓存列)。可选 op20 fused collect(R=1)进一步并趟。
3. **miss 路径**:M 个实测 (thr,count) 点必然给出实测括号 → op26 iter5 的
   log-falsi(几何中心/edge 瞄准按 iter5d 表)+ fb_fix 语义
   (只接受 [kK,kCC]、双向扩张、耗尽落实测 undershoot 侧)。
   **R2 类假种子问题结构性消灭**。
4. **P4**:先保留 op26 rank-scatter 调度(fp32 ∪ BS≥256);op20 band-snap
   作对照消融臂。
5. **16-bit**:K2048 用尾梯系 (0.75,0.45,0.048)(op27 判决);tie 平台由
   fb_fix fail-soft 兜底(iter3 包络结论不变)。

**预期机理**:经典 GVR real 轴平均 ≥2 趟 count + 1 趟 collect;op26_r0 →
1 趟 M-ary(×1.25-1.4)+ 1 趟 collect(或 fused 后 ≈1.4 趟总),
对 radix(≈2 趟)fp32 首次在趟数账上占优;对 SGLang(1 趟流式)把差距
从 2.5:1 压到 1.4:1,叠加 GVR 免排序优势。

**mc 臂(iter6b)**:同一 R0 铺法进 cluster 内核(slice counts + DSMEM 归并);
cluster P2 eval 摊薄证伪的是 log 插值收益,**不是准入趟数收益**(R0 砍的是
全行趟数,per-slice 同样付费)。

**hugeN 探索(iter6c,预研)**:N≥262K 低 BS 的带宽墙需要 radix 式行内多 CTA
并行(GVR 阈值逻辑 + grid 级 count 归并;cluster>4 GPC 证伪 → 需 GMEM atomic
归并 + 两段 kernel 或 cooperative launch)。先出可行性判决再定实现。

## 4 · 验证流水线(8 GPU 并行)

1. **host 筛选**(`screen_r0_qfracs.py`,op22rr bundles 全 (scenario,K,dtype,N)):
   候选梯 {w3a, w3a+0.75(M4), base3, K2048尾梯, uniform-h M4} × 经典窗
   → 静态接受率 / rung count 分布(slot 容量)/ miss 时括号质量。
2. **smoke + 291 门禁**(gate_op26.py 扩 op26_r0;GPU 分片)。
3. **单格 nsys 消融**(8 卡分片):缺口代表格(K2048 fp16 262K、fp32 65K vs
   sglang 带、1cta 131K、mc 8192)+ win 保持格;臂对 = anchor + op26_r0
   (+ radix_cutedsl 同批直接比值)。
4. 赢了 → 全网格 81 批(8 卡 dtype×K 分片,~70min)→
   update_report_op26_iter5.py 系 last-writer 扩 op26_r0。

## 5 · 里程碑判据

- M1(本 session):筛选出静态接受率 ≥0.9(real 轴)的梯 + kernel v0 smoke 过。
- M2:门禁 291 绿 + 缺口代表格 nsys:K2048 16-bit @262K vs radix ≥0.9(from 0.45),
  fp32 65K vs sglang ≥1.0(from 0.75-0.79),win 格无回归(≥0.97×iter5)。
- M3:全网格 op26_r0 vs radix_cutedsl gm ≥1.0 且 8K-262K win10 率显著抬升;
  vs anchor 无 <0.9 新洞。
- M4(mc/hugeN):另立判据。
