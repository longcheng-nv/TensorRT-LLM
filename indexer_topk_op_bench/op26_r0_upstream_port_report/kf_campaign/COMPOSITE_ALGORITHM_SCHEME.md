# GVR-Composite 整合算法方案(方向 i,2026-07-25)

对应性能全貌:`COMPOSITE_ENVELOPE_20260725.md`(BS=1 gm 1.6531 / BS>1 gm 1.3334 / min 0.951 / 1615 格全 bar PASS)。

## 0. 算子契约

- 输入:`logits[b, npad]` fp32(有效长 `n_valid`,尾部 pad 不进 top-K)、`pre_idx[b, K]` int32(上一 decode 步 top-K,温启先验)、`n_valid`。
- 输出:`indices[b, K]` int32,**不要求保序**。
- 正确性:exact — 与 `torch.topk` 的 value 集合完全一致(tie-robust set 判卷:严格大于第 K 值的索引必须全出现,第 K 值并列名额可任选)。
- K ∈ {512(V4-Flash), 1024(V4-Pro), 2048(V3.2)};n 动态至 ~1.05M。

## 1. 顶层分派(host 侧,O(1) 查表)

分派键 = `(b, npad, K)` — 全部推理时可知,**不依赖 hit-rate 等不可知量**。

```
if b == 1:            → ARM-A  (R4 champion 28dc11f6)
else:                 → 查 250-rung 路由表[K, npad-rung, BS-rung]:
                          205 rungs → ARM-B (v3mt)
                          17  rungs → ARM-C (e6)
                          28  rungs → ARM-D (PR#16457 原 kernel 回退)
```

- 路由表 = `composite_dispatch_routing.json`(rung 粒度 = model×ISL×BS 实测格;落表为 (K, npad 区间, BS 区间) 常量表编译进 launcher)。
- PR 回退 rung 集中在 **BS 256-1024 × 16k-256k**(op39 LOCK 已证该吞吐域对 GVR 家族结构性不利;回退优于硬上,gm 1.3279→1.3334)。
- 非枚举 BS/npad 落在 rung 边界之间时取**保守侧**(向 PR 回退方向圆整);边界值须按 npad 实测锚定(R1 战役 SMALL_N=16384 vs npad 16387 的 dispatch-boundary 事故教训)。

## 2. ARM-A:R4 champion `28dc11f6`(BS=1,865 格 gm 1.6531 / 0 真实回退 / 865 exact)

GVR 骨架四阶段,内部按 `(npad, K)` 三档再分派:

| npad 档 | 执行形态 |
|---|---|
| ≤12288 | direct 1×1024 单块(阈值求解在 kC≥npad 下解析退化;不消费 pre_idx — 操作员已裁定合规并在 R4_CLOSEOUT 披露) |
| ≤262144 | **寄存器驻留 GVR** 1/4/8/16-CTA:整行载入寄存器,多 pass 零重扫;per-(tier,K) AR6/AR8 实测梯 |
| >262144 | 流式 16-CTA cluster |

四阶段:
- **P1(先验猜测)**:从 `pre_idx` gather K 个先验值 → hint-CCDF **两级直方图** → 放置 8 分位阈值梯(单遍多阈值,替代朴素单点猜)。
- **P2(阈值求解)**:单 pass **8 阈值并行计数** + **log 域 secant 括号收缩**(~9×/pass);tie 平台(plateau)时精确回退,保 exact。
- **P3(收集)**:DSMEM(cluster 分布式 SMEM)收集,奇偶双 bank 消 conflict。
- **P4(精确尾)**:CTA0 **4×8bit radix** 精选第 K 边界 + **tie-ticket** 分配并列名额。
- 源码:`harvest/r3_28dc11f6/gvr.cu` · fork 分支 `kf/r4-champion-final-bs1`。

## 3. ARM-B:v3mt(BS≥2 主力,205 rungs;单臂 750 格 gm 1.3064)

= **r3_v11 批量化(op38 v3)** ⊕ **per-K rung 分位 multithresh(op41)**:

- base r3_v11:R4 血统寄存器驻留 GVR 的 **grid.y 批量化**(每行独立 CTA 组,行间无耦合);op38 v3 附带 15 个 per-(model, npad, BS) 配置开关(TB/CS/MAXV/AR/HS),每个开关经 confirm-probe 门控(新 cfg 必须在该 key 全层确认)。
- op41 multithresh(纯常数改动,零 P2 成本,骨架不动):P1 阈值梯分位按 K 定制 —
  - AR4:K2048 → {55, 88}(gated npad<49152 ‖ >98304;例外 npad≈65600),K1024 → {35, 70},K512 → 原 {25, 65};
  - AR6:K2048 → {25, 50, 75, 92},其余 → {15, 40, 70, 92}。
  - 依据:P2 收敛是 layer 属性,最优 rung 位置跟随 K(模型族 hit-rate 分布);无全局最优集(已证伪)。
- **异质批已验证**:32/32 格 ≥1.00(生产真实形态);复制行轴无格 <0.97。
- 源码:`op41_gvr_hint_mt/src/v3mt` @bce921d0b1(base:`op38_r3v11_bs/kernel_bs`)。

## 4. ARM-C:e6(大 npad × BS≥256 胜区,17 rungs;win band 至 2.56×)

op39 顺序发射收割臂(arm v2 + iter12-14 收割):

- **K0**:hint-min + clustered-sample 分位阈值(从 pre_idx 采样估计初始阈值)。
- **K1**:fused tile collect + last-CTA 精确 4 级 reduce(边收集边归约,单波流过大行)。
- **K2**:second-chance rescue(undershoot 时的补救 pass,保 exact;empty-launch 税 +1.6-5.7% 已量化接受)。
- 调度形态:per-case **chunks ladder**;**ILP 按 BS 分派**(BS<512 用 8,≥512 用 4 — 统一 ILP-8 在 nsys 包络 BS≥512 回退已证);`__ldcs` 流式 load @npad≥262144(+2-9.5%);npad≤8192 走小行单 launch。
- 已证伪并排除:cp.async 双缓冲(0.93-0.98)、CDP2 尾 launch(-rdc 全局 15-20% 税)。
- 750/750 tie-aware exact + 对抗数据(常量/近 tie)全绿;**异质行未验证**(遗留项,见 §7)。
- 源码:`op39_gvr_bsx/src/arm_v2`(e6 = iter14 收割终态)。

## 5. ARM-D:PR#16457 原 kernel(28 rungs 回退)

原生 cuteDSL GVR(P1 preIdx 种子 + log-secant + exact collect,含 vseed 与 K2048 尾梯)。回退格拿 1.0×,即"不伤害"保证——这是 ≤5% 回退 bar 在吞吐墙域的达成机制。

## 6. Exactness 论证(整体)

每臂在各自全网格独立过 tie-robust set 判卷(ARM-A 865/865 · ARM-B 750/750 复制行 + 32/32 异质 · ARM-C 750/750 + 对抗集 · ARM-D 生产本体);顶层分派只选 kernel、不改数值路径,故复合算子 exact ⇔ 各臂 exact。

## 7. 落地计划与遗留风险

1. **实现**:单 custom-op 入口(扩展 trtllm gvr launcher),四臂编译进同一扩展,路由常量表 + 保守圆整;R4 champion 与 v3mt 同血统(寄存器驻留 GVR),可共享大部分 device 代码。
2. **统一复测**:三套网格是各自战役时点的拼接估计 → ship 前单机对最终 dispatcher 做一次背靠背 1615 格 nsys cold-L2 全网格终判(含 per-rung 锚检查)。
3. **e6 异质行补验**:其 win band(大 npad × BS≥256)需补 32 格异质批验证;若回退,将该 17 rungs 降级给 v3mt/PR(对 gm 影响 ≤0.5%)。
4. **边界锚定**:所有 rung 边界(npad 档、BS 档)在复测中显式放探针格,防 dispatch-boundary artifact。
5. **上游合规**:ARM-A ≤12288 direct 路径不消费 pre_idx 的边界裁定需在 PR 描述中原样披露。
