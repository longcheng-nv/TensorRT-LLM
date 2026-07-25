# gvr-repair35 — 战役状态与接续指南 (2026-07-25)

- **Campaign ID**: `z3z1e5gn017d56m473sxnjy0kg` (display gvr-repair35, artifact
  indexer-topk-repair35-hetero); 两条同名 Failed (xnpb…/nz28…) = 废弃的
  baseline-solution prepare 尝试, 忽略。
- **目标**: 修复 2 臂 combined dispatch 的 35 个 <0.95 输格 (16k-256k ×
  BS256-1024 带 + pro_1024k@BS16 口袋) — 21 个主 workload >1.0×, 8 个 guard
  (*_guard) ≥0.95×; GVR 骨架强制; exact 逐行 (异质参照)。
- **反剥削设计 (吸取 R5 + fresh champion DQ)**: workload = 异质行 (batch 行
  循环全层, gen_inputs cycle); 逐层 mand/tie 参照存 asset; 外部终判另加
  原地突变测试。
- **baseline**: e612 PR head (op40 gvrpkg40b) 本地 nsys cold-L2 外测,
  per-workload pick_config(fp32, BS, N) 公平配置; baselines_repair.jsonl
  (29 格, gm 30.6µs); nsys rep = baseline_repair.nsys-rep。
- **pool**: 3×codex gpt-5.6-sol (high) + 3×n3 opus-4.8 — 无 fable-5 死槽。
  max-rounds 8, stagnation 3, effort-high 默认时限。
- **平台缺口备案**: --baseline-solution 的 prepare 评测不带 --asset (gen_inputs
  FileNotFound) → 必须走外测 --baselines 路线。
- **接续**: kf campaign show z3z1e5gn017d56m473sxnjy0kg; 收割 = kernel list
  --top → results --output-dir → 门禁 (原地突变 + 异质行 + 35 格 nsys 配对
  A/B vs baselines_repair) → 若达标, 并入 combined dispatch 第三臂重算包络。
