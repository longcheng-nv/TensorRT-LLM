# COST — GVR 复盘 + omni-kernel v2 升级会话(2026-07-12)

> 权威总额以 `/cost` 为准(惯例同 op26 COST.md)。
> 状态:**待用户贴 /cost 输出回填**;下方为会话内可直接核算的部分。

## 已知核算(会话内工具返回的实测值)

| 项 | 值 | 备注 |
|---|---|---|
| 考古 subagent ×5(token) | **730,245** | Era-0 94,751 + Era-1 202,259 + op1-16 170,620 + op17-21 154,384 + op22-27 108,231 |
| 考古 subagent 墙钟 | ~5.6 min(并行) | 314s/409s/318s/366s/282s,后台并行 |
| 主循环 token | 未单列 | 含 5 章回填、双语 HTML(~1272KB 产物)、两张图、SKILL v2 改写、两次 GitHub 推送、缺口分析 |
| GPU-h | **0** | 本会话纯档案考古/写作,无 GPU 计时任务 |

## 产出清单(对账用)

- TRT-LLM 仓 commits:aabd114db3(复盘+提案+图)、281ad942b9(双语 HTML)、
  f5173ddfb4(op21 progress 图)、skill_v2_draft 存档 ×2、OMNI_KERNEL_V2_GAPS.md
- 个人仓 longcheng-nv/omni-distill:6b352c7(v1 快照)、27fe8ed(v2)
- omni-kernel live:SKILL.md v2 + templates ×6 + scripts ×4 + LEARNINGS 回填

## 权威总额(/cost)

```
<待回填:粘贴 /cost 输出>
```

## 参照锚(历史战役)

- op26 全战役:15 GPU-h + ~$108
- op21 旗舰战役(iter0.5-14):≈$797
