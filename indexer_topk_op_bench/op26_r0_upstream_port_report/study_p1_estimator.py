# P1 estimator study: which single-point statistic of the gathered prev-topK
# values best predicts the current row's K-th-largest threshold, and which
# lands its full-row count inside the R0 admission window [K, kC]?
# Runs on the REAL decode captures (flash/pro all ISL rungs + v32).
import sys, torch
sys.path.insert(0, "/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/harness")
import real_data_v4cap as RD4

torch.cuda.set_device(0)
K = 512
KC = 3072  # op26 kc-diet, K512 admission window [K, kC]

QS = [0.85, 0.65, 0.50, 0.42, 0.35, 0.25, 0.15]

def study(model, isls):
    print(f"\n=== {model} (K={K}) — count(row >= pred); admissible iff in [{K},{KC}] ===")
    hdr = f"{'isl':>6} {'N':>7} {'hit':>5} | {'true':>5} | mean  medn  " + "  ".join(f"q{q:.2f}"[1:] for q in QS) + "   min(prevThr)"
    print(hdr)
    for isl in isls:
        try:
            cells = RD4.prepare(model, isl)
        except Exception as e:
            print(f"{isl:>6} SKIP {e}"); continue
        for layer, d in cells.items():
            row = d["logits"].float().cuda()
            pre = d["preidx"].int().cuda()
            N = row.numel()
            gath = row[pre.long()]
            true_thr = torch.topk(row, K).values[-1]
            hit = (row[pre.long()] >= true_thr).float().mean().item()  # = |prevTopK ∩ curTopK|/K
            preds = {"mean": gath.mean(), "medn": gath.median()}
            gs = torch.sort(gath, descending=True).values
            for q in QS:
                preds[f"q{q:.2f}"] = gs[min(K - 1, max(0, int(q * K) - 1))]
            preds["min"] = gath.min()
            cnts = {k: int((row >= v).sum()) for k, v in preds.items()}
            marks = {k: ("*" if K <= c <= KC else " ") for k, c in cnts.items()}
            true_cnt = int((row >= true_thr).sum())
            cells_str = "  ".join(f"{cnts[k]:>5d}{marks[k]}" for k in preds)
            print(f"{isl:>6} {N:>7} {hit:.2f} | {true_cnt:>5} | {cells_str}  L{layer}")
            break  # first bench layer per rung is enough for the study

for m, isls in [("flash", ["64k", "128k", "256k", "512k", "1024k"]),
                ("pro", ["128k", "1024k"])]:
    study(m, isls)
