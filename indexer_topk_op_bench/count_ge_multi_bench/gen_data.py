"""Generate V4-Pro (K=1024, beta_moderate) logit rows at N in {4K..256K},
matching indexer_topk_op_bench/report input (synth_data.get_bundle), and dump
fp32 + fp16 binaries for the block_count_ge multi-threshold micro-bench.

For each (dtype, N):
  data/logits_<dtype>_N<N>.bin   : N contiguous values in that dtype
  data/thr_N<N>.txt              : 8 fp32 thresholds spanning the K-th-rank
                                   neighborhood (rank K-ish quantiles); the
                                   count cost is threshold-value-independent,
                                   these are only for realism/repeatability.
"""
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "harness"))
from synth_data import get_bundle  # noqa: E402

N_SEQ = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
K = 1024  # V4 Pro
DTYPES = {"fp32": torch.float32, "fp16": torch.float16}

for N in N_SEQ:
    # Same call the report uses (K=1024 -> v4pro, beta_moderate, cr=4, seed=42).
    b = get_bundle(K, torch.float32, N, cfg="beta_moderate", seed=42)
    row_f32 = b["logits"][0, :N].to(torch.float32).contiguous().cpu().numpy().astype(np.float32)
    # 8 thresholds around the K-th largest value (rank K-1) +/- neighbors.
    srt = np.sort(row_f32)[::-1]
    kth = float(srt[K - 1])
    span = float(srt[max(0, K // 2)] - srt[min(N - 1, K + K // 2)])
    thr8 = (kth + (np.linspace(-0.5, 0.5, 8) * (span if span > 0 else 1.0))).astype(np.float32)
    (HERE / "data" / f"thr_N{N}.txt").write_text(" ".join(f"{t:.8g}" for t in thr8))
    for name, dt in DTYPES.items():
        arr = torch.from_numpy(row_f32).to(dt).contiguous().cpu().numpy()
        arr.tofile(HERE / "data" / f"logits_{name}_N{N}.bin")
    print(f"N={N}: kth={kth:.5f} thr=[{thr8[0]:.4f}..{thr8[-1]:.4f}] "
          f"fp32={row_f32.nbytes}B", flush=True)
print("done")
