# SPDX-License-Identifier: NVIDIA
# op17 crux measurement: at BS=1, how does the wall-clock of G *independent*
# full-N count_ge scans scale with G?
#
# Premise under test (the make-or-break for the threshold-portfolio idea):
#   single-CTA-per-row is bandwidth-STARVED (~1/148 of peak), so running G CTAs
#   that each redundantly read the whole row + do count_ge should cost ~the same
#   wall time as 1 CTA up to some plateau G* = peak_BW / single_CTA_BW.
#   G* is exactly "how many initial thresholds we can try for FREE".
#
# Method mirrors harness/sweep.py: 512MB L2 evict + CUDA-graph replay + cudaEvent
# cold-L2 median. Each Triton program = one CTA (1024 threads = num_warps=32),
# grid=(G,), each program grid-strides over the full row (faithful to GVR's
# 1024-thread single-CTA count_ge pass).
import torch, triton, triton.language as tl

DEV = "cuda"
NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
_EVICT = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=DEV)


@triton.jit
def count_ge_redundant(x_ptr, thr, n, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros((), tl.int32)
    for start in range(0, n, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=-1e30)
        acc += tl.sum((x >= thr).to(tl.int32))
    tl.store(out_ptr + pid, acc)


def make_call(x, G, thr, BLOCK=8192):
    n = x.numel()
    out = torch.empty(G, dtype=torch.int32, device=DEV)
    grid = (G,)

    def call():
        count_ge_redundant[grid](x, thr, n, out, BLOCK=BLOCK, num_warps=32)

    return call


def time_cold(call, cold_reps=40, warmup=5):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            call()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    cold = []
    for _ in range(cold_reps):
        _EVICT.uniform_(0, 1)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); g.replay(); e1.record()
        torch.cuda.synchronize()
        cold.append(e0.elapsed_time(e1) * 1e3)  # us
    cold.sort()
    del g
    return cold[len(cold) // 2]


if __name__ == "__main__":
    torch.manual_seed(42)
    Ns = [8192, 16384, 32768, 65536, 131072, 262144]
    Gs = [1, 2, 4, 8, 16, 32, 64, 128, NUM_SMS]
    print(f"NUM_SMS={NUM_SMS}")
    print(f"{'N':>8} | " + " ".join(f"G={g:<3d}" for g in Gs) + " | plateau_G* (t<=1.5x of G=1)")
    for N in Ns:
        x = torch.randn(N, dtype=torch.float32, device=DEV)
        thr = 0.0
        ts = []
        for G in Gs:
            call = make_call(x, G, thr)
            ts.append(time_cold(call))
        base = ts[0]
        # largest G whose time stays within 1.5x of the single-CTA time
        gstar = 1
        for G, t in zip(Gs, ts):
            if t <= 1.5 * base:
                gstar = G
        row = " ".join(f"{t:6.1f}" for t in ts)
        print(f"{N:>8} | {row} | G*={gstar}  (base={base:.1f}us)")
