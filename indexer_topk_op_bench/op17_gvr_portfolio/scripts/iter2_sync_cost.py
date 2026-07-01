# SPDX-License-Identifier: NVIDIA
# op17 iter2: cost of the portfolio's leader-selection sync at BS=1.
#
# Portfolio structure: kernelA (grid=G) = each CTA does one count_ge over full N
# at its own threshold, writes count to global[pid]. Then a leader is chosen
# (count closest to K, >=K) and that CTA does the normal P3+P4 tail. The tail is
# work the baseline ALSO does (with looser cand), so it is NOT overhead. The
# overhead beyond baseline is only: (extra launch gap) + (read G counts + argmin).
#
# We measure that overhead as a conservative 2-kernel-in-CUDA-graph proxy:
#   t_sweep    = cold graph{ A(G=148) }              # ~= baseline 1 count pass (crux)
#   t_sweep_B  = cold graph{ A(G=148); B(argmin) }   # sweep + leader pick
#   overhead   = t_sweep_B - t_sweep
# Compare to baseline single-CTA count t_base = cold graph{ A(G=1) }.
# The single-kernel last-block-atomic design would cost <= this 2-kernel proxy.
import torch, triton, triton.language as tl

DEV = "cuda"
NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
_EVICT = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=DEV)


@triton.jit
def sweep_counts(x_ptr, thr_ptr, n, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    thr = tl.load(thr_ptr + pid)
    acc = tl.zeros((), tl.int32)
    for start in range(0, n, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=-1e30)
        acc += tl.sum((x >= thr).to(tl.int32))
    tl.store(out_ptr + pid, acc)


@triton.jit
def pick_leader(counts_ptr, G, K, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < G
    c = tl.load(counts_ptr + offs, mask=mask, other=1 << 30)
    # smallest count that is still >= K (tightest valid threshold)
    valid = tl.where(c >= K, c, 1 << 30)
    best = tl.min(valid)
    is_best = (valid == best) & mask
    pos = tl.min(tl.where(is_best, offs, 1 << 30))
    tl.store(out_ptr, pos)


def cold_us(build_graph, reps=60, warmup=8):
    g, tensors = build_graph()
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    cold = []
    for _ in range(reps):
        _EVICT.uniform_(0, 1); torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        cold.append(e0.elapsed_time(e1) * 1e3)
    cold.sort()
    return cold[len(cold) // 2]


def make_graph(x, G, with_B, BLOCK=8192):
    n = x.numel()
    thr = torch.zeros(G, dtype=torch.float32, device=DEV)  # all threshold 0.0 (worst-case count)
    counts = torch.empty(G, dtype=torch.int32, device=DEV)
    leader = torch.empty(1, dtype=torch.int32, device=DEV)
    Bpow = 1
    while Bpow < G:
        Bpow *= 2

    def build():
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            sweep_counts[(G,)](x, thr, n, counts, BLOCK=BLOCK, num_warps=32)
            if with_B:
                pick_leader[(1,)](counts, G, 512, leader, BLOCK=Bpow, num_warps=4)
        torch.cuda.current_stream().wait_stream(s)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            sweep_counts[(G,)](x, thr, n, counts, BLOCK=BLOCK, num_warps=32)
            if with_B:
                pick_leader[(1,)](counts, G, 512, leader, BLOCK=Bpow, num_warps=4)
        for _ in range(10):
            gr.replay()
        torch.cuda.synchronize()
        return gr, (thr, counts, leader)

    return build


if __name__ == "__main__":
    torch.manual_seed(42)
    print(f"NUM_SMS={NUM_SMS}")
    print(f"{'N':>8} | t_base(G=1) t_sweep(G=148) t_sweep+B  overhead=B  | (all cold-L2 us)")
    for N in [4096, 8192, 16384, 65536, 131072, 262144]:
        x = torch.randn(N, dtype=torch.float32, device=DEV)
        t_base = cold_us(make_graph(x, 1, False))
        t_sweep = cold_us(make_graph(x, NUM_SMS, False))
        t_sweep_B = cold_us(make_graph(x, NUM_SMS, True))
        print(f"{N:>8} | {t_base:9.1f} {t_sweep:12.1f} {t_sweep_B:10.1f} {t_sweep_B - t_sweep:10.2f}")
