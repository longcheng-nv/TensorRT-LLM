# M2 verification for PR #15756: does TRUNCATE emit garbage (uninitialized
# s_indices tail) instead of -1 when top_k > filtered_topk_smem_input_size?
#
# Trigger analysis: underfill needs total retained candidates < topk_remaining,
# i.e. top_k > S. fp32 large_occupancy (num_buffer=2, Uint16 idx @ N<=64K):
# S = 32KB / (2*2B) = 8192.  So top_k=16384 (allowed max) > S=8192.
# Adversarial data: all elements identical -> single coarse bin, threshold bin
# holds all N elements, only S survive TRUNCATE; every refine round the same.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import cutlass
import cutlass.cute as cute

from pk.top_k.filtered_top_k_decode_varlen import FilteredTopKKernelVarlenDecode

DTYPE = cutlass.Float32
TORCH_DTYPE = torch.float32
NUM_TOKENS = 32768
TOP_K = 16384
BATCH = 4          # large_occupancy is a compile-time flag we set directly
NEXT_N = 1

def compile_kernel(overflow_policy, large_occupancy=True):
    n_rows, n_cols, n_batch = cute.sym_int(), cute.sym_int(), cute.sym_int()
    input_fake = cute.runtime.make_fake_compact_tensor(
        DTYPE, (n_rows, n_cols), stride_order=(1, 0), assumed_align=32)
    if overflow_policy == "GMEM_SPILL":
        buffer_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32, (cute.sym_int(), cute.sym_int(), cute.sym_int()),
            stride_order=(2, 1, 0), assumed_align=32)
    else:
        buffer_fake = None
    seqlen_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (n_batch,), stride_order=(0,))
    out_idx_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (n_rows, TOP_K), stride_order=(1, 0))
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    func = FilteredTopKKernelVarlenDecode(
        DTYPE, NUM_TOKENS, TOP_K, NEXT_N,
        num_copy_bits=256, return_val=False,
        large_occupancy=large_occupancy,
        overflow_policy=overflow_policy,
    )
    print(f"[{overflow_policy}] S={func.filtered_topk_smem_input_size} "
          f"idx_type={func.index_type} nb={func.num_buffer_smem_input_idx} "
          f"threads={func.num_threads_per_cta} "
          f"enable_truncate={getattr(func,'enable_truncate',None)}")
    ck = cute.compile(func, input_fake, None, buffer_fake, seqlen_fake,
                      out_idx_fake, None, stream=fake_stream,
                      min_blocks_per_mp=1, options="--enable-tvm-ffi")
    return ck, func

def run(ck, func, logits, seq_lens, overflow_policy):
    num_rows = logits.shape[0]
    out = torch.full((num_rows, TOP_K), -777, dtype=torch.int32, device="cuda")
    if overflow_policy == "GMEM_SPILL":
        buf = torch.empty(num_rows, 2, NUM_TOKENS, dtype=torch.int32, device="cuda")
    else:
        buf = None
    ck(logits, None, buf, seq_lens, out, None)
    torch.cuda.synchronize()
    return out

def check(tag, out, eff_len):
    bad = 0
    for r in range(out.shape[0]):
        row = out[r]
        valid = (row >= 0) & (row < eff_len)
        neg1 = row == -1
        sentinel = row == -777      # kernel never wrote this slot at all
        garbage = ~(valid | neg1 | sentinel)
        n_v, n_n, n_s, n_g = int(valid.sum()), int(neg1.sum()), int(sentinel.sum()), int(garbage.sum())
        # duplicates among valid
        v = row[valid].sort().values
        n_dup = int(((v[1:] == v[:-1]).sum())) if n_v > 1 else 0
        status = "OK" if (n_v == TOP_K and n_g == 0 and n_s == 0 and n_dup == 0) else "BAD"
        if status == "BAD":
            bad += 1
        print(f"  {tag} row{r}: valid={n_v}/{TOP_K} neg1={n_n} unwritten={n_s} "
              f"out_of_range_garbage={n_g} dups={n_dup} -> {status}")
        if n_g > 0:
            g = row[garbage]
            print(f"    sample garbage values: {g[:8].tolist()}")
    return bad

torch.manual_seed(0)
seq_lens = torch.full((BATCH,), NUM_TOKENS, dtype=torch.int32, device="cuda")

print("=" * 70)
print(f"Config: fp32 N={NUM_TOKENS} top_k={TOP_K} batch={BATCH} large_occupancy=True")
print("=" * 70)

# --- Case 1: all-identical logits (single coarse bin, maximal truncation) ---
logits_tie = torch.full((BATCH, NUM_TOKENS), 1.0, dtype=TORCH_DTYPE, device="cuda")

# --- Case 2: production-like random logits, same shapes (severity bound) ---
logits_rnd = torch.randn(BATCH, NUM_TOKENS, dtype=TORCH_DTYPE, device="cuda")

for policy in ("TRUNCATE", "REREAD"):
    ck, func = compile_kernel(policy)
    print(f"\n--- {policy} / all-tie logits ---")
    out = run(ck, func, logits_tie, seq_lens, policy)
    b1 = check(policy + "/tie", out, NUM_TOKENS)
    print(f"--- {policy} / randn logits ---")
    out = run(ck, func, logits_rnd, seq_lens, policy)
    b2 = check(policy + "/rnd", out, NUM_TOKENS)
    print(f"[{policy}] bad rows: tie={b1} rnd={b2}")
