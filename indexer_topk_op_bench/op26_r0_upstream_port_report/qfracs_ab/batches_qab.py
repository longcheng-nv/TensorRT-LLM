# SPDX-License-Identifier: Apache-2.0
"""qfracs A/B batch plan — phase 1: fp32, BS=1 seq-len scan (synth 3-axis + real)."""
B = []
for scen in ("best", "worst"):
    for K in (512, 1024, 2048):
        B.append(f"synth seqlen {scen} {K} fp32")
for m in ("flash", "pro", "v32"):
    B.append(f"real seqlen {m} fp32")
print("\n".join(B))
