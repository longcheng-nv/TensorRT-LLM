# SPDX-License-Identifier: Apache-2.0
"""bundle-v2 4-arm A/B batch plan: full 77-cell fp32 grid."""
B = []
for scen in ("best", "worst"):
    for K in (512, 1024, 2048):
        B.append(f"synth seqlen {scen} {K} fp32")
for m in ("flash", "pro", "v32"):
    B.append(f"real seqlen {m} fp32")
print("\n".join(B))
