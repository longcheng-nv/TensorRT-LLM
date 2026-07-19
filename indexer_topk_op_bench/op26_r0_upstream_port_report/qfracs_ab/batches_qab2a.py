# SPDX-License-Identifier: Apache-2.0
"""qfracs A/B phase 2a (light, GPU6): real BS axis fp32 + 16-bit spot checks."""
B = ["real bs v32 fp32",
     "real seqlen v32 bf16",
     "real seqlen v32 fp16",
     "synth seqlen best 2048 bf16",
     "synth seqlen worst 2048 bf16",
     "synth seqlen best 2048 fp16",
     "synth seqlen worst 2048 fp16"]
print("\n".join(B))
