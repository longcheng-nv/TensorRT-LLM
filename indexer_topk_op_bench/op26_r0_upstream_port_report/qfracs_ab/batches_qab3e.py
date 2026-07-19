# SPDX-License-Identifier: Apache-2.0
B=["synth seqlen best 2048 fp16","synth seqlen worst 2048 fp16","real seqlen v32 fp16"]
print("\n".join(B))
