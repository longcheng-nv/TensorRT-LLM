# SPDX-License-Identifier: Apache-2.0
B=["synth seqlen best 2048 bf16","synth seqlen worst 2048 bf16","real seqlen v32 bf16"]
print("\n".join(B))
