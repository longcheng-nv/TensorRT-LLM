#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ============================================================================
# LongBench V1 accuracy + end-to-end performance for DeepSeek V3.2
#   with DSA sparse attention + heuristic TopK + MTP
#
# Workflow (following the pattern from blog16):
#   Step 1: trtllm-eval longbench_v1  -->  accuracy scores + dumped_ids.json
#   Step 2: trtllm-bench throughput   -->  TTFT / TPOT / throughput
# ============================================================================
set -ex

# ---- Model ----
model_card="deepseek-ai/DeepSeek-V3.2-Exp"
model_path="/home/scratch.trt_llm_data/llm-models/DeepSeek-V3.2-Exp-FP4-v2/"

# ---- Parallelism ----
ep=8

# ---- Batch / Tokens ----
max_batch_size=32
max_num_tokens=8192            # chunked prefill: each chunk <= this size
max_seq_len=32768              # 130K context window (KV cache capacity)

# ---- KV cache ----
kv_fraction=0.8

# ---- Speculative (MTP) ----
MTP=1

# ---- Output ----
timestamp=$(date +'%m%d%H%M')
output_dir="tmp/ds32_longbench_v1_dsa_heuristic_topk_${timestamp}"
mkdir -p ${output_dir}

# ---- GPU boost (optional, needs sudo) ----
sudo nvidia-smi -pm 0; sudo nvidia-smi -pm 1; sudo nvidia-smi boost-slider --vboost 4 || true

# ============================================================================
# Generate extra_llm_api_options YAML
#   This single YAML drives both trtllm-eval and trtllm-bench with identical
#   engine configuration, matching run_perf_DEP.sh exactly.
# ============================================================================
cat <<EOF > ${output_dir}/extra_llm_api_options.yaml
cuda_graph_config:
    enable_padding: true
    max_batch_size: ${max_batch_size}
kv_cache_config:
    free_gpu_memory_fraction: ${kv_fraction}
    enable_block_reuse: false
    tokens_per_block: 64
    dtype: fp8
enable_chunked_prefill: true
print_iter_log: true
enable_attention_dp: true
moe_config:
    backend: CUTLASS
speculative_config:
    decoding_type: MTP
    num_nextn_predict_layers: ${MTP}
sparse_attention_config:
    algorithm: dsa
    enable_heuristic_topk: true
EOF

# ============================================================================
# Step 1: Accuracy — trtllm-eval longbench_v1
#   --output_dir will dump dumped_ids.json (trtllm-bench-compatible dataset)
# ============================================================================
echo "===== Step 1: trtllm-eval longbench_v1 ====="
trtllm-eval \
    --model ${model_path} \
    --backend pytorch \
    --tp_size ${ep} \
    --ep_size ${ep} \
    --max_batch_size ${max_batch_size} \
    --max_num_tokens ${max_num_tokens} \
    --max_seq_len ${max_seq_len} \
    --kv_cache_free_gpu_memory_fraction ${kv_fraction} \
    --disable_kv_cache_reuse \
    --extra_llm_api_options ${output_dir}/extra_llm_api_options.yaml \
    longbench_v1 \
    --output_dir ${output_dir} \
    |& tee ${output_dir}/eval_log.txt

# ============================================================================
# Step 2: Performance — trtllm-bench throughput
#   Uses dumped_ids.json from Step 1 so ISL/OSL exactly match the real
#   LongBench prompts the model was evaluated on.
# ============================================================================
echo "===== Step 2: trtllm-bench throughput ====="
trtllm-bench \
    -m ${model_card} \
    --model_path ${model_path} \
    throughput \
    --tp ${ep} \
    --ep ${ep} \
    --warmup 1 \
    --dataset ${output_dir}/dumped_ids.json \
    --backend pytorch \
    --max_batch_size ${max_batch_size} \
    --max_num_tokens ${max_num_tokens} \
    --kv_cache_free_gpu_mem_fraction ${kv_fraction} \
    --concurrency ${max_batch_size} \
    --extra_llm_api_options ${output_dir}/extra_llm_api_options.yaml \
    --streaming \
    --report_json ${output_dir}/bench_report.json \
    |& tee ${output_dir}/bench_log.txt

echo "===== Done ====="
echo "Eval log:    ${output_dir}/eval_log.txt"
echo "Bench log:   ${output_dir}/bench_log.txt"
echo "Bench report: ${output_dir}/bench_report.json"
echo "Dumped IDs:  ${output_dir}/dumped_ids.json"
