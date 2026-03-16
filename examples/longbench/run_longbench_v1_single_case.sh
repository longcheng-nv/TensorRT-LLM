#!/bin/bash
# Find the LongBench v1 sample closest to target_isl across all subtasks and run single-sample evaluation
set -ex

# ---- Model ----
model_path="/home/scratch.trt_llm_data/llm-models/DeepSeek-V3.2-Exp-FP4-v2/"

# ---- Search targets ----
target_isl=32000          # Target ISL in tokens
min_osl=128               # Minimum OSL cap (skip subtasks with max_new_tokens below this)

# ---- Parallelism ----
ep=8

# ---- Batch / Tokens ----
max_batch_size=1
max_num_tokens=8192
max_seq_len=133120        # KV cache capacity upper bound (ISL + OSL)

# ---- KV Cache ----
kv_fraction=0.8
kv_cache_dtype="fp8"
tokens_per_block=64

# ---- Speculative (MTP) ----
MTP=1

# ---- Output ----
output_dir="tmp/longbench_single_case"
mkdir -p ${output_dir}

# Step 1: Scan all subtasks, find sample with ISL closest to target_isl and OSL cap >= min_osl
python3 -c "
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('${model_path}', trust_remote_code=True)

DATASETS = [
    'narrativeqa', 'qasper', 'multifieldqa_en', 'multifieldqa_zh',
    'hotpotqa', '2wikimqa', 'musique', 'dureader',
    'gov_report', 'qmsum', 'multi_news', 'vcsum',
    'trec', 'triviaqa', 'samsum', 'lsht',
    'passage_count', 'passage_retrieval_en', 'passage_retrieval_zh',
    'lcc', 'repobench-p'
]

MAX_NEW_TOKENS = {
    'narrativeqa': 128, 'qasper': 128, 'multifieldqa_en': 64, 'multifieldqa_zh': 64,
    'hotpotqa': 32, '2wikimqa': 32, 'musique': 32, 'dureader': 128,
    'gov_report': 512, 'qmsum': 512, 'multi_news': 512, 'vcsum': 512,
    'trec': 64, 'triviaqa': 32, 'samsum': 128, 'lsht': 64,
    'passage_count': 32, 'passage_retrieval_en': 32, 'passage_retrieval_zh': 32,
    'lcc': 64, 'repobench-p': 64
}

target = ${target_isl}
min_osl = ${min_osl}

best_ds, best_idx, best_len, best_diff, best_osl = '', 0, 0, float('inf'), 0

for ds_name in DATASETS:
    osl_cap = MAX_NEW_TOKENS.get(ds_name, 64)
    if osl_cap < min_osl:
        print(f'[SKIP] {ds_name:30s}  max_new_tokens={osl_cap} < min_osl={min_osl}')
        continue

    data = list(load_dataset('THUDM/LongBench', ds_name, split='test', trust_remote_code=True))
    for i, sample in enumerate(data):
        context = sample.get('context', '') + sample.get('input', '')
        token_len = len(tokenizer.encode(context, truncation=False))
        diff = abs(token_len - target)
        if diff < best_diff:
            best_ds, best_idx, best_len, best_diff, best_osl = ds_name, i, token_len, diff, osl_cap

    lens = [len(tokenizer.encode(s.get('context','') + s.get('input',''), truncation=False)) for s in data]
    print(f'[SCAN] {ds_name:30s}  samples={len(data):4d}  ISL range=[{min(lens):6d}, {max(lens):6d}]  max_new_tokens={osl_cap}')

print()
print(f'Best match: dataset={best_ds}, sample_idx={best_idx}, ISL={best_len}, OSL_cap={best_osl} (target_isl={target}, diff={best_diff})')
print(f'BEST_DS={best_ds}')
print(f'BEST_IDX={best_idx}')
print(f'BEST_ISL={best_len}')
print(f'BEST_OSL={best_osl}')
" 2>&1 | tee ${output_dir}/scan_log.txt

# Extract best match from scan results
BEST_DS=$(grep "^BEST_DS=" ${output_dir}/scan_log.txt | cut -d= -f2)
BEST_IDX=$(grep "^BEST_IDX=" ${output_dir}/scan_log.txt | cut -d= -f2)
BEST_ISL=$(grep "^BEST_ISL=" ${output_dir}/scan_log.txt | cut -d= -f2)
BEST_OSL=$(grep "^BEST_OSL=" ${output_dir}/scan_log.txt | cut -d= -f2)
echo "Selected: dataset=${BEST_DS}, sample_idx=${BEST_IDX}, ISL=${BEST_ISL}, OSL_cap=${BEST_OSL}"

# Step 2: Generate YAML config
cat <<EOF > ${output_dir}/extra_llm_api_options.yaml
cuda_graph_config:
    enable_padding: true
    max_batch_size: ${max_batch_size}
kv_cache_config:
    free_gpu_memory_fraction: ${kv_fraction}
    enable_block_reuse: false
    tokens_per_block: ${tokens_per_block}
    dtype: ${kv_cache_dtype}
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

# Step 3: Run single-sample accuracy evaluation
python examples/longbench/eval_longbench_v1.py \
    --model_path ${model_path} \
    --dataset ${BEST_DS} \
    --longbench_path ./LongBench \
    --output_dir ${output_dir} \
    --backend pytorch \
    --tp_size ${ep} \
    --moe_ep_size ${ep} \
    --max_batch_size ${max_batch_size} \
    --max_seq_len ${max_seq_len} \
    --max_num_tokens ${max_num_tokens} \
    --dsa_sparse \
    --enable_heuristic_topk \
    --mtp ${MTP} \
    --kv_cache_dtype ${kv_cache_dtype} \
    --tokens_per_block ${tokens_per_block} \
    --kv_cache_fraction ${kv_fraction} \
    --enable_attention_dp \
    --use_cuda_graph \
    --cuda_graph_padding_enabled \
    --enable_chunked_prefill \
    --start_idx ${BEST_IDX} \
    --num_samples 1 \
    --print_iter_log
