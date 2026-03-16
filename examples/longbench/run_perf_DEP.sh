#! /bin/bash
set -ex
model_card="deepseek-ai/DeepSeek-V3.2-Exp"
model_path="/home/scratch.trt_llm_data/llm-models/DeepSeek-V3.2-Exp-FP4-v2/"
max_batch_size=1
kv_fraction=0.8
multi_round=1 # change this to 1 to make it faster
num_prompts=$((${max_batch_size} * ${multi_round}))

dataset_isl=4096
dataset_osl=131072
ep=8
MTP=1
max_num_tokens=$(( (${max_batch_size}+${dataset_isl}+128+63)/64*64 ))

export PATH=${HOME}/.local/bin:${PATH}
sudo nvidia-smi -pm 0; sudo nvidia-smi -pm 1; sudo nvidia-smi boost-slider --vboost 4

log_dir=tmp/ds32/ds_B${max_batch_size}_ISL${dataset_isl}_OSL${dataset_osl}_fp4_heuristic_topk_perf_CUDAGrapha
mkdir -p ${log_dir}
timestamp=$(date +'%m%d%H%M')
log_prefix=${log_dir}/run_${num_prompts}_${dataset_isl}_${dataset_osl}_DEP${ep}_${max_batch_size}_${max_num_tokens}_FP4_B200_MTP${MTP}_${timestamp}

cat <<EOF > extra-llm-api-config.yml
cuda_graph_config:
    enable_padding: true
    max_batch_size: ${max_batch_size}
kv_cache_config:
    free_gpu_memory_fraction: ${kv_fraction}
    enable_block_reuse: false
    tokens_per_block: 64
    dtype: fp8
enable_chunked_prefill: false
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

export TRT_LLM_HOME=/home/scratch.loncheng_gpu/workspace/perf/TensorRT-LLM
python $TRT_LLM_HOME/benchmarks/cpp/prepare_dataset.py \
    --tokenizer=${model_path} --stdout token-norm-dist --num-requests=${num_prompts} \
    --input-mean=${dataset_isl} --output-mean=${dataset_osl} --input-stdev=0 --output-stdev=0 > dataset.json

#export TLLM_PROFILE_RECORD_GC=1
#export TLLM_PROFILE_START_STOP=500-550
#nsys profile -o ${log_prefix} -f true -t cuda,nvtx -c cudaProfilerApi --cuda-graph-trace node \
#export TLLM_NVTX_DEBUG=1
#export TLLM_PROFILE_RECORD_GC=1
# prefill: 515-520
# decode: 550-1000
#export TLLM_PROFILE_START_STOP=16000-16050
#export TLLM_LLMAPI_ENABLE_NVTX=1
#nsys profile -o ${log_prefix}_decode_trace -f true -t 'cuda,nvtx,python-gil' -c cudaProfilerApi --cuda-graph-trace node -e --trace-fork-before-exec=true \
trtllm-bench -m ${model_card} --model_path ${model_path} throughput \
    --tp ${ep} \
    --ep ${ep} \
    --warmup 1 \
    --dataset dataset.json \
    --backend pytorch \
    --max_batch_size ${max_batch_size} \
    --max_num_tokens ${max_num_tokens} \
    --kv_cache_free_gpu_mem_fraction ${kv_fraction} \
    --concurrency ${max_batch_size} \
    --extra_llm_api_options extra-llm-api-config.yml \
    --num_requests ${num_prompts} \
    --streaming |& tee ${log_prefix}.txt

