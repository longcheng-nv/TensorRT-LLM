max_input_len: ${MAX_INPUT_LEN}
max_seq_len: ${MAX_SEQ_LEN}
cuda_graph_config:
    enable_padding: true
    batch_sizes: ${CUDA_GRAPH_BATCH_SIZES}
kv_cache_config:
    # G1: KV reuse OFF is MANDATORY for DSv4 cuda_graph stability
    # (sparse-MLA + KV reuse triggers async CUDA OOB in deepseek_v4.py).
    enable_block_reuse: false
    enable_partial_reuse: false
    tokens_per_block: 128
    dtype: ${KV_CACHE_DTYPE}
moe_config:
    max_num_tokens: ${MOE_MAX_NUM_TOKENS}
    backend: ${MOE_BACKEND}
${MOE_LP_COMBINE_BLOCK}print_iter_log: true
stream_interval: ${STREAM_INTERVAL}
enable_attention_dp: ${ATTN_DP}
${G10_ADP_BLOCK}enable_autotuner: true
${ALLREDUCE_BLOCK}${SPECULATIVE_CONFIG_BLOCK}
sparse_attention_config:
    algorithm: deepseek_v4
    enable_heuristic_topk: ${ENABLE_HEURISTIC_TOPK}
