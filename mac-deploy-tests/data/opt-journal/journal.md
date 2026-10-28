| iter | sha | description | decode ms/tok (median of 3) | tok/s | hit | GB/tok | SSD wait ms | ISL1024 prefill s | ISL1024 decode tok/s | parity | probe |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | e9ff5a1 | baseline: upstream a4df33c + hf.py/server.py patches, cache 18, SPEC_K 3 | 306 | 3.19 | 0.76 | 1.20 | 122 | 14.7 | 3.18 | PASS | ok |
| 1 | f9d8137 | iter1: drop per-index-layer mx.eval on decode; merge idx/pred host syncs into one | 286 | 3.79 | 0.75 | 1.20 | 123 | 14.9 | 3.18 | PASS | ok |
| 2 | 04db30f | iter2: fuse Block elementwise chains (mHC mixes+Sinkhorn+hc_pre+RMSNorm; hc_post+next chain) into 2 compiled graphs per layer | 290 | 3.47 | 0.76 | 1.20 | 122 | 14.75 | 3.27 | PASS | ok |
| 3 | ccfe179 | iter3: SPEC_K default 3 -> 4 (speculation precision 87%, recall 58%) [cache 18, SPEC_K 4] | 273 | 3.65 | 0.74 | 1.29 | 124 | 14.64 | 3.23 | PASS | ok |
| 4 | 5f820e5 | iter4: second-level speculation (l+2 top-k) implemented; measured negative (k=1/2/3: +3/+6/+14% ms), default OFF [cache 18, SPEC_K 4] | 270 | 3.63 | 0.74 | 1.29 | 124 | 14.57 | 3.13 | PASS | ok |
| 5 | 77a63c6 | iter5: MergedLinear -- wq_a+wkv and shared w1+w3 as single quantized_matmul (−2 matmuls, −2 act_quant per layer) [cache 18, SPEC_K 4] | 270 | 3.73 | 0.73 | 1.31 | 127 | 14.68 | 3.06 | PASS | ok |
| 6 | ff97db0 | iter6: io_threads 8 vs 16 vs 24 measured (275/280, 264/273 ms vs 266-273): no gain, kept 8; speculation confidence filter analysed (linear trade-off, not implemented) [cache 18, SPEC_K 4] | 259 | 3.54 | 0.74 | 1.31 | 123 | 14.58 | 3.25 | PASS | ok |
