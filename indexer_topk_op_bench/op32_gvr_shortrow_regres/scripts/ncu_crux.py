import os,sys
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parents[1]; BENCH=HERE.parents[0]
sys.path.insert(0,str(BENCH/"op22_temporal_fixed_hr_bench"))
sys.path.insert(0,str(BENCH/"op26_gvr_logfalsi_rs"/"src"))
import bundle_data_rr
from gvr_op26_op import gvr_cutedsl_op26
N=int(os.environ.get("N","8192")); K=int(os.environ.get("K","512"))
b=bundle_data_rr.get_bundle("real",K,torch.float32,N)
logits,pre,cr=b["logits"],b["preIdx"],b["cr"]
seq=torch.full((1,),N*cr,dtype=torch.int32,device="cuda")
for _ in range(3): gvr_cutedsl_op26(logits,pre,seq,K,compress_ratio=cr)
torch.cuda.synchronize()
gvr_cutedsl_op26(logits,pre,seq,K,compress_ratio=cr)  # profiled launch
torch.cuda.synchronize()
