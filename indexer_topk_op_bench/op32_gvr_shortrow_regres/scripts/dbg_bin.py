import sys, math
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parents[1]; BENCH=HERE.parents[0]
sys.path.insert(0,str(BENCH/"op22_temporal_fixed_hr_bench"))
import bundle_data_rr
K,N,scen=512,8192,"best"
b=bundle_data_rr.get_bundle(scen,K,torch.float32,N)
lg=b["logits"][0][:N].float(); pre=b["preIdx"][0].long()
print("preIdx range:",int(pre.min()),int(pre.max()),"len",len(pre),"cr",b["cr"],"hr",b["kernel_hit_rate"])
pre=pre[(pre>=0)&(pre<N)]
pv=lg[pre]
print("full-N logits: min/mean/max",float(lg.min()),float(lg.mean()),float(lg.max()))
print("prevK vals   : min/mean/max",float(pv.min()),float(pv.mean()),float(pv.max()),"n",len(pv))
# true top-K boundary value
topk_val=torch.topk(lg,K).values
print("true topK: Kth (min of topK)=",float(topk_val.min()),"  1st=",float(topk_val.max()))
# how many full-N >= various prevK quantiles
for q in [0.85,0.5,0.35,0.15]:
    need=max(1,math.ceil(q*K))
    rung=float(torch.kthvalue(pv, len(pv)-need+1).values)  # need-th largest prevK
    cnt=int((lg>=rung).sum())
    print(f"  q={q} need={need} rung(={need}th-largest-prevK)={rung:.3f}  count(full-N>=rung)={cnt}")
