# HOST ablation (rung-0/1): does exponential/log binning of the 256-hist place
# the rank-quantile rungs more precisely than the current LINEAR binning?
# Metrics per (scen,K,N): admission (any rung count in [K,kC]?), count error vs
# target, sandwich band width (rung with count<K to rung with count>=K).
# Binnings: LINEAR (current), LOG (geometric, dense near v_lo), QUANTILE (ideal,
# data-adaptive upper bound). No kernel — pure torch on the synth bundles.
import sys, math
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parents[1]; BENCH=HERE.parents[0]
sys.path.insert(0,str(BENCH/"op22_temporal_fixed_hr_bench"))
import bundle_data_rr
NB=256
def kC_of(K):  # candidate cap (stock): min(5*K,5120) region; use 5120 floor per op26
    return min(5*K, 6144) if K<=1024 else 6144
def rungs_from_hist(edges, prevk_vals, qneeds):
    # edges: NB+1 ascending bin edges. count prevk per bin, then from-the-top
    # cumulative; rung m = edge where cum-from-top first reaches qneeds[m].
    counts=torch.histogram(prevk_vals.cpu(), bins=edges.cpu()).hist  # NB
    cum_top=torch.flip(torch.cumsum(torch.flip(counts,[0]),0),[0])  # cum(>= edge_i low)
    # cum_top[i] = number of prevk >= edges[i]
    out=[]
    for need in qneeds:
        # smallest i (highest value) with cum_top[i] >= need -> rung = edges[i]
        idx=torch.nonzero(cum_top>=need)
        out.append(float(edges[int(idx[0])]) if len(idx)>0 else float(edges[0]))
    return out
def eval_cell(scen,K,N):
    b=bundle_data_rr.get_bundle(scen,K,torch.float32,N)
    lg=b["logits"][0][:N].float(); pre=b["preIdx"][0].long()
    pre=pre[(pre>=0)&(pre<N)]
    pv=lg[pre]                       # prev-topK gathered values
    vlo=float(pv.min()); vhi=float(pv.max())
    kC=kC_of(K)
    qfracs=[0.85,0.35]; qneeds=[max(1,math.ceil(q*K)) for q in qfracs]
    res={}
    # LINEAR
    lin=torch.linspace(vlo,vhi,NB+1)
    # LOG (geometric, dense near vlo): shift to positive, log-space
    eps=(vhi-vlo)/1e4
    lg_e=torch.exp(torch.linspace(math.log(eps),math.log(vhi-vlo+eps),NB+1))+vlo-eps
    lg_e[0]=vlo; lg_e[-1]=vhi
    # QUANTILE (ideal, data-adaptive)
    qe=torch.quantile(pv.cpu(), torch.linspace(0,1,NB+1))
    for name,edges in [("linear",lin),("log",lg_e),("quantile",qe)]:
        rungs=rungs_from_hist(edges,pv,qneeds)
        cnts=[int((lg>=r).sum()) for r in rungs]
        adm=any(K<=c<=kC for c in cnts)
        # best count error (closest rung to [K,kC] midpoint target sqrt(K*kC))
        tgt=math.sqrt(K*kC)
        err=min(abs(c-tgt)/tgt for c in cnts)
        # sandwich: need a rung count<K (thr0) and one count>=K (thr1); band=M1-M0
        below=[c for c in cnts if c<K]; above=[c for c in cnts if c>=K]
        band=(min(above)-max(below)) if (below and above) else None
        res[name]=(adm,cnts,round(err,3),band)
    return res
print(f"{'scen':5}{'K':>5}{'N':>7} | {'LINEAR adm/err/band':>26} | {'LOG':>22} | {'QUANTILE':>22}")
for K in [512,1024]:
  for N in [8192,16384]:
    for scen in ["best","real","worst"]:
        r=eval_cell(scen,K,N)
        def fmt(t): a,c,e,bd=t; return f"{'Y' if a else 'n'} err{e} band{bd}"
        print(f"{scen:5}{K:>5}{N:>7} | {fmt(r['linear']):>26} | {fmt(r['log']):>22} | {fmt(r['quantile']):>22}")
