# FIXED ablation: measure the COUNT ERROR that 256-bin LINEAR vs LOG binning
# introduces vs the EXACT rank-quantile rung (kthvalue ground truth).
import sys, math
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parents[1]; BENCH=HERE.parents[0]
sys.path.insert(0,str(BENCH/"op22_temporal_fixed_hr_bench"))
import bundle_data_rr
NB=256
def rung_binned(pv,vlo,vhi,edges,need):
    # counts per bin (bin i = [edges[i],edges[i+1])); from-the-top cumulative;
    # rung = LOW edge of the bin where cum-from-top first reaches need.
    b=torch.bucketize(pv,edges[1:-1])          # bin index 0..NB-1
    cnt=torch.bincount(b,minlength=NB).float()
    cum_top=torch.flip(torch.cumsum(torch.flip(cnt,[0]),0),[0])  # >= edges[i]
    idx=torch.nonzero(cum_top>=need)
    i=int(idx[-1]) if len(idx)>0 else 0
    return float(edges[i])
def cell(scen,K,N,qfracs):
    b=bundle_data_rr.get_bundle(scen,K,torch.float32,N)
    lg=b["logits"][0][:N].float().cpu(); pre=b["preIdx"][0].long().cpu(); pre=pre[(pre>=0)&(pre<N)]
    pv=lg[pre]; vlo=float(pv.min()); vhi=float(pv.max())
    lin=torch.linspace(vlo,vhi,NB+1)
    eps=(vhi-vlo)/1e4
    logd=torch.exp(torch.linspace(math.log(eps),math.log(vhi-vlo+eps),NB+1))+vlo-eps
    logd[0]=vlo; logd[-1]=vhi
    out=[]
    for q in qfracs:
        need=max(1,math.ceil(q*K))
        r_ex=float(torch.kthvalue(pv,len(pv)-need+1).values)   # exact need-th largest
        c_ex=int((lg>=r_ex).sum())
        c_lin=int((lg>=rung_binned(pv,vlo,vhi,lin,need)).sum())
        c_log=int((lg>=rung_binned(pv,vlo,vhi,logd,need)).sum())
        out.append((q,c_ex,c_lin,c_log))
    return out
QF=[0.85,0.35,0.15,0.05]
from collections import defaultdict
by_q=defaultdict(lambda:[0,0,0])
print("deep-tail added: q0.15/0.05 = thr0 (count<K, sandwich guaranteed-winners, exponential tail)")
print(f"{'scen':5}{'K':>5}{'N':>7}{'q':>6}{'exact':>7}{'lin':>6}{'log':>6}{'|lin-ex|':>9}{'|log-ex|':>9}")
tot_lin=tot_log=n=0
for K in [512,1024]:
  for N in [8192,16384]:
    for scen in ["best","real","worst"]:
      for q,ce,cl,cg in cell(scen,K,N,QF):
        el=abs(cl-ce); eg=abs(cg-ce); tot_lin+=el; tot_log+=eg; n+=1
        by_q[q][0]+=el; by_q[q][1]+=eg; by_q[q][2]+=1
print(f"{'q(rung)':>8}{'meanExactCnt':>13}{'LIN err':>9}{'LOG err':>9}{'winner':>8}")
for q in QF:
    el,eg,c=by_q[q]; print(f"{q:>8}{'':>13}{el/c:>9.1f}{eg/c:>9.1f}{('LOG' if eg<el else 'LIN'):>8}")
print(f"OVERALL: LINEAR={tot_lin/n:.1f}  LOG={tot_log/n:.1f}")
