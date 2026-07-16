import argparse, os, torch
import torch.cuda.profiler as prof
from torch.utils.cpp_extension import load
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, default=1)
ap.add_argument("--reps", type=int, default=30); a = ap.parse_args()
torch.cuda.set_device(a.gpu)
ext = load(name="floor_probe", sources=[os.path.join(HERE, "../src/floor_probe.cu")],
           extra_cuda_cflags=["-O3","--use_fast_math","-gencode=arch=compute_100,code=sm_100"],
           build_directory=os.environ.get("BUILD_DIR","/tmp/op35_build"), verbose=False)
EVICT = torch.empty(512*1024*1024//4, dtype=torch.float32, device="cuda")
CASES = [(1,131072,148),(1,262144,148),(1,1048576,296),(32,262144,8),(256,262144,1),(1024,65536,1)]
# exactness screen for v10 at both NT
for NT in (512, 1024):
    BS,N,cpr = 4,262144,8
    x = torch.rand(BS,N,device="cuda")+1.0
    K=512
    q = torch.quantile(x[0].float(), torch.tensor([1-2.0*K/N,1-0.8*K/N],device="cuda"))
    t_lo,t_hi = q[0].item(),q[1].item()
    cap = 16*cpr*(NT//32)*((max(4096,int(N*0.08))//(16*cpr))//1)  # generous, segcap = cap/nseg
    nseg = (NT//32)*cpr
    segcap = max(4096//nseg, 512)
    cap = nseg*segcap
    cv = torch.zeros(BS,cap,device="cuda"); ci = torch.zeros(BS,cap,dtype=torch.int32,device="cuda")
    counts = torch.zeros(BS*(2+nseg),dtype=torch.int32,device="cuda")
    tk = torch.zeros(BS,dtype=torch.int32,device="cuda")
    ext.filter_v10(x,t_hi,t_lo,cv,ci,counts,tk,cpr,NT)
    torch.cuda.synchronize()
    C = counts.view(BS,2+nseg); ok=True
    for r in range(BS):
        wc = C[r,2:].clamp(max=segcap)
        gv=[];gi=[]
        for sg in range(nseg):
            n=int(wc[sg]); gv.append(cv[r,sg*segcap:sg*segcap+n]); gi.append(ci[r,sg*segcap:sg*segcap+n])
        gv=torch.cat(gv).sort().values; gi=torch.cat(gi).long().sort().values
        m = x[r]>=t_lo
        if int(C[r,2:].sum())!=int(m.sum()) or not torch.equal(gv,x[r][m].sort().values) \
           or not torch.equal(gi,m.nonzero().flatten().sort().values) or int(C[r,0])!=int((x[r]>=t_hi).sum()):
            ok=False; break
    print(f"v10 NT={NT} exactness: {'OK' if ok else 'FAIL'}")
prof.start()
for BS,N,cpr in CASES:
    x = torch.rand(BS,N,device="cuda")+1.0
    K=512
    q = torch.quantile(x[0].float(), torch.tensor([1-2.0*K/N,1-0.8*K/N],device="cuda"))
    t_lo,t_hi=q[0].item(),q[1].item()
    for NT in (512,1024):
        nseg=(NT//32)*cpr; segcap=max(8192//nseg,64); cap=nseg*segcap
        cv=torch.zeros(BS,cap,device="cuda"); ci=torch.zeros(BS,cap,dtype=torch.int32,device="cuda")
        counts=torch.zeros(BS*(2+nseg),dtype=torch.int32,device="cuda")
        tk=torch.zeros(BS,dtype=torch.int32,device="cuda")
        ext.filter_v10(x,t_hi,t_lo,cv,ci,counts,tk,cpr,NT)
        torch.cuda.synchronize()
        for _ in range(a.reps):
            EVICT.uniform_(); torch.cuda.synchronize()
            ext.filter_v10(x,t_hi,t_lo,cv,ci,counts,tk,cpr,NT)
            torch.cuda.synchronize()
prof.stop()
print("profiled")
