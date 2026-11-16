import os, sys, time, numpy as np
sys.path.insert(0, "/Users/loncheng/Models/DeepInfer-Mac")
import mlx.core as mx
from deepinfer_mac.store import ExpertStore
root = "/Users/loncheng/Models/dsv41-mac"
st = ExpertStore(root, 2 << 30)
assert st.prio is not None
def check(key, e):
    slab = st.slabs[key]; lay = st.layouts[key]
    slot = slab.lru[(key, e)]
    with open(f"{root}/experts/{key}.bin", "rb") as f:
        f.seek(e * lay.stride); raw = f.read(lay.nbytes)
    got = b"".join(bytes(v) for v in slab.iovecs(slot))
    return got == raw
ok = True
# demand
slab, slots = st.get_slots("L21", [3, 77, 200])
ok &= all(check("L21", e) for e in (3, 77, 200)); print("demand ok", ok)
# speculative then used (promotion)
st.speculate("L22", [10, 11, 12, 13], [])
slab, slots = st.get_slots("L22", [11, 12])
ok &= all(check("L22", e) for e in (11, 12)); print("spec->promote ok", ok)
# speculative then cancelled: flood low queue, cancel before they start
st.speculate("L23", list(range(100, 140)), [])
st.cancel_unneeded("L23", [100])
print("cancelled", st.stats.get("spec_cancelled", 0))
slab, slots = st.get_slots("L23", [100, 101, 5])
ok &= all(check("L23", e) for e in (100, 101, 5)); print("after cancel ok", ok)
# everything still resident must be intact
time.sleep(1.0)
bad = 0
for (k, e), slot in list(st.slabs["L23"].lru.items()):
    fut = st.inflight.get((k, e))
    if fut is None or fut.done():
        if fut is not None and fut.error is not None: continue
        if not check(k, e): bad += 1
print("resident intact:", bad == 0); ok &= bad == 0
print("ALL OK" if ok else "FAIL")
