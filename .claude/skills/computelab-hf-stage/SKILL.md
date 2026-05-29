---
name: computelab-hf-stage
description: Download HuggingFace model weights directly to local fast storage (`/raid` persistent NVMe or `/dev/shm` RAM) on NVIDIA computelab nodes, bypassing slow NFS scratch cold-reads (~2.5 h for 806 GB) and slow cross-host transfers (often capped at ~1 Gbps mgmt eth when high-speed NICs are DOWN). Uses `hf_transfer` parallel chunked downloads + 16-worker pool to hit ~500 MB/s – 2 GB/s from HF CDN. Provides a single-command staging recipe for any large model (DeepSeek-V4 Pro / Flash / Flash-Base, Llama-405B, Qwen-235B, etc.) after fresh host login, container restart, or host reboot. Wraps `/home/scratch.loncheng_gpu/Deps/download_hf.py`. Trigger keywords: "stage model from HF", "download HuggingFace weights to local NVMe", "fast model loading after reboot", "skip NFS cold copy", "fresh host model weights", "set up large model after login", "post-reboot model staging", "/raid stage from HF", "max-speed weight loading", "computelab HF download".
license: LicenseRef-NvidiaProprietary
metadata:
  author: loncheng
  documentation: https://huggingface.co/docs/huggingface_hub/guides/download
  related: dsv4-pareto-bench, dsv4-gsm8k-eval, trtllm-machine-local-install
---

# Computelab HuggingFace model staging — direct HF Hub → local NVMe

After fresh login on a computelab node (post-reboot, post-container-restart, or first-time on a new host), the fastest way to get a large model onto local fast storage is to **bypass the cluster's slow shared NFS** and pull directly from HuggingFace Hub into `/raid/data/${USER}-stage/` (NVMe, persistent across reboot) or `/dev/shm/` (RAM, ephemeral).

This pattern works on **any computelab node** that has:
1. Local NVMe mounted at `/raid` (or any persistent local fast filesystem)
2. Outbound HTTPS access to `huggingface.co` (or an internal mirror)
3. `huggingface_hub` + `hf_transfer` Python packages (standard in NVIDIA dev containers)

## When to use

- Just logged into a computelab node that doesn't have the model cached locally
- After host reboot — `/dev/shm` is wiped; `/raid` persists, but a new host has neither
- After Docker container restart on a host where `/raid/data/...` bind-mount was just added
- Setting up a new node for the first time
- Cluster's high-speed Mellanox NICs are DOWN (only 1 Gbps mgmt UP) → cross-host rsync also slow
- Want a clean known-revision snapshot of the model rather than the NFS-staged copy of unknown vintage

## Speed comparison (large-model staging on computelab)

| Staging path | Source → Target | Sustained rate | 806 GB ETA | Reboot-persistent |
|---|---|---|---|---|
| **HF Hub → /raid** (this skill) | HF CDN → local NVMe | 500 MB/s – 2 GB/s | **~10–25 min** | ✅ `/raid` |
| NFS cold read → /dev/shm | shared NFS → tmpfs | ~100 MB/s | ~2.5 h | ❌ (RAM) |
| NFS cold read → /raid | shared NFS → NVMe | ~100 MB/s | ~2.5 h | ✅ |
| rsync from peer host's /raid | host-A → host-B mgmt eth | ~113 MB/s (1 Gbps cap) | ~2.0 h | ✅ |

HF Hub direct is the **only** path that bypasses both the shared NFS server and the cluster's potentially slow inter-host network. The exact HF throughput depends on the cluster's outbound bandwidth — see the speed test below.

## Prerequisites (one-time per environment)

```bash
# Tools — in standard loncheng dev container both are pre-installed:
python3 -c "import huggingface_hub, hf_transfer; print(huggingface_hub.__version__)" \
    || pip install --user 'huggingface_hub>=0.20' 'hf_transfer>=0.1'

# Auth — HF_TOKEN already in container env on most nodes. Verify:
echo "${HF_TOKEN:?missing HF_TOKEN — get from https://huggingface.co/settings/tokens}"

# For gated repos (DSv4 / Llama / Mistral often gated): visit the model's HF page
# once with the account whose token you're using and accept the license.
```

## Quick recipe (any large model)

```bash
# 0. Pick your model repo
REPO=deepseek-ai/DeepSeek-V4-Pro        # ~806 GB, 64 safetensors
# REPO=deepseek-ai/DeepSeek-V4-Flash    # ~500 GB
# REPO=meta-llama/Llama-3.1-405B        # ~800 GB
# REPO=Qwen/Qwen3-235B-A22B-Instruct    # ~470 GB
MODEL_NAME=$(basename $REPO)

# 1. Pick target stage path
USER_STAGE=/raid/data/${USER}-stage             # persistent NVMe (recommended)
# USER_STAGE=/dev/shm                            # ephemeral RAM (lost on reboot)
mkdir -p $USER_STAGE
touch $USER_STAGE/.write_test && rm $USER_STAGE/.write_test \
    || { echo "ERROR: $USER_STAGE not writable; try /raid/data or /dev/shm"; exit 1; }

# 2. Set max-speed download env
export HF_HUB_ENABLE_HF_TRANSFER=1               # MANDATORY for full speed
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HOME=$USER_STAGE/hf-home               # keep locks/metadata off NFS
export HF_HUB_CACHE=$USER_STAGE/hf-hub

# 3. Download
python3 /home/scratch.loncheng_gpu/Deps/download_hf.py \
    $REPO \
    $USER_STAGE/$MODEL_NAME \
    --max-workers 16

# 4. Verify
ls $USER_STAGE/$MODEL_NAME/*.safetensors | wc -l    # expect repo-specific shard count
du -sh $USER_STAGE/$MODEL_NAME                       # expect repo-specific size

# 5. Use
export MODEL_PATH=$USER_STAGE/$MODEL_NAME
# now pass MODEL_PATH to trtllm-bench / trtllm-serve / launch_phase{A,B}_*.sh
```

After step 4, this host has the model at NVMe speed forever (until you `rm` it). Future logins skip steps 1-4 — just `export MODEL_PATH=...`.

## Path strategy

### Default: `/raid/data/${USER}-stage/` (persistent NVMe, recommended)

- **Pros**: Survives host reboot AND container restart; ~5 GB/s NVMe read; can be bind-mounted into containers via `-v /raid/data/${USER}-stage:/dsv4-models` (or any target alias).
- **Cons**: Counts against the host's /raid budget; track usage with `du -sh /raid/data/${USER}-stage`.

### Alternative: `/dev/shm/` (ephemeral RAM)

- **Pros**: ~30 GB/s read (RAM-speed); shared with container via `--ipc=host` on standard dev containers (the in-container `/dev/shm` IS the host's `/dev/shm`).
- **Cons**: Wiped on host reboot; competes with host RAM (a 806 GB model consumes ~40 % of a 2 TB host).
- **When**: Latency-sensitive benchmarks where per-cell model load time matters AND a re-stage is cheap (i.e., already have `/raid` fallback).

### Combined: `/raid` → `/dev/shm` per session

```bash
# Persistent /raid stage (this skill, ~15 min one-time)
python3 /home/scratch.loncheng_gpu/Deps/download_hf.py \
    $REPO /raid/data/${USER}-stage/$MODEL_NAME --max-workers 16

# Per-session warm-up: /raid (NVMe) → /dev/shm (RAM), ~3 min for 806 GB via cp -rp
cp -rp /raid/data/${USER}-stage/$MODEL_NAME /dev/shm/
export MODEL_PATH=/dev/shm/$MODEL_NAME
```

## Throughput tuning

| Knob | Default | Aggressive | Notes |
|---|---|---|---|
| `HF_HUB_ENABLE_HF_TRANSFER` | unset | `1` | Rust-based chunked downloader. **MANDATORY** — without it: ~50-100 MB/s; with it: ~500-2000 MB/s. |
| `--max-workers` | 8 | 16–32 | Parallel file workers. Past 16 returns diminish on most CDN paths. |
| `HF_HUB_DOWNLOAD_TIMEOUT` | 10 s | 120 s | First-chunk timeout; raise if HF CDN is slow to start. |
| `HF_ENDPOINT` | huggingface.co | mirror URL | Use NVIDIA-internal mirror if `huggingface.co` is firewalled or slow. |

## Pre-download speed test (~5 sec)

Before kicking off a multi-GB download, sanity-check the path:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "
from huggingface_hub import hf_hub_download
import time, os
os.environ['HF_HOME'] = '/tmp/_hf_probe'
t = time.time()
p = hf_hub_download('deepseek-ai/DeepSeek-V4-Pro', 'config.json', local_dir='/tmp/_hf_probe')
print(f'config.json: {os.path.getsize(p)} B in {time.time()-t:.2f}s')
"
```

If round-trip > 5 s → expect a slower-than-usual transfer (DNS, CDN cold-start, slow outbound). If it fails with 401/403 → check `HF_TOKEN` and gated-repo access.

## Resume from interruption

`snapshot_download` (called via `download_hf.py`) uses `resume_download=True` by default. Re-run the same command — already-complete files (matching size + sha256) are skipped, partial files resume from last byte. Safe to Ctrl-C and restart.

## Verification

```bash
TARGET=$USER_STAGE/$MODEL_NAME

echo "safetensors: $(ls $TARGET/*.safetensors 2>/dev/null | wc -l)"
echo "total size:  $(du -sh $TARGET | awk '{print $1}')"

# Optional: cross-check every file size against HF Hub metadata
TARGET=$TARGET REPO=$REPO python3 - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi()
info = api.repo_info(os.environ['REPO'], files_metadata=True)
target = os.environ['TARGET']
bad = 0
for f in info.siblings:
    local = os.path.join(target, f.rfilename)
    actual = os.path.getsize(local) if os.path.exists(local) else 0
    if actual != f.size:
        print(f'MISMATCH {f.rfilename}: expected {f.size} got {actual}')
        bad += 1
print(f'verify: {bad} mismatches')
PY
```

## Gotchas

### G1 — `HF_HUB_ENABLE_HF_TRANSFER=1` requires the `hf_transfer` package
Without it, downloads fall back to legacy Python HTTP and crawl at ~50-100 MB/s even with parallel workers. Pre-flight:
```bash
python3 -c "import hf_transfer" || pip install --user hf_transfer
```

### G2 — `HF_HOME` on NFS causes lock contention
The default container `HF_HOME=/home/scratch.loncheng_gpu/.cache/huggingface` (or similar) often lives on NFS. Even when `snapshot_download(local_dir=...)` writes file bodies directly to `local_dir`, **lock files and partial-tracking go through `HF_HOME`** — so NFS write latency leaks back in. Override before downloading:
```bash
export HF_HOME=/raid/data/${USER}-stage/hf-home
export HF_HUB_CACHE=/raid/data/${USER}-stage/hf-hub
```

### G3 — gated HF repos
Many large-model repos (DSv4, Llama, Mistral, Qwen) are gated. Visit each HF page once and accept the license with the account whose `$HF_TOKEN` you use. Otherwise download fails with 403.

### G4 — `/raid` permissions vary per host
On observed computelab nodes:
- Some nodes: `/raid` is `drwxrwxrwx` (you can `mkdir /raid/${USER}-stage` directly)
- Other nodes: `/raid` is `drwxr-xr-x root:root` (only root can mkdir at root)

**Always use `/raid/data/${USER}-stage/`** — `/raid/data` is `drwxrwxrwx` on every observed computelab host. Don't hard-code `/raid/${USER}-stage/`.

### G5 — container vs host toolchain
The python toolchain (`huggingface_hub`, `hf_transfer`) is usually installed inside the dev container (via `PYTHONUSERBASE`), NOT on the bare host. SSH'ing directly to a host's shell may lack the tools. If so, run inside the container:
```bash
ssh <host> 'docker exec -u $USER <container-name> bash -c "<your download command>"'
```

### G6 — disk space accounting
A 806 GB model fills ~3 % of a 28 TB `/raid`, so ~30 fully-different models fit. Track:
```bash
du -h --max-depth=2 /raid/data/${USER}-stage
df -h /raid
```

### G7 — HF endpoint / firewall
If outbound to `huggingface.co` is firewalled or slow on your cluster, set:
```bash
export HF_ENDPOINT=https://<your-internal-mirror>
```

### G8 — cross-host network may be SLOWER than HF download
On some computelab clusters the high-speed Mellanox NICs are DOWN and inter-host transfer falls back to 1 Gbps mgmt eth (~113 MB/s). **Re-downloading each new host from HF is usually faster than rsync between hosts**. Don't assume "I'll just rsync from the host that already has it" — measure first.

Probe before staging strategy decisions:
```bash
# 1. Is high-speed NIC up?
ssh <peer> 'ip -br link show | grep -E "^enp.*np.*UP|^ib.*UP"'
# 2. What's the actual cross-host SSH speed?
timeout 10 dd if=/dev/zero bs=1M count=10000 2>/dev/null | \
    ssh <peer> "dd of=/dev/null bs=1M 2>&1 | grep MB/s"
```
If output shows ~113 MB/s → cluster is on mgmt eth only; HF direct is the faster path.

## After-login one-liner (DSv4 Pro example)

For the impatient. Paste verbatim into a fresh login:

```bash
REPO=deepseek-ai/DeepSeek-V4-Pro; MODEL_NAME=$(basename $REPO); \
USER_STAGE=/raid/data/${USER}-stage; mkdir -p $USER_STAGE; \
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DOWNLOAD_TIMEOUT=120 \
       HF_HOME=$USER_STAGE/hf-home HF_HUB_CACHE=$USER_STAGE/hf-hub; \
python3 /home/scratch.loncheng_gpu/Deps/download_hf.py \
    $REPO $USER_STAGE/$MODEL_NAME --max-workers 16 \
&& ls $USER_STAGE/$MODEL_NAME/*.safetensors | wc -l \
&& echo "export MODEL_PATH=$USER_STAGE/$MODEL_NAME"
```

## Example: stage multiple large models in one go

```bash
USER_STAGE=/raid/data/${USER}-stage; mkdir -p $USER_STAGE
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DOWNLOAD_TIMEOUT=120 \
       HF_HOME=$USER_STAGE/hf-home HF_HUB_CACHE=$USER_STAGE/hf-hub

for REPO in \
    deepseek-ai/DeepSeek-V4-Pro \
    deepseek-ai/DeepSeek-V4-Flash \
    Qwen/Qwen3-235B-A22B-Instruct
do
    MODEL_NAME=$(basename $REPO)
    [[ -d $USER_STAGE/$MODEL_NAME ]] && { echo "$MODEL_NAME already staged"; continue; }
    python3 /home/scratch.loncheng_gpu/Deps/download_hf.py \
        $REPO $USER_STAGE/$MODEL_NAME --max-workers 16
done
```

## Integration with other skills

- **`dsv4-pareto-bench`** — `G7 host-reboot recovery` should default to **this skill** instead of NFS recopy.
- **`dsv4-gsm8k-eval`** — same `MODEL_PATH=$USER_STAGE/$MODEL_NAME` works as eval input.
- **`trtllm-machine-local-install`** — orthogonal: that skill handles TRT-LLM wheel install isolation; this skill handles model weight staging.
- **Per-host docker wrapper** — to make `/raid/data/${USER}-stage` visible inside containers, add to the `docker run` wrapper (e.g. `/home/loncheng/bin/launch_nvidia_docker_v1.sh`):
  ```
  --mount type=bind,source=/raid/data/${USER}-stage,target=/models-stage
  ```
  Then launchers can reference `MODEL_PATH=/models-stage/<model-name>` regardless of which underlying host they're on.

## Reference

- Wrapper script: `/home/scratch.loncheng_gpu/Deps/download_hf.py`
- HF docs: https://huggingface.co/docs/huggingface_hub/guides/download
- `hf_transfer`: https://github.com/huggingface/hf_transfer
- Cross-cluster network finding (Mellanox NICs DOWN observation on some computelab nodes): `.perfbot/learnings/20260528T093225-agent.yaml#F001`
