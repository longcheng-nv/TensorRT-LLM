# gvr-topk-cold60

Scaffolded by `kf campaign init`. These files describe the campaign:

| File | What it is |
|---|---|
| `campaign.yaml`   | Run-time settings used by `campaign prepare` and `campaign start --from`. Edit first. |
| `definition.json` | Problem spec — tensor shapes, dtypes, axes, and the PyTorch reference. |
| `workload.jsonl`  | One JSON object per line — the workloads your kernel must beat. |
| `knowledge.yaml`  | Skills + contexts the agent pods load. Live snapshot from your cluster. |
| `README.md`       | This file. |

## Quick start

1. Edit `definition.json` — replace placeholder shapes / dtypes / `reference` with the operation you want optimised.
2. Edit `workload.jsonl` — one line per workload; set `axes`, `inputs`, and any workload-specific data.
3. (Optional) Tune `campaign.yaml` — adjust `effort`, `max_rounds`, `agent_pool`, `gpu_spec`, or baseline inputs.
4. (Optional) Edit `knowledge.yaml` — delete lines for skills / contexts you don't need.
5. Prepare baselines and stamp the inputs:

   ```shell
   kf campaign prepare --from campaign.yaml
   ```

6. Launch:

   ```shell
   kf campaign start --from campaign.yaml --watch
   ```

   `--watch` streams progress and exits when the campaign reaches a terminal phase. Drop it to fire-and-forget.

## Configured defaults

| Field      | Value           |
|---|---|
| Language   | `cuda_cpp_only`  |
Re-running `kf campaign init gvr-topk-cold60` refuses to overwrite this directory. Pass `--force` to scaffold over it.
