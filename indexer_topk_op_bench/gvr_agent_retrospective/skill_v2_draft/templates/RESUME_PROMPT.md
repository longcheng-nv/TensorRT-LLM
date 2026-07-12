# RESUME_PROMPT — paste-ready relay (refresh at EVERY commit)

<!-- PASTE-READY PROMPT (copy from here) -->

## 1. Context (1 minute)
Campaign: <name> · objective: <one line> · state: iter <N>, <verdict so far>.
Read PLAN.md + ITERATIONS.md tail + FALSIFIED.md before proposing anything.

## 2. Preflight checklist
- [ ] `git log -1` shows `[iter <N>]` at HEAD <sha>
- [ ] env: <python/torch/toolkit versions or venv path>
- [ ] GPU thermal blacklist: <node:gpu list>; idle >50 °C ⇒ do not time
- [ ] no co-resident driver: check OUTPUT-FILE GROWTH (ps/nvidia-smi are
      namespace-blind), progress markers count = <X>/<Y>
- [ ] anchor cell re-run: <cell> expected <µs> ± 3% (drift > 3% ⇒ re-baseline)

## 3. Work split (if multi-node)
| Node | Shard (env-var selector) | Log name |
|---|---|---|
Shard nodes run and STOP; the coordinating session does parse/update/commit.

## 4. Launch commands (byte-exact; single line; no && chains)
```bash
setsid env <VARS> python3 <driver> > <named>.log 2>&1 &
# stop = pkill -f <driver>; pkill -f <sweeper>; pkill -f "nsys profile";
#        sleep 30; re-check respawn; kill -9 -<pgid> stragglers
```

## 5. Known gotchas
- profilers embed env → `env -u GITHUB_TOKEN -u HF_TOKEN nsys ...`
- <campaign-specific gotchas here>

<!-- END PASTE -->
