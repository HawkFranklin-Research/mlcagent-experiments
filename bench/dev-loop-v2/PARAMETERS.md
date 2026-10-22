# Development parameters, second version (reopening, rule 7)

Author-approved 2026-10-22. Work order: `docs/work-orders/reopening-v2.md`. Development evidence only.

- Starting code: `../mlca-revized` master after the engine fixes and rails-0.14 of the work order. Development tables: the 12 accepted tables plus GLI-85, SMK-CAN-187 and GCM (`benchmarks/manifest.yaml`, commit `1c21032`).
- Navigator and Worker: `deepseek-flash`, direct DeepSeek API (`--provider deepseek`); reasoning effort low; temperature 0; max output 32,768 tokens; 60 requests/minute shared by all processes. The same as the held-out and biomedical runs.
- Concurrency 6; numerical threads 2 per study (16 cores ÷ 6); per-study deadline 5,400 s; model-call deadline 300 s; three Worker attempts; sandbox A.
- Batch-01: all 15 development tables once. Later batches: failed tables only, one consolidated change per batch.
- Success: completed, setup agrees, leakage-clean (three checks).
- Acceptance: pooled 2-of-3 (STUDY_DESIGN §3 item 6). New tables and any table that failed in this series run 3 times on one unchanged version. Old tables that pass batch-01 keep their earlier acceptance.
- Caps: 6 batches or 3 wall-clock days. No ablation.
- Command (batch-01):

```bash
python scripts/run_batch.py --batch 1 --out bench/dev-loop-v2/batch-01 --model deepseek-flash \
  --provider deepseek --reasoning-effort low --concurrency 6 --threads 2 --max-output-tokens 32768 --rpm 60
```
