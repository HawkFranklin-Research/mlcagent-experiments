# Running the biomedical evaluation (after the freeze)

1. **Data.** On the run machine, with a Hugging Face **read** token in `HF_TOKEN` (or `--token-file`), run:
   `python scripts/fetch_bio_bench.py`
   This downloads the pinned release (revision `111de16`) to `data_prep/hf-111de16/`, which is never pushed, and verifies every SHA-256. Run it again with `--verify-only` before a batch.
2. **Preregistration (author).** In `bio-bench/manifest.yaml`, settle each task's problem type, targets and leakage columns, and set `approved: true`. Write each starting brief in `bio-bench/briefs/<task id>.md`. Commit `PREREGISTRATION.md`.
3. **Freeze.** Tag the MLCA commit with `freeze-...`. Held-out and evaluation runs refuse untagged code. `--allow-draft` exists only for smoke tests on synthetic data.
4. **Arm A (autonomous).** For example:
   `python scripts/run_batch.py --role evaluation --data-root data_prep/hf-111de16 --out bio-bench/runs/<model>/eval-01 --batch 1 --model <model> --provider deepseek --reasoning-effort low --replicates 3 --concurrency 2 --threads 2 --max-output-tokens 32768`
   - Add `--tables <ids>` to run a subset.
   - The external cohort (S3, S4) is registered sealed: setup never sees it, and the engine opens it only after the test unlock.
   - Scoring, accounting, the disk gate and pruning work as in the development batches.
5. **Arm B (human on the loop)** is interactive: one study at a time in a terminal, where the human can halt and override within the window:
   `mlca run data_prep/hf-111de16/<file> --halt-window 60 --brief "$(cat bio-bench/briefs/<id>.md)" --external data_prep/hf-111de16/<external> --config <config.yaml> --out bio-bench/runs/<model>/armB/<id>-NNN`
   Halts and overrides are recorded in the event log.
6. **Held-out tables** use the same controller with `--role heldout` (data in MLCA's `benchmarks/`).
7. **AutoML:** `automl/README.md`, after the MLCA runs.
