# Experiment preparation and execution

Use the code repository's environment; these scripts make no model calls during preparation.

1. Set `SOURCE_DATA` in git-ignored `paths.local.yaml`.
2. Run `python bio-bench/S*/prepare_data.py` for each study. S1 records a source checksum only;
   S2–S4 write deterministic tables and input/output hashes under git-ignored `data_prep/`.
3. `python scripts/run_card.py <card> --out <fresh-directory>` resolves local paths. It writes
   a git-ignored resolved card alongside the run.
4. `mlca bio-bench export <run-output> --to <fresh-export-directory>` curates results.
5. `python scripts/summarize.py <exports-root>` builds cross-arm tables.

The cards and PREREGISTRATION.md are DRAFT, not authorization for measured or live runs. Finalize
scientific placeholders before measured runs. Record both original and resolved task-card hashes.
Source inputs are never modified. Preparation does not fit preprocessing or select predictors.
Within-study repeated splits remain custom instructions; no E3 built-in is introduced.

## External scoring (work order 04)

`score_inference.py` compares recorded setup decisions with researcher metadata.
It reports exact metadata agreement, not proof that other scientific aims are invalid.

`score_unsupervised.py` joins original hidden labels to local per-row outputs by
row identifier after the MLCA run. Clustering uses test ARI/NMI. Representation
uses a fixed external linear probe fitted on training embeddings and labels and
scored on test embeddings. The outputs explicitly identify this as external
scoring, not agent work. Hidden labels must never be registered with MLCA.

Use fresh `--out` paths; the scorers never overwrite a result or launch an agent.
