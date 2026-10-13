# bio-bench: biomedical studies with MLCA v2

Four studies, ordered from mostly built-in choices to mostly custom instructions. Each `STUDY.md` gives the complete protocol: data and preparation, every event step (built-in choice or custom instruction, with draft wording), decision points, outputs and runs.

| Study | Task | Workflow | Runs today? |
|---|---|---|---|
| `S1_anemia_cbc` | CBC anemia classification | built-in, plus 3–4 custom instructions | yes (as a bench card) |
| `S2_brca_receptors` | ER/PR/HER2 status from RNA vs proteome (6 tasks) | built-in, plus a few custom instructions | yes, after `prepare_data.py` |
| `S3_brca_pam50_transfer` | PAM50 subtype transfer TCGA → CPTAC | built-in multiclass, plus custom harmonisation variants | needs E1 |
| `S4_hnscc_subtype_prognosis` | subtype discovery → prognostic signature → external validation | built-in up to a Cox baseline, then 7 custom instructions | needs E1 (and E2 to run unattended) |

**Ambition:** the protocols aim at every step. Where MLCA cannot achieve a step reliably, that is recorded as *not achieved*, with the reason, and is an acceptable result (see `../AGENTS.md`). Custom instructions are mandatory research artefacts: each one is exported verbatim with its run.

**Engine requirements** (in `../../mlca-revized`, plan P17):
- **E1 external cohort:** a second table sealed until the test unlock, then scored once.
- **E2 `bio-bench` driver:** ordered custom instructions per event; questions and approvals as policy.
- **Optional E3:** within-study repeated splits as a built-in action.

**Per-study layout:** `STUDY.md`, `prepare_data.py`, `task_card.yaml`, `PREREGISTRATION.md`, `runs/<model-slug>/…`, `summary/`. Each run's curated folder adds `custom_instructions.md`.
