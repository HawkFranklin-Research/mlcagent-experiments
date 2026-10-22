# Reopening-v2: implementation readiness

22 October 2026. Development preparation, not a freeze or an evaluation result.

## Verified inputs

All 15 development entries are present in the manifest. The three new public
wide tables match both the manifest and SHA256SUMS.txt, with observed sizes:

| ID | Rows | Total columns | SHA-256 |
|---|---:|---:|---|
| gli85_glioma | 85 | 22,284 | 60a74ac6fbfd017c938ef27f4b142e7e92a67c097b7b73bb7a30767b2c77276c |
| smk187_airway | 187 | 19,994 | 09d7404e27a53f6779108505e14ed8478743dd1dd759fc7656f84fbc303d435b |
| gcm_tumour_types | 190 | 16,064 | ad34c7afcd155c1e127baeebda0543616f8478deecc82b7142abd451f67569fd |

These files were already committed in 1c21032; no duplicate table commit was needed.

## Engine changes, locally committed

| Requirement | Commit | Change |
|---|---|---|
| E-W1 | 63597cf | Bounded setup schemas and model-only context views, full local validation retained |
| E-W2 | 65232e5 | Redacted provider error type/message, at most 500 characters each, including deadline transport |
| E-M1 | f7e14b0 | Exact selected-model metric call in the existing scoring-only retry |
| E-S1 | e7bb53f | Retain duration/event fields needed by scoring, while existing predictor guards remain |
| RL-0028 | 7e3a635 | Non-survival validation metric instruction; rails-0.14 |

Rails SHA-256: `e15e1d3fe87c11e6582066c3c2f66b3e3ce65531136c089428dfa4679d1d0cec`.

E-S1 was established from run records: the proposal named duration correctly but
also flagged it as a leakage suspect. Setup protected only its targets list;
preparation consequently deleted duration before the training-only review.
This is fixed as an engine outcome-retention bug, not a survival instruction rail.
No biomedical tables or held-out tables were opened for this diagnosis.

For E-W1, requests from the specified 25,000-feature synthetic fixture remain
under 200,000 bytes, including a real stubbed setup request and its egress digest,
all template-event context/work-order construction and Worker repairs. This is
not a full live wide-table completion claim. Very large user text, generated code
or thousands of mandatory detailed categorical fields can still produce large
requests; required fields and code are not silently removed.

## Validation

- Focused wide-schema/inference/Worker, provider, scoring and survival regressions passed.
- Experiments-controller suite: **125 passed in 21.02 seconds**.
- Ruff on code and tests passed; changed Python files pass formatting checks.
- Full engine suite: **536 passed, 1 skipped in 1291.95 seconds (21m31s)**, exit 0, on source commit `7e3a635`. No live calls.
- Logs: `/tmp/mlca-reopening-v2-pytest.log` and `/tmp/mlca-reopening-controller-pytest.log` (temporary local test output).

## Live-run blocker and exact next step

### Coding agent versus testing agent

The author's correction applies to the testing agent: it must not edit engine
code or rails. The combined work order's Step 2 was an implementation task and
has now been completed by the coding agent. The testing agent should verify
readiness, run the next authorized batch and report the observed failures. Any
consolidated code/rail change required between batches returns to the coding
agent; the testing agent must not make that change itself.

`DEEPSEEK_API_KEY` is absent from the implementing agent's environment. No
credential file was read or used. Exporting it in another existing terminal does
not update the environment of this running agent process. Launch the testing
agent or batch controller from the same shell where it is exported; do not paste
the key into chat or put it into a committed file.

Free disk measured approximately 54 GiB after cleanup, above the earlier 30 GB
gate at inspection time. Re-check at launch; the controller's per-job disk gates
remain authoritative.

Once the full offline gate passes and the environment is available, resume
reopening-v2.md **Step 4**, all 15 development tables in batch-01. Follow its
pooled acceptance and caps before freeze-v2 or eval-02. No acceptance, freeze-v2
tag, new development batch, evaluation rerun or live call has occurred here.
Original eval-01 artifacts and its re-scoring are preserved. Nothing was pushed,
uploaded, or changed in the paper repository. Pre-existing untracked material is
excluded from these commits.
