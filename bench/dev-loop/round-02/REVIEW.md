# Round 2 closure — work order 05 Step 0

**Superseded by author decision; not resumed or completed.** At Step 0 the controller process was already absent. The persisted round had three finished failed attempts and one interrupted attempt. Its raw round.json has no end timestamp; the cause of the earlier process exit is unavailable. No SIGINT was necessary. Original logs and outcomes are unchanged.

| Table | Status at closure | Furthest event | Completed steps | Recorded category |
|---|---|---|---:|---|
| breast_cancer_diagnostic | failed | model | 6 | runtime_error:ValueError |
| heart_disease_cleveland | failed | model | 6 | runtime_error:ValueError |
| adult | not started | not started | 0 | not applicable |
| dry_bean | not started | not started | 0 | not applicable |
| covertype | not started | not started | 0 | not applicable |
| anuran_calls | not started | not started | 0 | not applicable |
| abalone | failed | preprocess | 4 | output_truncated |
| bike_sharing_hourly | not started | not started | 0 | not applicable |
| energy_efficiency | interrupted | eda | 2 | interrupted (no terminal result) |
| seeds | not started | not started | 0 | not applicable |
| online_retail | not started | not started | 0 | not applicable |
| optical_digits | not started | not started | 0 | not applicable |

## Brief cause review

- Breast cancer and Heart stopped in model after saved-pipeline chaining failures; the proposed rails address named DataFrames and avoiding repeated preprocessing.
- Abalone stopped in preprocess with output_truncated; the retained EDA history also contains a mixed-type summary failure.
- Energy Efficiency stopped during EDA with no final attempt outcome. Pending usage is unavailable, not zero.
- These are exploratory development attempts, not measured evaluation or successful workflows. Adult conclusions from Round 1 require the documented label-spelling qualification.
- New work uses batch-01 onward. No further free-router requests will be started.
