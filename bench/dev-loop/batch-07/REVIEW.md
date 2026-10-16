# Batch 07 review

Fixed code 5f3d8cb / controller ef3d703 / rails-0.8; low reasoning effort unchanged.
Full offline gate 486 passed, 1 skipped; experiments 18 passed.
Two failed tables rerun: Optical Digits qualifies; Online Retail completes end-to-end but records setup disagreement on time_columns.
87 requests, 250707 input + 202583 output (173463 reported reasoning, not added), $0.219823268624.
Cumulative: 1214 requests, 7499727 tokens, $4.701877847824.
Series: 11/12 tables qualified at least once.

## Optical Digits: complete qualification
- Optical Digits completed all 12 steps to report (12_report_generate_reports_XNF6QB5E).
- Setup agreed: problem_type=representation, targets=[], group=None, time=[].
- Leakage-clean across all three criteria (fitted_columns=True, group_disjoint=True, time_ordered=True).
- Representation validation and test pass cleanly using the explicit 2D bottleneck array and public activation interface from RL-0022 and B-0025.
- 42 requests, 1 repair, 493.57s wall time, $0.074836174 cost.
- Result: **Qualified** (11th of 12 development tables).

## Online Retail: full end-to-end completion, setup disagreement on time_columns
- Online Retail completed all 12 steps end-to-end to report (12_report_generate_reports_12HST9Y3).
- B-0024 and RL-0020 were verified: the entity-building step (03_eda_build_entities) preserved CustomerID and InvoiceDate, the joint splitter partitioned properly, and clustering (baseline and alternative) completed through validation, test, and report.
- Leakage-clean: fitted_columns=True, group_disjoint=True, time_ordered=True.
- Setup agreement:
  - problem_type: clustering (agrees with manifest) -> True
  - targets: [] (agrees with manifest) -> True
  - group_column: CustomerID (agrees with manifest) -> True
  - time_columns: ["InvoiceDate"] vs expected [] -> False
- Cause: Navigator listed InvoiceDate in time_columns during onboard setup because the raw transaction rows contain timestamps and RL-0004 currently states that timestamps used for static features "do not by itself require" chronological splitting, which the model interpreted permissively ("does not require, but preserving chronology is defensible"). The external scorer checks strict list equality (`["InvoiceDate"] == []`), marking time_columns as disagreeing.

## Next steps
- Strengthen RL-0004 to explicitly instruct that for clustering and segmentation of entities from transactional data, time_columns must be left empty (`[]`), as timestamps used only for static feature aggregation (e.g. recency) are not time columns for partitioning.
- Run Batch 08 on Online Retail only to achieve 12/12 qualification under strict metadata agreement.
- Maintain all existing logs, artifacts, and scores without modification.
