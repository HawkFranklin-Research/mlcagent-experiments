# WP5 material acceptance — 14 October 2026

Preparation scripts and draft task-card variants are written for S1–S4 and all six benchmark
inputs. There are 65 cards including the existing smoke card. All validate with TaskCard and
Config; declared instruction events exist in their templates. Three material tests pass; Ruff
passes. S2 counts reproduce ER 81/39, PR 68/47 and HER2 15/107 for both modalities. S4 joins
reproduce 415/102. S3 writes 981 TCGA and 122 CPTAC rows. Repeated preparation retains identical
input and table hashes. No AI, fitted preprocessing or original-input changes occurred.

Prepared tables and local path configuration remain git-ignored. Preregistration is DRAFT.
Scientific choices and source citations listed in STUDY.md remain author items before measured
runs; placeholder instructions must not be interpreted as permission for model-chosen methods.
No live model calls, push, remote creation, upload or publication.

## WP7 offline acceptance finished

Seven reference development attempts recorded zero model requests. S1 core completed with
four custom instructions not achieved; six S2 attempts failed (four interpretation timeouts,
two HER2 bootstrap NaN errors). Their exports passed the manifest hash/exclusion audit.
Experiments suite: 7 passed; code suite: 296 passed, 1 skipped. No measured AI run or freeze.
See [acceptance report](bio-bench/acceptance/2026-10-14/README.md) and the timestamped
[interim report](INTERIM_REPORT.md). Draft scientific choices remain outstanding.
