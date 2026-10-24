# AutoML comparison

Differences are AutoML minus MLCA; lower RMSE is better. Failures are not assigned zero scores.

## s1_anemia

| Run | Tool | MLCA (source) | AutoML | Status |
|---|---|---|---|---|
| s1_anemia-001/attempt-001 | autogluon | 0.9999999999999999 (shared_scorer) | 1.0 | completed |
| s1_anemia-001/attempt-001 | flaml | 0.9999999999999999 (shared_scorer) | 1.0 | completed |
| s1_anemia-002/attempt-001 | autogluon | 0.5410344827586206 (shared_scorer) | 0.6529310344827586 | completed |
| s1_anemia-002/attempt-001 | flaml | 0.5410344827586206 (shared_scorer) | 0.5825862068965517 | completed |
| s1_anemia-003/attempt-001 | autogluon | 0.9999999999999999 (shared_scorer) | 1.0 | completed |
| s1_anemia-003/attempt-001 | flaml | 0.9999999999999999 (shared_scorer) | 1.0 | completed |

autogluon: 3/3 completed; 3 matched scores; matched medians MLCA=0.9999999999999999, AutoML=1.0; median paired difference: 1.1102230246251565e-16.
flaml: 3/3 completed; 3 matched scores; matched medians MLCA=0.9999999999999999, AutoML=1.0; median paired difference: 1.1102230246251565e-16.

## s2_proteome_er

| Run | Tool | MLCA (source) | AutoML | Status |
|---|---|---|---|---|
| s2_proteome_er-001/attempt-001 | autogluon | 0.9738562091503268 (shared_scorer) | 0.9477124183006536 | completed |
| s2_proteome_er-001/attempt-001 | flaml | 0.9738562091503268 (shared_scorer) | 0.9673202614379085 | completed |
| s2_proteome_er-002/attempt-001 | autogluon | 0.9738562091503268 (shared_scorer) | 0.9477124183006536 | completed |
| s2_proteome_er-002/attempt-001 | flaml | 0.9738562091503268 (shared_scorer) | 0.9673202614379085 | completed |
| s2_proteome_er-003/attempt-001 | autogluon | 0.9738562091503268 (shared_scorer) | 0.9477124183006536 | completed |
| s2_proteome_er-003/attempt-001 | flaml | 0.9738562091503268 (shared_scorer) | 0.9673202614379085 | completed |

autogluon: 3/3 completed; 3 matched scores; matched medians MLCA=0.9738562091503268, AutoML=0.9477124183006536; median paired difference: -0.02614379084967322.
flaml: 3/3 completed; 3 matched scores; matched medians MLCA=0.9738562091503268, AutoML=0.9673202614379085; median paired difference: -0.006535947712418277.

## s2_proteome_her2

| Run | Tool | MLCA (source) | AutoML | Status |
|---|---|---|---|---|
| s2_proteome_her2-001/attempt-001 | autogluon | 1.0 (shared_scorer) | 1.0 | completed |
| s2_proteome_her2-001/attempt-001 | flaml | 1.0 (shared_scorer) | 1.0 | completed |
| s2_proteome_her2-002/attempt-001 | autogluon | 1.0 (shared_scorer) | 1.0 | completed |
| s2_proteome_her2-002/attempt-001 | flaml | 1.0 (shared_scorer) | 1.0 | completed |
| s2_proteome_her2-003/attempt-001 | autogluon | 1.0 (shared_scorer) | 1.0 | completed |
| s2_proteome_her2-003/attempt-001 | flaml | 1.0 (shared_scorer) | 1.0 | completed |

autogluon: 3/3 completed; 3 matched scores; matched medians MLCA=1.0, AutoML=1.0; median paired difference: 0.0.
flaml: 3/3 completed; 3 matched scores; matched medians MLCA=1.0, AutoML=1.0; median paired difference: 0.0.

## s2_proteome_pr

| Run | Tool | MLCA (source) | AutoML | Status |
|---|---|---|---|---|
| s2_proteome_pr-001/attempt-001 | autogluon | None (mlca_metric_unavailable) | None | unavailable |
| s2_proteome_pr-001/attempt-001 | flaml | None (mlca_metric_unavailable) | None | unavailable |
| s2_proteome_pr-001/attempt-003 | autogluon | 0.96 (shared_scorer) | 0.9400000000000001 | completed |
| s2_proteome_pr-001/attempt-003 | flaml | 0.96 (shared_scorer) | 0.9400000000000001 | completed |
| s2_proteome_pr-002/attempt-001 | autogluon | None (mlca_metric_unavailable) | None | unavailable |
| s2_proteome_pr-002/attempt-001 | flaml | None (mlca_metric_unavailable) | None | unavailable |
| s2_proteome_pr-003/attempt-001 | autogluon | 0.9466666666666667 (shared_scorer) | 0.9400000000000001 | completed |
| s2_proteome_pr-003/attempt-001 | flaml | 0.9466666666666667 (shared_scorer) | 0.9400000000000001 | completed |

autogluon: 2/4 completed; 2 matched scores; matched medians MLCA=0.9533333333333334, AutoML=0.9400000000000001; median paired difference: -0.013333333333333253.
flaml: 2/4 completed; 2 matched scores; matched medians MLCA=0.9533333333333334, AutoML=0.9400000000000001; median paired difference: -0.013333333333333253.

## s2_rna_er

| Run | Tool | MLCA (source) | AutoML | Status |
|---|---|---|---|---|
| s2_rna_er-001/attempt-001 | autogluon | 0.954248366013072 (shared_scorer) | 0.9673202614379085 | completed |
| s2_rna_er-001/attempt-001 | flaml | 0.954248366013072 (shared_scorer) | 0.9411764705882353 | completed |
| s2_rna_er-002/attempt-001 | autogluon | 0.954248366013072 (shared_scorer) | 0.9673202614379085 | completed |
| s2_rna_er-002/attempt-001 | flaml | 0.954248366013072 (shared_scorer) | 0.9411764705882353 | completed |
| s2_rna_er-003/attempt-001 | autogluon | 0.9607843137254901 (shared_scorer) | 0.9673202614379085 | completed |
| s2_rna_er-003/attempt-001 | flaml | 0.9607843137254901 (shared_scorer) | 0.9411764705882353 | completed |

autogluon: 3/3 completed; 3 matched scores; matched medians MLCA=0.954248366013072, AutoML=0.9673202614379085; median paired difference: 0.013071895424836555.
flaml: 3/3 completed; 3 matched scores; matched medians MLCA=0.954248366013072, AutoML=0.9411764705882353; median paired difference: -0.013071895424836666.

## s2_rna_her2

| Run | Tool | MLCA (source) | AutoML | Status |
|---|---|---|---|---|
| s2_rna_her2-001/attempt-001 | autogluon | 1.0 (shared_scorer) | 1.0 | completed |
| s2_rna_her2-001/attempt-001 | flaml | 1.0 (shared_scorer) | 1.0 | completed |
| s2_rna_her2-002/attempt-001 | autogluon | 1.0 (shared_scorer) | 1.0 | completed |
| s2_rna_her2-002/attempt-001 | flaml | 1.0 (shared_scorer) | 1.0 | completed |
| s2_rna_her2-003/attempt-001 | autogluon | 1.0 (shared_scorer) | 1.0 | completed |
| s2_rna_her2-003/attempt-001 | flaml | 1.0 (shared_scorer) | 1.0 | completed |

autogluon: 3/3 completed; 3 matched scores; matched medians MLCA=1.0, AutoML=1.0; median paired difference: 0.0.
flaml: 3/3 completed; 3 matched scores; matched medians MLCA=1.0, AutoML=1.0; median paired difference: 0.0.

## s2_rna_pr

| Run | Tool | MLCA (source) | AutoML | Status |
|---|---|---|---|---|
| s2_rna_pr-001/attempt-001 | autogluon | 0.96 (shared_scorer) | 0.9866666666666667 | completed |
| s2_rna_pr-001/attempt-001 | flaml | 0.96 (shared_scorer) | 0.8366666666666667 | completed |
| s2_rna_pr-002/attempt-001 | autogluon | 0.96 (shared_scorer) | 0.9866666666666667 | completed |
| s2_rna_pr-002/attempt-001 | flaml | 0.96 (shared_scorer) | 0.8366666666666667 | completed |
| s2_rna_pr-003/attempt-001 | autogluon | 0.9666666666666667 (shared_scorer) | 0.9866666666666667 | completed |
| s2_rna_pr-003/attempt-001 | flaml | 0.9666666666666667 (shared_scorer) | 0.8366666666666667 | completed |

autogluon: 3/3 completed; 3 matched scores; matched medians MLCA=0.96, AutoML=0.9866666666666667; median paired difference: 0.026666666666666727.
flaml: 3/3 completed; 3 matched scores; matched medians MLCA=0.96, AutoML=0.8366666666666667; median paired difference: -0.1233333333333333.

## s3_pam50

| Run | Tool | MLCA (source) | AutoML | Status |
|---|---|---|---|---|
| s3_pam50-001/attempt-001 | autogluon | None (mlca_metric_unavailable) | None | unavailable |
| s3_pam50-001/attempt-001 | flaml | None (mlca_metric_unavailable) | None | unavailable |
| s3_pam50-001/attempt-003 | autogluon | 0.9736007588387112 (shared_scorer) | 0.9781711253573988 | completed |
| s3_pam50-001/attempt-003 | flaml | 0.9736007588387112 (shared_scorer) | 0.9722844468253811 | completed |
| s3_pam50-002/attempt-001 | autogluon | None (mlca_metric_unavailable) | None | unavailable |
| s3_pam50-002/attempt-001 | flaml | None (mlca_metric_unavailable) | None | unavailable |
| s3_pam50-002/attempt-003 | autogluon | 0.9736007588387112 (shared_scorer) | 0.9781711253573988 | completed |
| s3_pam50-002/attempt-003 | flaml | 0.9736007588387112 (shared_scorer) | 0.9722844468253811 | completed |
| s3_pam50-003/attempt-001 | autogluon | 0.9736007588387112 (shared_scorer) | 0.9781711253573988 | completed |
| s3_pam50-003/attempt-001 | flaml | 0.9736007588387112 (shared_scorer) | 0.9722844468253811 | completed |

autogluon: 3/5 completed; 3 matched scores; matched medians MLCA=0.9736007588387112, AutoML=0.9781711253573988; median paired difference: 0.004570366518687585.
flaml: 3/5 completed; 3 matched scores; matched medians MLCA=0.9736007588387112, AutoML=0.9722844468253811; median paired difference: -0.0013163120133301565.

