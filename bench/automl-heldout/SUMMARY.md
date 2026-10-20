# AutoML comparison

Differences are AutoML minus MLCA; lower RMSE is better. Failures are not assigned zero scores.

## bank_marketing

| Run | Tool | MLCA (source) | AutoML | Status |
|---|---|---|---|---|
| bank_marketing-001/attempt-001 | autogluon | None (mlca_metric_unavailable) | 0.770860850402342 | completed |
| bank_marketing-001/attempt-001 | flaml | None (mlca_metric_unavailable) | 0.7686981551220625 | completed |
| bank_marketing-002/attempt-001 | autogluon | 0.7318327624446923 (shared_scorer) | 0.7886105498801754 | completed |
| bank_marketing-002/attempt-001 | flaml | 0.7318327624446923 (shared_scorer) | 0.7880041839544518 | completed |
| bank_marketing-003/attempt-001 | autogluon | 0.7839601873910182 (shared_scorer) | 0.7886105498801754 | completed |
| bank_marketing-003/attempt-001 | flaml | 0.7839601873910182 (shared_scorer) | 0.7880041839544518 | completed |

autogluon: 3/3 completed; 2 matched scores; matched medians MLCA=0.7578964749178553, AutoML=0.7886105498801754; median paired difference: 0.030714074962320148.
flaml: 3/3 completed; 2 matched scores; matched medians MLCA=0.7578964749178553, AutoML=0.7880041839544518; median paired difference: 0.03010770903659654.

## steel_plates_faults

| Run | Tool | MLCA (source) | AutoML | Status |
|---|---|---|---|---|
| steel_plates_faults-001/attempt-001 | autogluon | None (mlca_metric_unavailable) | 0.9592372869859179 | completed |
| steel_plates_faults-001/attempt-001 | flaml | None (mlca_metric_unavailable) | 0.9532213424101499 | completed |
| steel_plates_faults-002/attempt-001 | autogluon | None (mlca_metric_unavailable) | 0.9616944227448501 | completed |
| steel_plates_faults-002/attempt-001 | flaml | None (mlca_metric_unavailable) | 0.9636004860679518 | completed |
| steel_plates_faults-003/attempt-001 | autogluon | None (mlca_metric_unavailable) | 0.9663680497421742 | completed |
| steel_plates_faults-003/attempt-001 | flaml | None (mlca_metric_unavailable) | 0.9593154227096584 | completed |

autogluon: 3/3 completed; 0 matched scores; matched medians MLCA=unavailable, AutoML=unavailable; median paired difference: unavailable.
flaml: 3/3 completed; 0 matched scores; matched medians MLCA=unavailable, AutoML=unavailable; median paired difference: unavailable.

## wine_quality

| Run | Tool | MLCA (source) | AutoML | Status |
|---|---|---|---|---|
| wine_quality-001/attempt-001 | autogluon | 0.8126686340075636 (shared_scorer) | 0.6907984977386101 | completed |
| wine_quality-001/attempt-001 | flaml | 0.8126686340075636 (shared_scorer) | 0.6925308435160957 | completed |
| wine_quality-002/attempt-001 | autogluon | 0.8126686340075636 (shared_scorer) | 0.6907984977386101 | completed |
| wine_quality-002/attempt-001 | flaml | 0.8126686340075636 (shared_scorer) | 0.6925308435160957 | completed |
| wine_quality-003/attempt-001 | autogluon | None (mlca_metric_unavailable) | 0.6907984977386101 | completed |
| wine_quality-003/attempt-001 | flaml | None (mlca_metric_unavailable) | 0.6925308435160957 | completed |

autogluon: 3/3 completed; 2 matched scores; matched medians MLCA=0.8126686340075636, AutoML=0.6907984977386101; median paired difference: -0.12187013626895349.
flaml: 3/3 completed; 2 matched scores; matched medians MLCA=0.8126686340075636, AutoML=0.6925308435160957; median paired difference: -0.12013779049146789.

