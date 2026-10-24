# AutoML comparison

Differences are AutoML minus MLCA; lower RMSE is better. Failures are not assigned zero scores.

## bank_marketing

| Run | Tool | MLCA (source) | AutoML | Status |
|---|---|---|---|---|
| bank_marketing-001/attempt-001 | autogluon | 0.7771353629887958 (shared_scorer) | 0.7886105498801754 | completed |
| bank_marketing-001/attempt-001 | flaml | 0.7771353629887958 (shared_scorer) | 0.7880041839544518 | completed |
| bank_marketing-002/attempt-001 | autogluon | 0.7771353629887958 (shared_scorer) | 0.7886105498801754 | completed |
| bank_marketing-002/attempt-001 | flaml | 0.7771353629887958 (shared_scorer) | 0.7880041839544518 | completed |
| bank_marketing-003/attempt-001 | autogluon | 0.7771353629887958 (shared_scorer) | 0.7886105498801754 | completed |
| bank_marketing-003/attempt-001 | flaml | 0.7771353629887958 (shared_scorer) | 0.7880041839544518 | completed |

autogluon: 3/3 completed; 3 matched scores; matched medians MLCA=0.7771353629887958, AutoML=0.7886105498801754; median paired difference: 0.011475186891379607.
flaml: 3/3 completed; 3 matched scores; matched medians MLCA=0.7771353629887958, AutoML=0.7880041839544518; median paired difference: 0.010868820965656.

## steel_plates_faults

| Run | Tool | MLCA (source) | AutoML | Status |
|---|---|---|---|---|
| steel_plates_faults-001/attempt-001 | autogluon | None (mlca_metric_unavailable) | 0.9640223861668896 | completed |
| steel_plates_faults-001/attempt-001 | flaml | None (mlca_metric_unavailable) | 0.9526755347041737 | completed |
| steel_plates_faults-002/attempt-001 | autogluon | None (mlca_metric_unavailable) | None | unavailable |
| steel_plates_faults-002/attempt-001 | flaml | None (mlca_metric_unavailable) | None | unavailable |
| steel_plates_faults-003/attempt-001 | autogluon | None (mlca_metric_unavailable) | 0.9663680497421742 | completed |
| steel_plates_faults-003/attempt-001 | flaml | None (mlca_metric_unavailable) | 0.9594334437159727 | completed |

autogluon: 2/3 completed; 0 matched scores; matched medians MLCA=unavailable, AutoML=unavailable; median paired difference: unavailable.
flaml: 2/3 completed; 0 matched scores; matched medians MLCA=unavailable, AutoML=unavailable; median paired difference: unavailable.

## wine_quality

| Run | Tool | MLCA (source) | AutoML | Status |
|---|---|---|---|---|
| wine_quality-001/attempt-001 | autogluon | 0.8126686340075636 (shared_scorer) | 0.6907984977386101 | completed |
| wine_quality-001/attempt-001 | flaml | 0.8126686340075636 (shared_scorer) | 0.6925308435160957 | completed |
| wine_quality-002/attempt-001 | autogluon | 0.8126686340075636 (shared_scorer) | 0.6907984977386101 | completed |
| wine_quality-002/attempt-001 | flaml | 0.8126686340075636 (shared_scorer) | 0.6925308435160957 | completed |
| wine_quality-003/attempt-001 | autogluon | 0.8149675597714843 (shared_scorer) | 0.6907984977386101 | completed |
| wine_quality-003/attempt-001 | flaml | 0.8149675597714843 (shared_scorer) | 0.6925308435160957 | completed |

autogluon: 3/3 completed; 3 matched scores; matched medians MLCA=0.8126686340075636, AutoML=0.6907984977386101; median paired difference: -0.12187013626895349.
flaml: 3/3 completed; 3 matched scores; matched medians MLCA=0.8126686340075636, AutoML=0.6925308435160957; median paired difference: -0.12013779049146789.

