# AutoML comparison

Differences are AutoML minus MLCA; lower RMSE is better. Failures are not assigned zero scores.

## abalone

| Run | Tool | MLCA (source) | AutoML | Status |
|---|---|---|---|---|
| abalone-001/attempt-001 | autogluon | 2.226 (recorded) | 2.2022342910440176 | completed |
| abalone-001/attempt-001 | flaml | 2.226 (recorded) | 2.2532709041775 | completed |
| abalone-002/attempt-001 | autogluon | 2.226 (recorded) | 2.2022342910440176 | completed |
| abalone-002/attempt-001 | flaml | 2.226 (recorded) | 2.2532709041775 | completed |
| abalone-003/attempt-001 | autogluon | 2.226 (recorded) | 2.2022342910440176 | completed |
| abalone-003/attempt-001 | flaml | 2.226 (recorded) | 2.2532709041775 | completed |

autogluon: 3/3 completed; 3 matched scores; matched medians MLCA=2.226, AutoML=2.2022342910440176; median paired difference: -0.0237657089559824.
flaml: 3/3 completed; 3 matched scores; matched medians MLCA=2.226, AutoML=2.2532709041775; median paired difference: 0.02727090417750011.

## adult

| Run | Tool | MLCA (source) | AutoML | Status |
|---|---|---|---|---|
| adult-001/attempt-001 | autogluon | 0.919324 (recorded) | 0.9308356918499991 | completed |
| adult-001/attempt-001 | flaml | 0.919324 (recorded) | 0.9294613890302923 | completed |
| adult-002/attempt-001 | autogluon | 0.9254 (recorded) | 0.9305075629306657 | completed |
| adult-002/attempt-001 | flaml | 0.9254 (recorded) | 0.927473508762708 | completed |
| adult-003/attempt-001 | autogluon | None (mlca_metric_unavailable) | 0.9305075629306657 | completed |
| adult-003/attempt-001 | flaml | None (mlca_metric_unavailable) | 0.927473508762708 | completed |

autogluon: 3/3 completed; 2 matched scores; matched medians MLCA=0.922362, AutoML=0.9306716273903324; median paired difference: 0.00830962739033242.
flaml: 3/3 completed; 2 matched scores; matched medians MLCA=0.922362, AutoML=0.9284674488965001; median paired difference: 0.00610544889650011.

## bike_sharing_hourly

| Run | Tool | MLCA (source) | AutoML | Status |
|---|---|---|---|---|
| bike_sharing_hourly-001/attempt-001 | autogluon | None (mlca_metric_unavailable) | None | failed |
| bike_sharing_hourly-001/attempt-001 | flaml | None (mlca_metric_unavailable) | None | failed |
| bike_sharing_hourly-002/attempt-001 | autogluon | None (mlca_metric_unavailable) | None | failed |
| bike_sharing_hourly-002/attempt-001 | flaml | None (mlca_metric_unavailable) | None | failed |
| bike_sharing_hourly-003/attempt-001 | autogluon | None (mlca_metric_unavailable) | None | failed |
| bike_sharing_hourly-003/attempt-001 | flaml | None (mlca_metric_unavailable) | None | failed |

autogluon: 0/3 completed; 0 matched scores; matched medians MLCA=unavailable, AutoML=unavailable; median paired difference: unavailable.
flaml: 0/3 completed; 0 matched scores; matched medians MLCA=unavailable, AutoML=unavailable; median paired difference: unavailable.

## breast_cancer_diagnostic

| Run | Tool | MLCA (source) | AutoML | Status |
|---|---|---|---|---|
| breast_cancer_diagnostic-001/attempt-001 | autogluon | 0.9987080103359173 (recorded) | 0.9996770025839794 | completed |
| breast_cancer_diagnostic-001/attempt-001 | flaml | 0.9987080103359173 (recorded) | 0.999031007751938 | completed |
| breast_cancer_diagnostic-002/attempt-001 | autogluon | 0.9987 (recorded) | 0.9996770025839794 | completed |
| breast_cancer_diagnostic-002/attempt-001 | flaml | 0.9987 (recorded) | 0.999031007751938 | completed |
| breast_cancer_diagnostic-003/attempt-001 | autogluon | 0.9987 (recorded) | 0.9996770025839794 | completed |
| breast_cancer_diagnostic-003/attempt-001 | flaml | 0.9987 (recorded) | 0.999031007751938 | completed |

autogluon: 3/3 completed; 3 matched scores; matched medians MLCA=0.9987, AutoML=0.9996770025839794; median paired difference: 0.0009770025839793695.
flaml: 3/3 completed; 3 matched scores; matched medians MLCA=0.9987, AutoML=0.999031007751938; median paired difference: 0.0003310077519379506.

## covertype

| Run | Tool | MLCA (source) | AutoML | Status |
|---|---|---|---|---|
| covertype-001/attempt-001 | autogluon | 0.9479850773732762 (recorded) | 0.981155937532091 | completed |
| covertype-001/attempt-001 | flaml | 0.9479850773732762 (recorded) | 0.9660998615791748 | completed |
| covertype-002/attempt-001 | autogluon | None (mlca_metric_unavailable) | 0.9809753371947194 | completed |
| covertype-002/attempt-001 | flaml | None (mlca_metric_unavailable) | 0.9660998615791748 | completed |

autogluon: 2/2 completed; 1 matched scores; matched medians MLCA=0.9479850773732762, AutoML=0.981155937532091; median paired difference: 0.03317086015881476.
flaml: 2/2 completed; 1 matched scores; matched medians MLCA=0.9479850773732762, AutoML=0.9660998615791748; median paired difference: 0.018114784205898626.

## dry_bean

| Run | Tool | MLCA (source) | AutoML | Status |
|---|---|---|---|---|
| dry_bean-001/attempt-001 | autogluon | 0.9948 (recorded) | 0.9955612989979784 | completed |
| dry_bean-001/attempt-001 | flaml | 0.9948 (recorded) | 0.9954276538442907 | completed |
| dry_bean-002/attempt-001 | autogluon | 0.9947503479557779 (recorded) | 0.9955612989979784 | completed |
| dry_bean-002/attempt-001 | flaml | 0.9947503479557779 (recorded) | 0.9954276538442907 | completed |
| dry_bean-003/attempt-001 | autogluon | 0.9947503479557779 (recorded) | 0.9955612989979784 | completed |
| dry_bean-003/attempt-001 | flaml | 0.9947503479557779 (recorded) | 0.9954276538442907 | completed |

autogluon: 3/3 completed; 3 matched scores; matched medians MLCA=0.9947503479557779, AutoML=0.9955612989979784; median paired difference: 0.000810951042200525.
flaml: 3/3 completed; 3 matched scores; matched medians MLCA=0.9947503479557779, AutoML=0.9954276538442907; median paired difference: 0.0006773058885127803.

## heart_disease_cleveland

| Run | Tool | MLCA (source) | AutoML | Status |
|---|---|---|---|---|
| heart_disease_cleveland-001/attempt-001 | autogluon | 0.8580121703853955 (recorded) | None | failed |
| heart_disease_cleveland-001/attempt-001 | flaml | 0.8580121703853955 (recorded) | None | failed |
| heart_disease_cleveland-002/attempt-001 | autogluon | None (mlca_metric_unavailable) | 0.8833333333333334 | completed |
| heart_disease_cleveland-002/attempt-001 | flaml | None (mlca_metric_unavailable) | 0.8803921568627451 | completed |
| heart_disease_cleveland-003/attempt-001 | autogluon | None (mlca_metric_unavailable) | 0.9208333333333334 | completed |
| heart_disease_cleveland-003/attempt-001 | flaml | None (mlca_metric_unavailable) | 0.9177083333333333 | completed |

autogluon: 2/3 completed; 0 matched scores; matched medians MLCA=unavailable, AutoML=unavailable; median paired difference: unavailable.
flaml: 2/3 completed; 0 matched scores; matched medians MLCA=unavailable, AutoML=unavailable; median paired difference: unavailable.

