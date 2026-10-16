# Batch 03 review

Fixed code 6fe7e90, controller c36c9bf, rails-0.5; DeepSeek V4.1 Flash settings unchanged.
Four of eight studies qualified: Breast Cancer, Adult, Dry Bean and full-size Covertype.
Seven of twelve tables have qualified at least once across batches, not K=3 acceptance.
279 requests, 663898 input + 1152202 output tokens (1076403 reasoning included), $1.1659709834.
Cumulative: 814 requests, 5193710 tokens, $3.3776949354; no unknown core usage.

## Consolidated causes
- Seeds: KMeans/threadpoolctl reads /proc/self/maps; audit hook blocks it. Engine compatibility defect. Author permits exactly this read-only path, not /proc broadly.
- Bike: declared Poisson deviance but regression action requested RMSE, repeating the task-metric contract defect seen with Covertype in batch 02. Missing primary metric led the fallback policy (not Navigator) to loop to preprocessing; final generation truncated. Exact setup time-column agreement also failed and chronological evidence was not clean. Keep the chronological requirement.
- Optical Digits: scoring calls predict on the PCA Pipeline; PCA requires transform/inverse_transform. Generated-code failure plus inadequate interface instructions, not sandbox denial.
- Anuran: unsupported multiclass-multioutput metric followed by truncated repairs. Per-target categorical metrics and averaging need consistent instructions.
- Output truncation remains a distinct output-budget failure. Generation settings stay fixed in batch 04; batch 05 will isolate a reasoning-effort change, based on supported settings.

## Evidence and limits
All original projects and failures retained; curated/ contains hash-verified text evidence, binaries/tables listed by hash only. Closed private fingerprint caches were losslessly archived with byte verification, retaining restoration metadata. No data, models, outputs or logs deleted. Covertype completed full size in 4236 s with four repairs; cache storage remains costly.
Breast and Dry Bean did not select optional prediction audits; Bike did complete its validation audit. Do not attribute those passes to audit execution.

## Prospective author decision after closure
Online Retail static customer segmentation is approved, with recency relative to the last transaction date and no claim of future prediction. Future manifest requires grouping and identifier exclusion but not chronological splitting. Original batch 01/02 scores stay unchanged. Retail was deferred in batch 03.
Next batch: the five remaining tables after consolidated fixes and full offline gate. Held-out/evaluation/paper untouched, no freeze or publication.
