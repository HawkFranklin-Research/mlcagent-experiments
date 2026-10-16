# Batch-01 review

Development evidence, not frozen evaluation. Baseline code 6827d09dd592412dc15d900501956e1b5d770d92, controller eb70bd39b837adbcf82725409a0f19a87c1cc904, rails rails-0.3. All roles used deepseek/deepseek-v4.1-flash.

## Outcome

0/12 workflows completed; zero accepted successes. Eleven studies ended with retained failures; Covertype was stopped early by the implementing agent for a confirmed engine collection bottleneck. That interruption is not a model failure or a timeout. See covertype-infrastructure-intervention.json.

173 model requests; 378,074 input + 714,702 output tokens; reasoning 664,452 is included in output. Provider-reported cost $0.742783848. No core usage fields are missing. No provider failures were observed.

## Consolidated causes and intended repairs

1. **Setup wire contract:** Adult, Dry Bean and Abalone omitted the optional/defaulted targets field despite naming the outcome in the reason; bounded repair repeated the omission. Successful proposals also defaulted group/time/identifier/leakage fields. Publish and validate an explicit wire proposal while retaining historical project decoding. Never insert manifest answers.
2. **Engine collection performance:** Covertype sandbox audit succeeded in 1.918 s; privacy collection then materialized very large common-value matches. At early stop, RSS was 5,762,836 KiB, wall 1,514 s including roughly twenty-minute cache construction. Optimize matching without changing its verdict/threshold; compare against old semantics in regression tests.
3. **Library and saved-object interfaces:** Heart custom transformers lack set_output; Energy/Anuran saved local __main__ objects cannot be loaded by a subsequent script; Optical double-applies named-column preprocessing after transformations. Add generic importability/feature-contract guidance and value-free diagnostic hints. Do not replace any failed script with hand-written analysis.
4. **Runtime metric diagnostics:** Retail first used a null group key, then supplied the event name eda as a metric partition; the runtime message did not enumerate allowed partitions and was not MLCA-prefixed for safe repair feedback. Keep the rule and make its refusal explicit.
5. **Output budget:** thirteen request errors consumed the 16,384-token cap; many used all tokens for reasoning and emitted no final script. This is not fixed by assuming a shorter visible-script instruction controls hidden reasoning. Official model metadata inspected during the batch advertises high as the default and low as supported. Any explicit effort change must be recorded at a batch boundary; keep model/rails changes separated as required.
6. **Task-type guidance:** One Anuran repair attempted numeric correlation with string outcomes. Limit numeric correlation screens to numeric outcomes and retain categorical derivation checks separately.

## Researcher scoring

Each attempt has separate setup-agreement.json and leakage-score.json under curated/. Missing fits/split evidence is unavailable, never clean. All three leakage criteria are retained independently. No hidden-label predictive scores were computed because no unsupervised workflow completed.

Retail inferred defensible static customer aggregation/clustering, but omitted group/time metadata required by the researcher manifest. The author was asked whether the temporal requirement should remain; generic repairs can proceed while that decision is pending. Manifest expectations have not been changed.

## Evidence and limits

curated/MANIFEST.csv lists source hashes for copied text evidence. Original attempts retain generated model binaries and tables; their hashes are in curated/OMITTED.csv. No private inputs, fingerprint databases, environments, vaults, raw matrices or local notes are committed. No analysis was substituted. Held-out tables and the paper repository remain untouched. No push, publication or freeze.

## Lossless private cache archival after outcomes

Completed-study fingerprint caches larger than 100 MB were compressed locally
with zstd. Each archive was decompressed and checked against the original
uncompressed SHA-256 before reclaiming its redundant original cache copy.
All cache bytes remain recoverable; sources, scripts, event logs, generated
models/tables and vaults are unchanged. The private archives are not committed.
`private-cache-archives.json` records paths, sizes, checksums and restore steps.
Restore the original cache before opening an archived project for privacy reads.
