# Concurrent acceptance and instruction-rails ablation

Implementation authorized by the author, 18 October 2026. This is controller
support, not an executed experiment, freeze, deployment or scientific result.
The author allows **separate studies to run concurrently**. Every study keeps its
rails setting from its first inference to its last step; no mid-study switching.

## Design and limits

- Acceptance: all 12 development tables, three fresh agent runs each, rails on.
- Ablation: those same 12 tables, one fresh run each, instruction rails off.
- Keep the same harness safeguards, code, dependencies, rail bundle, provider,
  endpoint, model alias, reasoning effort and generation settings across arms.
- The direct DeepSeek route is a new provider condition relative to development
  via OpenRouter. Record it; do not merge the routes as an unchanged-model series.
  Returned aliases may not identify an immutable upstream model revision.
- Both arms may start before acceptance has finished. Acceptance failures are
  retained and reported, not grounds for erasing correctly executed ablation.
  If a candidate changes, a new comparison plan is required. Do not pool versions.
- No spending cap. Existing execution deadlines remain: study 5,400 s, model
  call 300 s, SDK I/O 90 s, script 300 s, three Worker attempts, sandbox A only.
- Two numerical threads and a two-CPU affinity per study by default. This caps
  CPU placement, not a guaranteed CPU-time allocation or the number of arbitrary
  threads created by generated code. Plan creation can choose another count.
- No arbitrary RAM ceiling is imposed. Unset means **not enforced**, with an
  explicit memory-comparability caveat in pooled results. If a RAM ceiling is
  chosen, set it in the shared plan and supply an existing delegated cgroup-v2
  root on each host. The controller refuses an unenforceable ceiling; it does
  not provision cgroups or substitute address-space limits. memory.max applies
  to the whole study process subtree; swap is disabled for that group.
- Machine type, concurrency, available CPUs, RAM, pressure and actual enforcement
  are recorded. Matching thread counts alone does not make unlike hosts or
  workloads comparable for wall time or memory-sensitive failure rates.
- Pacing is **per node**. This is not a distributed/global API limiter. Account
  quota planning and infrastructure setup belong to the execution operator.
- The historical five-day development-loop cap does not stop a new planned
  acceptance/ablation comparison. Per-study deadlines still apply.

## Replicates and pairing

Every job has a unique arm/dataset/replicate study ID. Ablation is designated as
paired with acceptance **replicate 1** of the same table, selected before runs.
All runs retain MLCA's existing protocol seed 42: these are fresh agent repeats,
not three deliberately different-seed experiments. The controller does not force
an inferred task or override agent-chosen splitting. Actual frozen split
parameters and membership hashes are recorded; differing or unavailable splits
are visible in the pool and must not be described as identical partitions.

## Prepare one immutable plan (no model calls)

Use the pinned MLCA environment, with sibling checkouts `mlca-revized` and
`mlcagent-experiments`. Commit authorized controller work locally before creating
a plan: clean tracked trees and no untracked runtime source are required.

```bash
../mlca-revized/.venv/bin/python scripts/plan_comparison.py \
  --development-root bench/dev-loop \
  --provider deepseek --model deepseek-flash --reasoning-effort low \
  --threads 2 --out bench/comparison-plan.json
```

This reads only development metadata, verifies that each development table has
qualified at least once, and records evidence hashes. Prior development may use
another route: its feasibility evidence is not a direct-provider acceptance run.
The plan records all 48 studies, source commits, rails digest, dataset hashes,
evaluation-only exclusions, runtime/lock fingerprints and shared settings.
Only evaluation-only column names go to registration; researcher task answers
are never passed to the agent. Do not regenerate the plan independently per node:
use the exact same plan file and pinned runtime everywhere. A changed controller
commit or runtime requires a different plan.

Set DEEPSEEK_API_KEY in the execution environment through your chosen secret
mechanism. The controller stores only its variable name, never the value. It does
not read credential files, configure accounts or copy secrets between hosts.
OpenRouter remains supported by choosing its provider and appropriate model in
a separate plan; it is not an automatic fallback.

## Separate arms, potentially running at the same time

These commands describe execution **after** environment preparation by the
operator. They have not been launched as part of this implementation.

```bash
# Rails-on process / host
../mlca-revized/.venv/bin/python scripts/run_batch.py \
  --comparison-plan bench/comparison-plan.json --stage acceptance \
  --out bench/comparison/acceptance --concurrency 6

# Separate rails-off process / host; acceptance may still be pending
../mlca-revized/.venv/bin/python scripts/run_batch.py \
  --comparison-plan bench/comparison-plan.json --stage ablation \
  --provisional-ablation --out bench/comparison/ablation --concurrency 6
```

The compatibility name `--provisional-ablation` authorizes early execution, not
conditional deletion of failures. Without it, the legacy sequential acceptance
gate remains. For planned shards use this explicit concurrent path, and pool
all acceptance shards to assess the candidate.

Optional `--shard 1/3`, `2/3`, `3/3` selects disjoint studies within the arm.
Optional `--tables <ids...>` filters planned development tables. Every shard
needs its own fresh output directory. Avoid overlapping selections; pooling
rejects duplicate IDs rather than silently selecting a more successful retry.
At most one table above 500,000 rows runs concurrently **per controller**; a
multi-node operator must also account for simultaneous large tables across nodes.

Resume with `--out <original-folder> --resume`. Stored plan/provider/rail settings
remain authoritative; conflicting flags or edited settings are refused. For a
memory-limited plan, also supply the delegated `--cgroup-root` on every launch
and resume. Finished studies are not repeated. Interrupted attempts are retained.

## Pool without rerunning or moving scientific artifacts

Bring the original result folders into locations visible to the collector, then:

```bash
../mlca-revized/.venv/bin/python scripts/pool_batches.py \
  --plan bench/comparison-plan.json --out bench/comparison/pooled \
  bench/comparison/acceptance bench/comparison/ablation
```

For shards, list all shard folders instead. The collector validates plan identity,
settings, study assignments and available observed enforcement. It writes
`pooled.json` and `REPORT.md`, retaining failures, candidate acceptance outcome,
missing studies, designated pairs and source-file hashes. It recomputes accounting
from request records; absent records yield unavailable totals, not invented zeros.
No private dataset, vault, environment, credential or project tree is copied.
Original folders remain authoritative; the pool is not a standalone replay archive.

This controller uses its own comparison summary. The older `scripts/summarize.py`
expects curated cross-arm bio-bench exports and is not a merger for these batches.
Rails-off measures the contribution of the named instruction rails **within the
same harness**. It is not a bare-LLM comparison or proof that leakage is absent.

## Scope boundary

No Codespaces creation, cloud configuration, machine setup, credentials transfer,
paid inference, upload, push, formal freeze or held-out/biomedical run is part of
this change. The paper's canonical STUDY_DESIGN.md still needs the corresponding
protocol amendment before execution; it was left untouched under the code-only
scope and the MLCA repository's paper-read-only rule.

## Implementation verification

Full experiments suite: 49 passed in 11.67 s; Ruff and whitespace checks passed.
Tests use synthetic process outcomes and filesystem stubs, with no inference.
Actual delegated kernel memory containment and cross-host execution remain
unverified. No study was launched and no existing result was changed.
