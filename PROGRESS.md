# Progress

- **18 October 2026:** Author approved concurrent, separate acceptance and
  instruction-rails ablation studies. Provider-aware controller, immutable
  36+12 study plans, deterministic sharding and validated pooling implemented;
  verification recorded below. No study, deployment, push or freeze launched.

## Verification

- Full experiments suite: **49 passed in 11.67 s**. This includes synthetic
  process-queue checks for both arms, provider routing, source/runtime identity,
  disjoint shards, resume immutability, pairing, duplicate/incompatible pool
  rejection, retained failures and unavailable accounting.
- Ruff passed for all changed Python files; `git diff --check` passed. All three
  command help screens were checked without creating a plan or launching jobs.
- Optional cgroup code has filesystem-stub and refusal tests. Actual delegated
  kernel containment was not exercised; no real study or live model call ran.
- Native MLCA engine/rails unchanged; its full suite was not rerun for this
  experiments-controller change. Existing untracked runs were preserved.
- No deployment, push, upload, formal freeze, held-out or biomedical access.
- The paper's canonical study design remains untouched; carry the documented
  concurrent-arm amendment into it before executing the comparison.
