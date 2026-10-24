# AutoML on biomedical tasks, VM 2 (mlca-runner-2: holds eval-02-vm2).
RUN_TAG="automl-bio"
OUT="bench/automl-bio"
SPEC="automl/specs/biomedical.yaml"
COMMON=(--time-budget 600 --cpus 2 --seed 42)
LANES=(
  "bio-bench/runs/deepseek-flash/eval-02-vm2|s2_proteome_pr"
  "bio-bench/runs/deepseek-flash/eval-02-vm2|s2_proteome_her2"
)
