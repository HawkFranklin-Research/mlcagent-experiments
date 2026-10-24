# AutoML on biomedical tasks, VM 1 (mlca-eval-runner: holds eval-02-vm1, eval-02-vm1b).
RUN_TAG="automl-bio"
OUT="bench/automl-bio"
SPEC="automl/specs/biomedical.yaml"
COMMON=(--time-budget 600 --cpus 2 --seed 42)
LANES=(
  "bio-bench/runs/deepseek-flash/eval-02-vm1|s2_rna_her2"
  "bio-bench/runs/deepseek-flash/eval-02-vm1b|s2_proteome_er"
)
