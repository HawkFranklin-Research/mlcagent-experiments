# AutoML on biomedical tasks, VM 3 (mlca-runner-3: holds eval-02-vm3). S4 HNSCC is survival: not compared.
RUN_TAG="automl-bio"
OUT="bench/automl-bio"
SPEC="automl/specs/biomedical.yaml"
COMMON=(--time-budget 600 --cpus 2 --seed 42)
LANES=(
  "bio-bench/runs/deepseek-flash/eval-02-vm3|s3_pam50"
)
