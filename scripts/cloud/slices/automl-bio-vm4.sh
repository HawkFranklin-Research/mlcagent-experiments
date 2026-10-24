# AutoML on biomedical tasks, VM 4 (new, from the base snapshot; needs the laptop's eval-02 folder copied in).
RUN_TAG="automl-bio"
OUT="bench/automl-bio"
SPEC="automl/specs/biomedical.yaml"
COMMON=(--time-budget 600 --cpus 2 --seed 42)
LANES=(
  "bio-bench/runs/deepseek-flash/eval-02|s1_anemia s2_rna_er"
  "bio-bench/runs/deepseek-flash/eval-02|s2_rna_pr s2_proteome_er"
)
