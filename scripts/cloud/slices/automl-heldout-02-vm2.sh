# AutoML on the new held-out splits (released version), VM 2.
RUN_TAG="automl-heldout-02"
OUT="bench/automl-heldout-02"
SPEC="automl/specs/heldout.yaml"
COMMON=(--time-budget 600 --cpus 2 --seed 42)
LANES=(
  "bench/heldout/heldout-02-vm2|steel_plates_faults"
  "bench/heldout/heldout-02-vm2|wine_quality"
)
