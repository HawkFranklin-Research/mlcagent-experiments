# Held-out tables on the released version (rails 0.15), VM 2 of 2.
RUN_TAG="heldout-02"
COMMON=(--role heldout --batch 2 --model deepseek-flash --provider deepseek --reasoning-effort low --concurrency 2 --threads 2 --max-output-tokens 32768 --rpm 60)
BATCHES=(
  "bench/heldout/heldout-02-vm2|--tables steel_plates_faults wine_quality --replicates 3"
)
