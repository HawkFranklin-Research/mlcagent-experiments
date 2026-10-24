# Held-out tables on the released version (rails 0.15), VM 1 of 2. Earlier result (rails 0.13): bench/heldout/heldout-01.
RUN_TAG="heldout-02"
COMMON=(--role heldout --batch 2 --model deepseek-flash --provider deepseek --reasoning-effort low --concurrency 2 --threads 2 --max-output-tokens 32768 --rpm 60)
BATCHES=(
  "bench/heldout/heldout-02-vm1|--tables bank_marketing wholesale_customers --replicates 3"
)
