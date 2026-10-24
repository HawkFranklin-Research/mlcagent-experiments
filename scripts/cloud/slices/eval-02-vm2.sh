# eval-02, VM 2 of 3: s2_proteome_pr x3, s2_proteome_her2 x3.
RUN_TAG="eval-02"
COMMON=(--role evaluation --manifest bio-bench/manifest.yaml --data-root data_prep/hf-111de16 --batch 1 --model deepseek-flash --provider deepseek --reasoning-effort low --concurrency 2 --threads 2 --max-output-tokens 32768 --rpm 60)
BATCHES=(
  "bio-bench/runs/deepseek-flash/eval-02-vm2|--tables s2_proteome_pr s2_proteome_her2 --replicates 3"
)
