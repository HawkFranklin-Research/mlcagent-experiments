# eval-02, VM 1 of 3: s2_rna_her2 x3, then s2_proteome_er x2 (laptop eval-02 has proteome_er run 3).
RUN_TAG="eval-02"
COMMON=(--role evaluation --manifest bio-bench/manifest.yaml --data-root data_prep/hf-111de16 --batch 1 --model deepseek-flash --provider deepseek --reasoning-effort low --concurrency 2 --threads 2 --max-output-tokens 32768 --rpm 60)
BATCHES=(
  "bio-bench/runs/deepseek-flash/eval-02-vm1|--tables s2_rna_her2 --replicates 3"
  "bio-bench/runs/deepseek-flash/eval-02-vm1b|--tables s2_proteome_er --replicates 2"
)
