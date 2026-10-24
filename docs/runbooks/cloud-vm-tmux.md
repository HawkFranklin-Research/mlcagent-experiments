# Cloud VM Continuation Runbook: eval-02 in tmux

> **Superseded (24 October 2026)** by [`CLOUD-RUN-MANUAL.md`](CLOUD-RUN-MANUAL.md): parallel Spot VMs, environment preflight, results backup to Hugging Face. Kept as the record of the first single-VM attempt.

This runbook documents the exact procedure for completing the `eval-02` biomedical evaluation batch on a Google Cloud Compute Engine virtual machine inside a persistent `tmux` session.

Once the batch is running inside `tmux` on the VM, the local laptop/workstation can be completely shut down without interrupting the evaluation.

---

## Prerequisites & Pre-flight Checks on Laptop

1. **Verify local test suite and controller cleanliness:**
   ```bash
   cd /home/prime/Documents/github/mlca-paper-codebases/mlcagent-experiments
   PYTHONPATH=scripts:. ../mlca-revized/.venv/bin/pytest tests/
   ../mlca-revized/.venv/bin/ruff check scripts/run_batch.py tests/
   ```

2. **Verify connection failure filter on the paused batch:**
   ```bash
   PYTHONPATH=scripts ../mlca-revized/.venv/bin/python scripts/run_batch.py \
     --out bio-bench/runs/deepseek-flash/eval-02 --list-connection-failures
   ```
   *Expected output (exactly 5 jobs):*
   - `s2_rna_her2-001 (s2_rna_her2 replicate 1)`
   - `s2_rna_her2-002 (s2_rna_her2 replicate 2)`
   - `s2_rna_her2-003 (s2_rna_her2 replicate 3)`
   - `s2_proteome_er-001 (s2_proteome_er replicate 1)`
   - `s2_proteome_er-002 (s2_proteome_er replicate 2)`

3. **Verify DeepSeek key file locally:**
   ```bash
   scripts/with_deepseek_key.sh --check
   ```

---

## Step 1: Provision the Google Cloud VM

Create a standard (non-preemptible, non-Spot) Compute Engine instance:
- **Machine type:** `e2-standard-8` (8 vCPU, 32 GB RAM)
- **Image:** Ubuntu 24.04 LTS
- **Boot disk:** 100 GB Balanced Persistent Disk
- **Zone:** e.g., `us-central1-a` (or preferred active project zone)

```bash
gcloud compute instances create mlca-eval-runner \
  --project=pelliscope-scout \
  --zone=us-central1-a \
  --machine-type=e2-standard-8 \
  --image-family=ubuntu-2404-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-balanced
```

---

## Step 2: Transfer Key File Securely

Transfer the DeepSeek API key file to the VM and set strict permissions (mode 600):

```bash
gcloud compute ssh mlca-eval-runner --zone=us-central1-a --command="mkdir -p ~/Documents"
gcloud compute scp ~/Documents/cheap-seek.api-key.txt mlca-eval-runner:~/Documents/ --zone=us-central1-a
gcloud compute ssh mlca-eval-runner --zone=us-central1-a --command="chmod 600 ~/Documents/cheap-seek.api-key.txt"
```

---

## Step 3: Setup Dependencies on the VM

SSH into the VM:
```bash
gcloud compute ssh mlca-eval-runner --zone=us-central1-a
```

Inside the VM, install system packages and clone the frozen MLCA code:
```bash
sudo apt-get update && sudo apt-get install -y git tmux rsync build-essential python3 python3-venv python3-pip
mkdir -p ~/mlca && cd ~/mlca

# Clone and checkout the frozen tag (must be 7dc0aed / rails-0.15)
git clone git@github.com:HawkFranklin-Research/mlcagent.git mlca-revized
cd mlca-revized
git checkout freeze-v2

# Create virtual environment with identical packages
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
cd ~/mlca
```

---

## Step 4: Transfer Experiments Repository to VM

From the local laptop (in a separate terminal), sync the experiments workspace including data prep, historical outputs, and the paused `eval-02` directory:

```bash
rsync -a --info=progress2 \
  /home/prime/Documents/github/mlca-paper-codebases/mlcagent-experiments/ \
  mlca-eval-runner:~/mlca/mlcagent-experiments/
```

*Note: Nothing is pushed to GitHub or uploaded to Hugging Face.*

---

## Step 5: Launch Continuation Batch in tmux

SSH into the VM:
```bash
gcloud compute ssh mlca-eval-runner --zone=us-central1-a
```

Start a persistent `tmux` session and launch the batch:
```bash
tmux new -s eval02
cd ~/mlca/mlcagent-experiments

# 1. Verify credentials and launcher
scripts/with_deepseek_key.sh --check

# 2. Verify connection failures list on VM
../mlca-revized/.venv/bin/python scripts/run_batch.py \
  --out bio-bench/runs/deepseek-flash/eval-02 \
  --list-connection-failures

# 3. Resume the batch with controller upgrade and retry flag
scripts/with_deepseek_key.sh ../mlca-revized/.venv/bin/python scripts/run_batch.py \
  --role evaluation \
  --manifest bio-bench/manifest.yaml \
  --data-root data_prep/hf-111de16 \
  --out bio-bench/runs/deepseek-flash/eval-02 \
  --batch 1 \
  --model deepseek-flash \
  --provider deepseek \
  --reasoning-effort low \
  --replicates 3 \
  --concurrency 2 \
  --threads 2 \
  --max-output-tokens 32768 \
  --rpm 60 \
  --resume \
  --controller-upgrade "pause, network hold, connection retry; move to cloud VM" \
  --retry-connection-failures
```

Detach from `tmux`:
Press `Ctrl-b`, then release and press `d`.

You can now disconnect SSH and safely shut down the laptop.

---

## Step 6: Monitoring & Clean Pause (Optional)

To check batch progress later:
```bash
# View active execution logs in tmux:
tmux attach -t eval02

# Or inspect the generated Markdown report:
cat ~/mlca/mlcagent-experiments/bio-bench/runs/deepseek-flash/eval-02/REPORT.md
```

If you ever need to pause the batch cleanly on the VM:
```bash
touch ~/mlca/mlcagent-experiments/bio-bench/runs/deepseek-flash/eval-02/.control/PAUSE
# Or run:
../mlca-revized/.venv/bin/python scripts/run_batch.py \
  --out bio-bench/runs/deepseek-flash/eval-02 --pause
```
Running studies will finish and be scored normally before the controller pauses.

---

## Step 7: Retrieval and Teardown (After Batch Completion)

When all 17 remaining studies (12 queued + 5 retried) finish (expected duration: ~8–10 hours):

1. **Sync results back to the local laptop:**
   ```bash
   rsync -a --info=progress2 \
     mlca-eval-runner:~/mlca/mlcagent-experiments/bio-bench/runs/deepseek-flash/eval-02/ \
     /home/prime/Documents/github/mlca-paper-codebases/mlcagent-experiments/bio-bench/runs/deepseek-flash/eval-02/
   ```

2. **Clean up credentials and terminate the VM:**
   ```bash
   gcloud compute ssh mlca-eval-runner --zone=us-central1-a --command="rm -f ~/Documents/cheap-seek.api-key.txt"
   gcloud compute instances delete mlca-eval-runner --zone=us-central1-a --quiet
   ```

3. **Verify completed evidence:**
   - Write `EVALUATION_SUMMARY.md`.
   - Record total success count out of 27.
   - Record token usage and accounting.
   - Commit locally on `master` (no push without authorization).
