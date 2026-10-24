# Cloud run manual: parallel MLCA batches on several Spot VMs

How to split a batch over several Google Cloud VMs, run them in tmux while the laptop is off, back the results up and pool them on the laptop. Written 24 October 2026, after eval-02. It supersedes `cloud-vm-tmux.md` (single VM, standard instance).

This is operations only. The agent (`mlca-revized` at the freeze tag), the rails, the prompts, the model and every batch setting stay identical on every machine. The split only changes which machine runs which study.

## 1. The idea

- **One VM = one independent batch.** Split the work by table: each VM runs its own `run_batch.py` with its own output folder, local disk, lock and `batch.json`.
- **Never share a disk or a `batch.json` between machines.** The controller's lock is a local file lock.
- Every study is its own MLCA agent with its own DeepSeek calls, so 3 VMs × 2 studies = 6 agents calling DeepSeek in parallel. The `--rpm 60` limit applies per VM; 6 studies make about 6 requests per minute in total.
- Results come back as one compressed archive per batch folder in the private Hugging Face dataset `HawkFranklin-Research/mlcagent-runs`, under `cloud/<RUN_TAG>/<hostname>/`. The laptop downloads, verifies and unpacks them next to the other run folders. Small files are then committed to Git and pushed to GitHub.

## 2. Credentials (files only; never print them or put them in arguments)

| What | Laptop path | Where it goes |
|---|---|---|
| DeepSeek key | `/home/prime/Documents/cheap-seek.api-key.txt` | each VM `~/Documents/cheap-seek.api-key.txt` (mode 600); read by `scripts/with_deepseek_key.sh` |
| Hugging Face write token | `/home/prime/Documents/hugging-face-write-token.txt` | each VM `~/Documents/hugging-face-write-token.txt` (mode 600); read by `scripts/cloud/run_vm.sh` for the upload |
| Hugging Face read token | `/home/prime/Documents/hugging-face-read-token.txt` | laptop only; read by `scripts/cloud/fetch_cloud_runs.sh` |
| GitHub token (HawkFranklin-Research) | `/home/prime/Documents/HawkFranklin-Research-TOKEN-github.txt` | laptop only, and only if the SSH push fails; never on a VM, never in a remote URL |
| Google Cloud | `gcloud auth list` on the laptop | project `pelliscope-scout`, zone `us-central1-a` |

Copy key files with `gcloud compute scp <file> <vm>:~/Documents/ --zone=us-central1-a --project=pelliscope-scout --tunnel-through-iap`, then `chmod 600` them on the VM. Delete them from every VM when the run is over.

## 3. Machines

- **Spot** instances (author's choice, cheap), with `--instance-termination-action=STOP` so a preemption keeps the disk.
- Use **e2-highmem-4** (4 vCPU, 32 GB) for slices with wide RNA tables (HER2/PR/ER RNA, PAM50); e2-standard-4 (16 GB) is enough for proteome-only slices. Two wide RNA studies used about 15 GB on the laptop and swapped.
- `--concurrency 2 --threads 2` on a 4-vCPU VM.
- Build one base VM (section 4), snapshot its disk, and create the others from the snapshot.

```bash
gcloud compute instances create mlca-runner-1 --project=pelliscope-scout --zone=us-central1-a \
  --machine-type=e2-highmem-4 --provisioning-model=SPOT --instance-termination-action=STOP \
  --image-family=ubuntu-2404-lts-amd64 --image-project=ubuntu-os-cloud --boot-disk-size=100GB --boot-disk-type=pd-balanced
```

## 4. Base VM setup (once)

```bash
sudo apt-get update && sudo apt-get install -y git tmux rsync zstd bubblewrap build-essential
curl -LsSf https://astral.sh/uv/install.sh | sh && sudo ln -sf ~/.local/bin/uv /usr/local/bin/uv && sudo ln -sf ~/.local/bin/uvx /usr/local/bin/uvx
mkdir -p ~/mlca ~/Documents && cd ~/mlca
git clone git@github.com:HawkFranklin-Research/mlcagent.git mlca-revized && git -C mlca-revized checkout freeze-v2
(cd mlca-revized && uv sync --frozen --python 3.13)          # creates mlca-revized/.venv (CPython 3.13)
```

From the laptop, send the experiments repository as a Git bundle (about 200 MB; the checkout holds exactly the tracked files, so the working tree is clean) plus the prepared data, which Git ignores. Do not rsync the whole 9 GB folder, and do not exclude tracked folders: the controller refuses to start if tracked files are missing or modified.
```bash
cd /home/prime/Documents/github/mlca-paper-codebases/mlcagent-experiments
git bundle create /tmp/mlcagent-experiments.bundle master
gcloud compute scp /tmp/mlcagent-experiments.bundle <vm>:/tmp/ --zone=us-central1-a --project=pelliscope-scout --tunnel-through-iap
gcloud compute scp --recurse data_prep/hf-111de16 <vm>:/tmp/ --zone=us-central1-a --project=pelliscope-scout --tunnel-through-iap
# on the VM:
git clone -b master /tmp/mlcagent-experiments.bundle ~/mlca/mlcagent-experiments
mkdir -p ~/mlca/mlcagent-experiments/data_prep && mv /tmp/hf-111de16 ~/mlca/mlcagent-experiments/data_prep/
(cd ~/mlca/mlcagent-experiments/data_prep/hf-111de16 && sha256sum -c SHA256SUMS.txt --quiet --ignore-missing)   # the controller also checks each table against the manifest
```
To update a VM that already has the repository, send a new bundle, then run `git -C ~/mlca/mlcagent-experiments pull --ff-only /tmp/mlcagent-experiments.bundle master`.

**Package cache: the step that failed twice in eval-02.** Workers build each study environment with `UV_OFFLINE=1` and the interpreter of `mlca-revized/.venv` (CPython 3.13). The cache must hold cp313 wheels of exactly the reference versions in `bio-bench/env-reference/`. Seed and check it with that interpreter, never with the system `python3`:
```bash
cd ~/mlca/mlcagent-experiments
../mlca-revized/.venv/bin/python scripts/env_preflight.py --seed   # online download of the pinned versions, then the offline check
../mlca-revized/.venv/bin/python scripts/env_preflight.py          # offline check only; must print "matches_reference": true for every set
../mlca-revized/.venv/bin/python -c "from mlca.sandbox.level import detect; print(detect())"   # must print A
```
**Ubuntu 24.04 blocks the sandbox by default.** It restricts unprivileged user namespaces, and every study then fails at its first script with `sandbox_denied` and `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`. In eval-02 this cost 5 studies. `detect()` still prints A, because it only checks that `bwrap` exists. Allow the namespaces, and keep the setting across reboots:
```bash
echo 'kernel.apparmor_restrict_unprivileged_userns=0' | sudo tee /etc/sysctl.d/60-mlca-bwrap.conf
sudo sysctl --system
```
`env_preflight.py`, and therefore the controller before launch, now starts a real bubblewrap sandbox with MLCA's namespaces and refuses to run if it fails.
The controller runs the same offline check before it launches anything. Evaluation batches cannot skip it. If it fails, nothing starts.

Then copy the key files (section 2), run `scripts/with_deepseek_key.sh --check`, stop the VM and snapshot its disk:
```bash
gcloud compute instances stop mlca-runner-1 --zone=us-central1-a --project=pelliscope-scout
gcloud compute snapshots create mlca-base --source-disk=mlca-runner-1 --source-disk-zone=us-central1-a --project=pelliscope-scout
gcloud compute instances create mlca-runner-2 ... --source-snapshot=mlca-base   # same flags as section 3
```
On every clone, run the offline check again before launching.

## 5. Slices

A slice file in `scripts/cloud/slices/` lists the batches of one VM. All slices of one run share the same `COMMON` settings. See `scripts/cloud/slices/eval-02-vm1.sh`:
```bash
RUN_TAG="eval-02"
COMMON=(--role evaluation --manifest bio-bench/manifest.yaml --data-root data_prep/hf-111de16 --batch 1 \
  --model deepseek-flash --provider deepseek --reasoning-effort low --concurrency 2 --threads 2 \
  --max-output-tokens 32768 --rpm 60)
BATCHES=(
  "bio-bench/runs/deepseek-flash/eval-02-vm1|--tables s2_rna_her2 --replicates 3"
  "bio-bench/runs/deepseek-flash/eval-02-vm1b|--tables s2_proteome_er --replicates 2"
)
```
Split by table so that no table is on two VMs, and balance by size: wide RNA tables take about 45–50 minutes per study, small tables about 10. Studies that already have a result elsewhere are not repeated: list only the missing ones, and use `--replicates N` for a partly finished table.

## 6. Launch

Every slice file and script must be committed before launch: the controller refuses untracked or modified scripts. On each VM:
```bash
cd ~/mlca/mlcagent-experiments
tmux new -s run
scripts/cloud/run_vm.sh scripts/cloud/slices/eval-02-vm1.sh
# detach: Ctrl-b d. The laptop can be switched off.
```
`run_vm.sh` does the following:
1. Starts each batch, or resumes it if its `batch.json` exists.
2. After the last batch, or after an operator pause, archives every folder (secrets scan first; `.control/rate` and `.control/config` excluded).
3. Uploads the archives with a `.sha256` to `HawkFranklin-Research/mlcagent-runs` under `cloud/<RUN_TAG>/<hostname>/`.
4. Powers the VM off, so it does not sit idle and cost money. `MLCA_NO_POWEROFF=1` keeps it on.

Within 15 minutes of launch, check that every active study is past `setup` and has EDA steps and requests. A running process alone is not progress.

## 7. Monitoring

```bash
gcloud compute instances list --project=pelliscope-scout
gcloud compute ssh mlca-runner-1 --zone=us-central1-a --project=pelliscope-scout --tunnel-through-iap \
  --command 'cd ~/mlca/mlcagent-experiments; tail -5 cloud-run-$(hostname).log; grep -E "^\| s" bio-bench/runs/deepseek-flash/eval-02-vm1/REPORT.md | cut -d"|" -f2-7'
```
The controller prints `OPERATOR_NEEDED` and pauses its scheduler (running studies finish, no new ones start) when a study ends with:
- an **infrastructure failure**: a connection error at setup, or an environment-build failure;
- or any **provider error** (invalid request, token or context limit, quota, authentication).

`REPORT.md` then has a section "Scheduler paused for the operator". A study that fails inside the agent's own work (any other category) is a **result**: it is recorded and the batch continues. Do not stop a batch because of results.

## 8. When something goes wrong

| Event | What happened | What to do |
|---|---|---|
| Spot preemption | VM stopped; the controller received SIGTERM and marked its running studies `machine_shutdown` | `gcloud compute instances start <vm>`, then in tmux run the same `run_vm.sh` line again. It resumes, re-queues the lost studies as new attempts (old attempts kept, study clock restarted) and continues |
| VM vanished without SIGTERM | resume finds dead studies without results | same as above; they are re-queued as `machine_lost` |
| `OPERATOR_NEEDED`, infrastructure | e.g. the cache is missing wheels, or the network is down | fix the machine, then run `scripts/run_batch.py --out <folder> --list-infrastructure-failures` and resume once with `... --resume --retry-infrastructure-failures`. Each job is retried at most once per reason. Record the decision |
| `OPERATOR_NEEDED`, provider error mid-study | invalid request, token limit, quota | do not retry automatically; report to the author with the message. It may need a code change |
| `sandbox_denied` with `bwrap: loopback …` | the machine blocks the sandbox (see §4) | fix the machine, then `--list-infrastructure-failures` (reason `sandbox_setup_failure`) and resume once with `--retry-infrastructure-failures` |
| Network outage | the controller holds the queue (probes the endpoint every 30 s) and logs `network_waits` | nothing; it continues by itself |

Never delete failed attempts or failed folders; move them aside with a README if they get in the way.

## 9. Pool the results on the laptop

```bash
cd /home/prime/Documents/github/mlca-paper-codebases/mlcagent-experiments
scripts/cloud/fetch_cloud_runs.sh eval-02 bio-bench/runs/deepseek-flash   # read token from ~/Documents/hugging-face-read-token.txt
```
The script downloads `cloud/eval-02/*`, checks each SHA-256, and unpacks each folder into `bio-bench/runs/deepseek-flash/`. It never overwrites an existing folder. Then:
1. `git add` the new folders. `.gitignore` keeps the large raw files out, and those stay on Hugging Face. Check that `git diff --cached --stat` has no parquet, pickle, CSV split or `.control` files, then commit.
2. Push to GitHub (`HawkFranklin-Research/mlcagent-experiments`) over SSH. Only if SSH fails, use the GitHub token file through the environment (`GH_TOKEN="$(cat …)" gh …`), never in a URL or argument.
3. Write the pooled summary (all folders of the run: successes, failures with category, requests, tokens, wall time) and commit it.

## 10. Clean up

1. Delete the key and token files on every VM.
2. Confirm all VMs are `TERMINATED`.
3. After the archives are verified on the laptop, delete the VMs and the snapshot. Stopped VMs still cost disk storage.

## Lessons from eval-02

1. Internet drop on the laptop: 5 studies failed in seconds at setup. Now the queue waits for the network, and the scheduler pauses after an infrastructure failure.
2. VM with an empty uv cache: 17 studies failed at the environment build. Now the controller runs the offline environment check before launching.
3. Cache seeded with the system Python 3.12 instead of the venv's 3.13: 17 more failed after 1–2 model calls each. Now `env_preflight.py --seed` always uses the controller's interpreter and the reference versions.
4. Spot preemption killed running studies. Now SIGTERM marks them `machine_shutdown`, and resume re-runs them as new attempts with a fresh study clock.
5. Idle VMs cost money. Now `run_vm.sh` powers off at the end.
