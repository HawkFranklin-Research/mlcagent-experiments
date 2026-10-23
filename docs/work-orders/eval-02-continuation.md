# Work order: finish eval-02 on a cloud VM (pause, resume, connection-failure retry)

Author request, 23 October 2026. These are batch-controller changes for running at scale. They do not touch the agent: MLCA stays at `freeze-v2` (`7dc0aed`, rails-0.15), and the rails, prompts and engine are unchanged.

## Where eval-02 stands (paused, 15 of 27 finished)

Folder `bio-bench/runs/deepseek-flash/eval-02`, controller commit `f3934c48`, code `7dc0aed`.

| Group | Studies | What happens next |
|---|---|---|
| Successful (10) | s1_anemia 1–3, s2_rna_er 1–3, s2_rna_pr 1–3, s2_proteome_er 3 | kept, not rerun |
| Failed at the first request because the internet was down (5) | s2_rna_her2 1–3, s2_proteome_er 1–2 | rerun once, as a new attempt; the failed attempt-001 stays as a record |
| Queued, never started (12) | s2_proteome_pr 1–3, s2_proteome_her2 1–3, s3_pam50 1–3, s4_hnscc 1–3 | run normally |

The 5 failures each have `category: provider_error`, `furthest_event: setup`, one request with no response, and the message `Setup failed: MlcaUserError: Connection error.` No agent work happened. Only these 5 are retried. Every other failure, now or later, is a result and is not retried.

## Part 1: coding agent (local, no paid model calls)

Edit only `scripts/run_batch.py`, new files under `scripts/`, `tests/` and this repository's docs. **Do not touch `mlca-revized`** (engine, rails, prompts, templates).

1. **Controller upgrade on resume.** Today `--resume` refuses whenever the controller commit changed, and any commit to this repository changes it, including documentation and evidence commits.
   - Add `--controller-upgrade "<reason>"` for use with `--resume`. It is allowed only if `code_commit` and `rails` (version and SHA-256) are identical to the stored state.
   - It appends `{from, to, at, reason, host}` to `state["controller_history"]` and updates `controller_commit`.
   - Without the flag, a changed controller still refuses, as now.
   - From now on, also store `controller_tree` (`git rev-parse HEAD:scripts`). A resume where only files outside `scripts/` changed is accepted without the flag and logged in `controller_history`.

2. **Graceful pause (drain).**
   - `touch <out>/.control/PAUSE`, or `run_batch.py --out <out> --pause`, makes the scheduler stop launching new studies. Running studies finish normally and are scored.
   - The controller then exits with `status: "paused"` and writes the report.
   - `--resume` removes `PAUSE` and records the pause in `state["pauses"]`.
   - **SIGTERM** (VM shutdown) behaves like SIGINT, as today. Ctrl-C keeps its current meaning: stop now.

3. **Hold the queue while the network is down.**
   - Before launching each study, probe the provider endpoint (`state["endpoint"]`) with a TCP or TLS connect. Send no key and make no model call.
   - If the probe fails, do not launch. Retry every 30 s, keep running studies untouched, and log each outage to `state["network_waits"]` with its start, end and seconds.
   - This prevents queued studies from failing in seconds during an outage.

4. **One-time retry of connection failures.**
   - `--list-connection-failures` prints the jobs that match all of the following, and changes nothing:
     - finished with `category == "provider_error"`;
     - `furthest_event == "setup"`;
     - zero model responses with known tokens;
     - the error message contains `Connection error` (or the provider client's connection-error type).
   - `--retry-connection-failures` (only together with `--resume`) re-queues exactly those jobs, once each. For every job it:
     - records `{"retry_of": "attempt-001", "reason": "connection_failure", "at": ...}` in `job["retries"]`;
     - keeps the old attempt folder;
     - resets the study deadline clock for the new attempt only (store the original `first_started` as `first_started_original`). Otherwise the 90-minute study deadline has already expired and the job is closed immediately (see `run_batch.py` around line 1092).
   - Refuse a second connection retry of the same job.
   - The report shows both attempts. The success column uses the last attempt. A note lists the retried jobs and the reason.

5. **Relocatable folder.** On resume, recompute `rate_limit_dir`, `.control` spec paths and config paths from `--out`. The folder can then move to another machine or path, and `batch.json` keeps no stale absolute paths in use.

6. **Per-attempt provenance.** Record `host` (hostname and logical CPUs) and `controller_commit` on every attempt. This shows which studies ran on the laptop and which on the VM.

7. **Key launcher in the repository.**
   - Add `scripts/with_deepseek_key.sh`, equivalent to `~/.local/bin/mlca-with-deepseek-key`.
   - The key file path comes from `MLCA_DEEPSEEK_KEY_FILE` (default `~/Documents/cheap-seek.api-key.txt`).
   - It exports `DEEPSEEK_API_KEY` into the child process only, then execs the command. `--check` verifies the file exists without printing it.
   - Never print the key or put it in arguments or logs.

8. **Tests.** Write them in `tests/test_batch_queue.py` with stub workers and no network or model calls:
   - pause drains, then resume continues;
   - controller upgrade is accepted with matching code and rails, and refused otherwise;
   - outside-`scripts/` commits resume without the flag;
   - the network hold launches nothing while the probe fails;
   - the connection-failure filter selects exactly the 5-shaped records and not mid-study provider errors;
   - a second retry is refused;
   - the deadline reset applies only to the retried attempt;
   - a relocated folder resumes.

   Run the full `tests/` and ruff. Then run `--list-connection-failures` on the real eval-02 folder (read-only); it must print exactly the 5 jobs above.

9. **Runbook.** Write `docs/runbooks/cloud-vm-tmux.md` with the steps in Part 2. Then commit locally on `master` (no push). Report the commit, the test results and the `--list-connection-failures` output.

## Part 2: run on a Google Cloud VM in tmux (after Part 1)

**Author steps (needs your account and key):**
1. Create a standard (not Spot) VM:
   - e2-standard-8 (8 vCPU, 32 GB RAM; the laptop's 14 GB was swapping on the wide tables);
   - Ubuntu 24.04;
   - 100 GB balanced disk.
2. Copy the key: `gcloud compute scp ~/Documents/cheap-seek.api-key.txt VM:~/Documents/`, then `chmod 600` on the VM.

**Setup on the VM:**
```bash
sudo apt-get update && sudo apt-get install -y git tmux rsync build-essential
mkdir -p ~/mlca && cd ~/mlca
git clone git@github.com:HawkFranklin-Research/mlcagent.git mlca-revized   # freeze-v2 is on GitHub
git -C mlca-revized checkout freeze-v2                                     # detached at 7dc0aed; resume requires this exact commit
# create mlca-revized/.venv with Python 3.13 from the repository lock, exactly as on the laptop
```
From the laptop, copy the experiments repository with its data, the paused run and `.git`. Nothing goes to GitHub or Hugging Face:
```bash
rsync -a --info=progress2 /home/prime/Documents/github/mlca-paper-codebases/mlcagent-experiments/ VM:~/mlca/mlcagent-experiments/
```

**Start in tmux:**
```bash
tmux new -s eval02
cd ~/mlca/mlcagent-experiments
scripts/with_deepseek_key.sh --check
../mlca-revized/.venv/bin/python scripts/run_batch.py --out bio-bench/runs/deepseek-flash/eval-02 --list-connection-failures   # expect the 5
scripts/with_deepseek_key.sh ../mlca-revized/.venv/bin/python scripts/run_batch.py \
  --role evaluation --manifest bio-bench/manifest.yaml --data-root data_prep/hf-111de16 \
  --out bio-bench/runs/deepseek-flash/eval-02 --batch 1 --model deepseek-flash --provider deepseek \
  --reasoning-effort low --replicates 3 --concurrency 2 --threads 2 --max-output-tokens 32768 --rpm 60 \
  --resume --controller-upgrade "pause, network hold, connection retry; move to cloud VM" --retry-connection-failures
# detach: Ctrl-b then d. The PC can now be shut down.
```
Check later with `tmux attach -t eval02` or by reading `bio-bench/runs/deepseek-flash/eval-02/REPORT.md`. To pause cleanly, run `touch bio-bench/runs/deepseek-flash/eval-02/.control/PAUSE`.

Rules while it runs: no commits, checkouts or edits of tracked files in either repository on the VM. The settings stay exactly as above.

**After it finishes (17 studies, roughly 8–10 hours):**
1. `rsync` the folder back to the laptop.
2. Delete the key file on the VM, then delete the VM.
3. The testing agent then writes `EVALUATION_SUMMARY.md`:
   - successes out of 27, against the 50% reopening threshold and the 80% the author expects;
   - the 5 connection retries listed separately;
   - token totals.
4. Commit locally. No push without the author's go-ahead.
