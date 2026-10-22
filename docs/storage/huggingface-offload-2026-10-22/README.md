# Private results offload — 22 October 2026

Author-authorized upload and local cleanup. No model calls, dataset analyses, Git commits or GitHub pushes.

- Dataset: https://huggingface.co/datasets/HawkFranklin-Research/mlcagent-runs (private).
- Verified snapshot: `d39aee54ed033c1ace8128aefec5a3e86f5f5255`.
- Contents: 194 main-study MLCA runs across 18 sets, plus two AutoML collections; 20 archives.
- Archive size: 591,198,835 bytes; retained source content: 3,306,216,057 bytes in 59,613 files.
- All 138 uploaded files were downloaded as streams and SHA-256 checked. The secret-pattern scan passed; its heuristic limitations are recorded in SECRET_SCAN.json.
- Exploratory/debug/smoke and cancelled reference-analysis sets were not uploaded.

## Local cleanup

After remote verification, removed 144 regenerable `.mlca/data` and `.mlca/env` directories and 12,212 untracked result-file copies. Git-tracked files were preserved. Source input datasets, private fingerprint stores, vaults and keys were preserved. The 92 split-membership CSVs formerly under copied-data folders survive in archive `offload-metadata/` paths, mapped in the per-set manifests.

The first raw cleanup stopped on a read-only step directory after 117 files. The retry checked the remaining file hashes, temporarily granted owner write permission to each affected parent, restored its mode, and removed the remaining 12,095 files. RAW_CLEANUP.json records both already-absent verified files and files removed on retry. The final status is complete.

Experiment-folder cleanup recovered about 1.83 GiB as observed from filesystem free space. Temporary upload archives (another 591 MB) were then removed; manifests and receipts remain locally. Approximately 55.32 GiB was free immediately after cleanup. Free-space readings may also reflect other processes.

The earlier estimate that about 20 GB was regenerable copied inputs/environments was inaccurate: about 15.25 GiB remains in excluded private row-fingerprint stores. Those stores were neither uploaded nor deleted. Deleting them would require rebuilding privacy-check databases from unchanged source inputs before using those projects again. No other cache was cleaned.

## Restore

Use an authenticated Hugging Face account with access to the private dataset. Download into a new directory:

```bash
hf download HawkFranklin-Research/mlcagent-runs --repo-type dataset --revision d39aee54ed033c1ace8128aefec5a3e86f5f5255 --local-dir ./mlcagent-results-backup
cd mlcagent-results-backup
sha256sum -c SHA256SUMS
mkdir ../restored-results
tar --zstd -xf archives/bench__dev-loop-v2__batch-01.tar.zst -C ../restored-results
```

Choose other archives from PACKAGING_INDEX.json. Each archive also contains OFFLOAD_MANIFEST.csv. Archives restore original repository-relative paths; do not unpack over an active experiment. See ARCHIVE_POLICY.md for split-membership relocation and exclusions. Copied inputs/environments and private/vault/key stores are absent from the backup: these are results archives, not standalone runnable project exports. Model/pickle artifacts should only be loaded in an isolated trusted environment.

## Audit files

REMOTE_VERIFIED.json records remote revision and every checked digest; LOCAL_CLEANUP.json records copied-data/environment removal; RAW_CLEANUP.json records untracked result removal; STAGING_CLEANUP.json records temporary archive removal. These receipts and this note are local, uncommitted documentation. The archived support-script snapshot predates the later `prune-raw` helper; the current helper is scripts/offload_results.py.

## Fingerprint stores deleted (23 October 2026, author-authorized)

The author authorized deleting the private row-fingerprint stores after the backup. 195 files (`.mlca/private/row_fingerprints`, `.zst`, `.archive.json`, `.lock`) were removed under `bench/dev-loop/` (batches 01–09, round-02), `bench/runs/` and `bench/validation/`, about 17.9 GiB; free disk went from 55 to 73 GiB. None was tracked by git or uploaded. The other files in those `private/` folders were kept. These projects are finished; reopening one would need its fingerprint database rebuilt from the unchanged source table.
