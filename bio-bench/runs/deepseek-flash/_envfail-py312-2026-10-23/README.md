# Failed eval-02 Attempts (Python 3.12 Wheels Environment Build Failure)

- **Date:** 2026-10-23
- **Context:** Parallel `eval-02` continuation attempt across 3 Spot VMs in `pelliscope-scout` (`us-central1-a`).
- **Failure Cause:** 
  The offline `uv` wheel cache on the VMs had been seeded using `/usr/bin/python3` (Python 3.12, generating `cp312` ABI tags). However, the batch controller and study environments execute under `mlca-revized/.venv/bin/python` (Python 3.13.17, requiring `cp313` ABI tags). When the worker attempted offline environment creation (`UV_OFFLINE=1`), `uv` could not locate `cp313` wheels in the local cache and aborted after initial setup calls.
- **Affected Slices & Studies:** 17 studies across 4 batch folders (`eval-02-vm1`, `eval-02-vm1b`, `eval-02-vm2`, `eval-02-vm3`).
- **Preservation Policy:** All 17 failed attempts are preserved verbatim for auditing and provenance. Nothing was overwritten or deleted.

## Resource & Token Usage Summary

| Batch Folder | Failed Studies | API Requests | Input Tokens | Output Tokens | Cached Tokens | Total Tokens |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `eval-02-vm1` | 3 (`s2_rna_her2` 1–3) | 4 | 1,642,390 | 1,999 | 1,639,552 | 1,644,389 |
| `eval-02-vm1b` | 2 (`s2_proteome_er` 1–2) | 2 | 469,894 | 875 | 469,504 | 470,769 |
| `eval-02-vm2` | 6 (`s2_proteome_pr` 1–3, `s2_proteome_her2` 1–3) | 8 | 1,877,797 | 4,104 | 1,872,256 | 1,881,901 |
| `eval-02-vm3` | 6 (`s3_pam50` 1–3, `s4_hnscc` 1–3) | 8 | 310,248 | 4,796 | 304,768 | 315,044 |
| **Total** | **17 studies** | **22** | **4,300,329** | **11,774** | **4,286,080** | **4,312,103** |

*(Reasoning tokens included in output tokens: 4,429)*
