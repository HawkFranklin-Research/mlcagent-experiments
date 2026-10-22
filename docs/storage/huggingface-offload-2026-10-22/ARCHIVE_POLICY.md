# Archive and restoration policy

Private storage snapshot authorized by the author. Contains the 194 main-study
MLCA jobs across 18 sets, plus two AutoML result collections. Exploratory rounds,
smoke/debugging folders, the cancelled reference analyses, source input tables,
private/vault/key stores, Python environments and controller control files are
excluded. No dataset was reanalysed and no model was called for this backup.

Each .tar.zst contains original result paths and OFFLOAD_MANIFEST.csv with every
member's source path, size and SHA-256. Matching CSVs are under manifests/.
PACKAGING_INDEX.json records each set's source and archive identity. Some earlier
sets also contain curated copies: they are retained, not counted as extra runs.
Derived per-row outputs and fitted models are raw results and may be included;
this is a private archive, not an anonymized public dataset or a confidentiality
certification. Do not load untrusted pickle/model files without an isolated environment.

Split-membership CSV metadata originally under .mlca/data is preserved instead
under offload-metadata/<original path>. Other .mlca/data files and .mlca/env are
regenerable and not archived. Inspect omissions/ for excluded paths. Input sources
and checksums are documented in support/. The biomedical input release remains
HawkFranklin-Research/mlcagent-bio-bench at its pinned revision; it is not duplicated.
These archives alone do not make a runnable native project: input partitions,
environments and any excluded vault material must be reconstructed separately.
Hash-chained logs remain byte-identical; their original artifact references are
not rewritten when excluded inputs are absent.

To inspect: zstd -t ARCHIVE.tar.zst; tar --zstd -tf ARCHIVE.tar.zst
To restore: verify SHA256SUMS, then tar --zstd -xf ARCHIVE.tar.zst -C EMPTY_DIRECTORY.
Restore any required membership metadata to its original path separately.
Do not overwrite active studies. Checksums verify preservation, not scientific correctness.

Secret scan: every included source file is scanned for recognized HF/GitHub/API
credentials, private-key headers and configured environment secrets, with overlap
across chunks. Findings stop packaging; matching values are never logged.
This heuristic cannot prove absence of every unknown or obfuscated secret.
Original RESULTs and reports are archival snapshots; no historical counts are altered.
