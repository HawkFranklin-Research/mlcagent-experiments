"""Download the pinned biomedical release from Hugging Face and verify every file's SHA-256.

The token is read from HF_TOKEN or from --token-file. It is sent only in the request
header and is never printed, logged or written.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = "HawkFranklin-Research/mlcagent-bio-bench"
REVISION = "111de16fe082d4beab8ca70e779601b156c596c8"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data_prep" / f"hf-{REVISION[:7]}"  # data_prep/ is never pushed
INDEX = ["SHA256SUMS.txt", "MANIFEST.csv"]


def token(path=None):
    if path:
        value = Path(path).expanduser().read_text(encoding="utf-8").strip()
    else:
        value = os.environ.get("HF_TOKEN", "").strip()
    if not value:
        raise SystemExit(
            "No token: set HF_TOKEN or pass --token-file (read access is enough)"
        )
    return value


def url(repo, revision, path):
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"


def http_fetch(address, secret, destination):
    request = urllib.request.Request(
        address, headers={"Authorization": f"Bearer {secret}"}
    )
    try:
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            destination.open("wb") as out,
        ):
            while chunk := response.read(1 << 20):
                out.write(chunk)
    except urllib.error.HTTPError as exc:
        # Never include request headers in an error message.
        raise RuntimeError(f"Download failed with HTTP {exc.code}: {address}") from None


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def sums(text):
    result = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        digest, path = line.split(maxsplit=1)
        path = path.lstrip("*").strip()
        if (
            not re.fullmatch(r"[0-9a-f]{64}", digest)
            or path.startswith("/")
            or ".." in Path(path).parts
        ):
            raise ValueError(f"Invalid checksum line: {line}")
        result[path] = digest
    return result


def get(repo, revision, path, out, secret, fetch, expected=None):
    target = out / path
    if target.is_file() and expected and sha256(target) == expected:
        return "kept"
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".part-", dir=target.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        fetch(url(repo, revision, path), secret, temporary)
        if expected and sha256(temporary) != expected:
            raise ValueError(f"SHA-256 mismatch after download: {path}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return "downloaded"


def fetch_release(
    repo, revision, out, secret, include=("prepared/",), fetch=http_fetch
):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Pin a full 40-character commit revision")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    get(repo, revision, "SHA256SUMS.txt", out, secret, fetch)
    checksums = sums((out / "SHA256SUMS.txt").read_text(encoding="utf-8"))
    if "MANIFEST.csv" not in checksums:
        raise ValueError("SHA256SUMS.txt does not cover MANIFEST.csv")
    wanted = ["MANIFEST.csv"] + sorted(
        p
        for p in checksums
        if p != "MANIFEST.csv" and any(p.startswith(prefix) for prefix in include)
    )
    if len(wanted) == 1:
        raise ValueError("No release files match the requested prefixes")
    status = {
        path: get(repo, revision, path, out, secret, fetch, checksums[path])
        for path in wanted
    }
    record = {
        "repo": repo,
        "revision": revision,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "files": {
            path: {"sha256": checksums[path], "status": status[path]} for path in wanted
        },
    }
    (out / "fetch.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    return record


def verify(out):
    out = Path(out)
    record = json.loads((out / "fetch.json").read_text(encoding="utf-8"))
    bad = [
        path
        for path, item in record["files"].items()
        if sha256(out / path) != item["sha256"]
    ]
    if bad:
        raise SystemExit(f"Changed or damaged files: {bad}")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--include", nargs="+", default=["prepared/"])
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_only:
        record = verify(args.out)
    else:
        record = fetch_release(
            args.repo, args.revision, args.out, token(args.token_file), args.include
        )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "revision": record["revision"],
                "files": len(record["files"]),
            }
        )
    )


if __name__ == "__main__":
    sys.exit(main())
