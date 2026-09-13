"""Download the external inputs of the coverage analysis (Section 5.3).

Downloads what is public and prints exact instructions for what is not. Nothing
is redistributed with this repository: PathVQA and PMC-VQA come from the Hub,
and ProbMed is a gated release whose access is granted by its maintainers.

Files land in --out-dir under the names src/analysis/coverage.py expects.

Usage:
    python -m src.analysis.fetch_coverage_data
    python -m src.analysis.fetch_coverage_data --out-dir data/coverage
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time

JOBS = [
    ("flaviagiammarino/path-vqa", False,
     ["data/test-00000-of-00003-e9adadb4799f44d3.parquet",
      "data/test-00001-of-00003-7ea98873fc919813.parquet",
      "data/test-00002-of-00003-1628308435019820.parquet"], "path-vqa__"),
    ("RadGenome/PMC-VQA", False, ["test_clean.csv", "test.csv"], "PMC-VQA__"),
    ("rippleripple/ProbMed", True, ["test/test.json"], "ProbMed__"),
]

GATED_NOTE = """
ProbMed is a GATED release. To obtain it:
  1. Request access at https://huggingface.co/datasets/rippleripple/ProbMed
  2. Once granted, authenticate:  huggingface-cli login
  3. Re-run this script.
Without the file, run the coverage analysis with --skip-missing; every other
benchmark still reports.
"""


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for block in iter(lambda: fp.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(repo: str, remote: str, dest: str, attempts: int = 8) -> bool:
    """Download one file, retrying on the Hub's rate limiting."""
    from huggingface_hub import hf_hub_download
    for attempt in range(attempts):
        try:
            cached = hf_hub_download(repo, remote, repo_type="dataset")
            shutil.copyfile(cached, dest)
            return True
        except Exception as exc:
            if "gated" in str(exc).lower() or "401" in str(exc):
                raise
            wait = min(20 * (attempt + 1), 120)
            print(f"    attempt {attempt + 1} failed ({type(exc).__name__}); "
                  f"waiting {wait}s", flush=True)
            time.sleep(wait)
    return False


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="data/coverage")
    p.add_argument("--manifest", default=None,
                   help="write sha256 and size of what was downloaded")
    return p.parse_args()


def main():
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    manifest, missing = {}, []

    for repo, gated, files, prefix in JOBS:
        for remote in files:
            dest = os.path.join(args.out_dir, prefix + os.path.basename(remote))
            if os.path.exists(dest):
                print(f"  have  {os.path.basename(dest)}")
            else:
                print(f"  fetch {os.path.basename(dest)} from {repo}")
                try:
                    if not fetch(repo, remote, dest):
                        missing.append((repo, remote, gated))
                        continue
                except Exception as exc:
                    print(f"    {type(exc).__name__}: {str(exc).splitlines()[0]}")
                    missing.append((repo, remote, gated))
                    continue
            manifest[os.path.basename(dest)] = {
                "repo": repo, "remote": remote, "gated": gated,
                "sha256": sha256(dest), "bytes": os.path.getsize(dest)}

    if args.manifest:
        with open(args.manifest, "w") as fp:
            json.dump(manifest, fp, indent=1)
        print(f"\nwrote {args.manifest}")

    if missing:
        print(f"\n{len(missing)} file(s) not obtained:")
        for repo, remote, gated in missing:
            print(f"  {repo}/{remote}{'   [GATED]' if gated else ''}")
        if any(g for _, _, g in missing):
            print(GATED_NOTE)
    else:
        print(f"\nall inputs present in {args.out_dir}")


if __name__ == "__main__":
    main()
