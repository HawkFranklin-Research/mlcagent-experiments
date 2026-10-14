"""Build cross-arm tables from curated exports; never performs inference."""

import argparse
from pathlib import Path

from mlca.evaluation.cross_arms import summarize

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exports", type=Path)
    args = parser.parse_args()
    print(summarize(args.exports))
