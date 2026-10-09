from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from rsdehamba_lite.data import discover_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()

    root = Path(args.data_root)
    if not root.exists():
        raise SystemExit(f"Dataset root does not exist: {root}")

    pairs = discover_pairs(root)
    print(f"root: {root}")
    print(f"pairs: {len(pairs)}")
    print("by split:", dict(Counter(p.split for p in pairs)))
    print("by haze level:", dict(Counter(p.haze_level for p in pairs)))
    for pair in pairs[:5]:
        print(f"sample: {pair.haze_level}/{pair.split} hazy={pair.hazy} clear={pair.clear}")

    if not pairs:
        print("No pairs found. Check folder names; supported aliases are documented in README.md.")


if __name__ == "__main__":
    main()

