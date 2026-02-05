import shutil
import argparse
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Set, Tuple


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="duplicate files in a directory")
    p.add_argument("--input", type=Path, default=Path("."), help="Input directory containing XML files.")
    p.add_argument("--count", type=int, default=4, help="Number of times to duplicate each file.")
    p.add_argument("--pattern", type=str, default="edge_records_*", help="Glob pattern for input XML files.")


    args = p.parse_args()
    in_dir = args.input
    count = args.count
    pattern = args.pattern

    files = sorted(in_dir.glob(pattern))
    if not files:
        print(f"No files found matching {pattern} in {in_dir}", file=sys.stderr)
    for file_path in files:
        for i in range(count):
            out_path = file_path.parent / f"{file_path.stem}_dup{i}{file_path.suffix}"
            shutil.copy(file_path, out_path)