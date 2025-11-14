import argparse
import re
from pathlib import Path
import pandas as pd


HEADER = [
    "file_id",
    "file_name",
    "interval_begin",
    "interval_end",
    "interval_id",
    "id",
    "speed",
    "volume",
]


def load_long(csv_path: Path, value_name: str) -> pd.DataFrame:
    """Load a wide matrix CSV with an 'Interval' column into long form.

    Result columns: Interval, id, <value_name>
    """
    df = pd.read_csv(csv_path)
    if "Interval" not in df.columns:
        raise ValueError(f"{csv_path} missing 'Interval' column")
    long_df = df.melt(id_vars=["Interval"], var_name="id", value_name=value_name)
    return long_df


def build_edge_records(speed_csv: Path, volume_csv: Path, out_csv: Path, file_id: int) -> None:
    speed_long = load_long(speed_csv, "speed")
    volume_long = load_long(volume_csv, "volume")

    # Outer join so edges present in either matrix are included
    merged = pd.merge(speed_long, volume_long, on=["Interval", "id"], how="outer")

    # Sort by interval then id for readability
    merged = merged.sort_values(["Interval", "id"]).reset_index(drop=True)

    # Compose fixed metadata
    f_id = str(file_id)
    f_name = f"edge_records_{file_id}.xml"
    interval_begin = (merged["Interval"].astype(float) * 60.0).map(lambda x: f"{x:.3f}")
    interval_end = (merged["Interval"].astype(float) * 60.0 + 60.0).map(lambda x: f"{x:.3f}")

    out = pd.DataFrame({
        "file_id": f_id,
        "file_name": f_name,
        "interval_begin": interval_begin,
        "interval_end": interval_end,
        "interval_id": "edge_records",
        "id": merged["id"],
        "speed": merged["speed"],
        "volume": merged["volume"],
    })[HEADER]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False, float_format="%.4f", na_rep="")


def main():
    p = argparse.ArgumentParser(description="Build edge_records_XX.csv from speed/volume matrices")

    # Two usage modes: single-file mode or scan mode
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dir_path", type=Path, help="Directory containing a single pair of matrices")
    mode.add_argument("--scan_dir", type=Path, help="Scan this directory recursively for matrix pairs")

    # Single-file mode arguments
    p.add_argument("--id", type=int, help="File id number (e.g., 22) for single-file mode")

    # Optional output directory (used in scan mode or to override default)
    p.add_argument("--out_dir", type=Path, default=None, help="Output directory for generated edge_records_XX.csv. Default: beside inputs")

    args = p.parse_args()

    # Helper to resolve output path
    def resolve_out_path(default_dir: Path, fid: int) -> Path:
        if args.out_dir is not None:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            return args.out_dir / f"edge_records_{fid}.csv"
        return default_dir / f"edge_records_{fid}.csv"

    if args.dir_path is not None:
        if args.id is None:
            raise SystemExit("--id is required when using --dir_path")
        speed_file_path = args.dir_path / f"speed_matrix_ecords_{args.id}.csv"
        volume_file_path = args.dir_path / f"volume_matrix_ecords_{args.id}.csv"
        out_file_path = resolve_out_path(args.dir_path, args.id)
        build_edge_records(speed_file_path, volume_file_path, out_file_path, args.id)
        print(f"Built file at {out_file_path}")
        return

    # Scan mode: find all matching speed/volume pairs by ID
    scan_root: Path = args.scan_dir
    if not scan_root.exists():
        raise SystemExit(f"Scan directory not found: {scan_root}")

    speed_re = re.compile(r"speed_matrix_ecords_(\d+)\.csv$", re.IGNORECASE)
    volume_re = re.compile(r"volume_matrix_ecords_(\d+)\.csv$", re.IGNORECASE)

    speeds = {}
    volumes = {}

    for pth in scan_root.rglob("*.csv"):
        name = pth.name
        m = speed_re.search(name)
        if m:
            speeds[int(m.group(1))] = pth
            continue
        m = volume_re.search(name)
        if m:
            volumes[int(m.group(1))] = pth

    common_ids = sorted(set(speeds.keys()) & set(volumes.keys()))
    if not common_ids:
        raise SystemExit("No matching speed/volume matrix pairs found under scan directory.")

    for fid in common_ids:
        speed_path = speeds[fid]
        volume_path = volumes[fid]
        # Place output beside the speed/volume files unless --out_dir provided
        out_dir = args.out_dir if args.out_dir is not None else speed_path.parent
        out_path = resolve_out_path(out_dir, fid)
        build_edge_records(speed_path, volume_path, out_path, fid)
        print(f"Built edge_records_{fid}.csv at {out_path}")


if __name__ == "__main__":
    main()
