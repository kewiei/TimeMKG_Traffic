#!/usr/bin/env python3
"""
SUMO meandata XML -> CSV converter

Converts one or many files named like "edge_records_XXXX.xml" into CSVs.
By default, writes one CSV per XML with the same stem in an output folder.

Usage:
  python sumo_meandata_xml_to_csv.py --input DIR --output DIR [--pattern "edge_records_*.xml"] [--combine]

Examples:
  # Per-file CSVs to ./csv_out
  python sumo_meandata_xml_to_csv.py --input ./ --output ./csv_out

  # Combine all rows from all matching XMLs into one CSV
  python sumo_meandata_xml_to_csv.py --input ./ --output ./csv_out --combine

Notes:
- The converter preserves interval context (begin, end, id) for each edge row.
- Columns are the union of attributes found across the file(s).
- Only Python standard library is used.
"""
from __future__ import annotations
import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
import xml.etree.ElementTree as ET


EDGE_FILE_RE = re.compile(r"edge_records_(\d+)\.xml$", re.IGNORECASE)


def discover_edge_fields(xml_path: Path) -> Set[str]:
    """First pass: return the union of all <edge> attribute names in the file (including 'id')."""
    fields: Set[str] = set()
    # iterparse is memory-friendly. We only care about 'edge' end events.
    for ev, elem in ET.iterparse(xml_path, events=("end",)):
        if elem.tag == "edge":
            fields.update(elem.attrib.keys())
            elem.clear()
    return fields


def extract_file_id(xml_path: Path) -> Optional[str]:
    m = EDGE_FILE_RE.search(xml_path.name)
    return m.group(1) if m else None


def iter_rows(xml_path: Path) -> Iterable[Tuple[Dict[str, str], Dict[str, str]]]:
    """
    Second pass: yield (interval_attribs, edge_attribs) for each <edge> inside an <interval>.
    interval_attribs keys: 'interval_begin','interval_end','interval_id'
    """
    current_interval: Optional[Dict[str, str]] = None
    for ev, elem in ET.iterparse(xml_path, events=("start", "end")):
        if ev == "start" and elem.tag == "interval":
            # capture the interval attributes we care about
            current_interval = {
                "interval_begin": elem.attrib.get("begin", ""),
                "interval_end": elem.attrib.get("end", ""),
                "interval_id": elem.attrib.get("id", ""),
            }
        elif ev == "end":
            if elem.tag == "edge":
                if current_interval is None:
                    # Should not happen in valid meandata files, but guard anyway
                    current_interval = {"interval_begin": "", "interval_end": "", "interval_id": ""}
                edge_attribs = dict(elem.attrib)  # shallow copy
                yield current_interval, edge_attribs
                elem.clear()
            elif elem.tag == "interval":
                # end of an interval scope
                current_interval = None
                elem.clear()


def write_csv(rows: List[Tuple[Dict[str, str], Dict[str, str]]],
              edge_fields: List[str],
              csv_path: Path,
              file_meta: Dict[str, str]) -> None:
    """
    Write rows to CSV. `edge_fields` is the ordered list of edge attribute columns.
    `file_meta` may include 'file_id' and 'file_name' to add per-row context.
    """
    # Compose column order: file meta + interval context + edge fields
    base_cols = []
    if "file_id" in file_meta:
        base_cols.append("file_id")
    if "file_name" in file_meta:
        base_cols.append("file_name")
    interval_cols = ["interval_begin", "interval_end", "interval_id"]
    fieldnames = base_cols + interval_cols + edge_fields

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for interval_attribs, edge_attribs in rows:
            out: Dict[str, str] = {}
            # file metadata
            for k in base_cols:
                out[k] = file_meta.get(k, "")
            # interval context
            out.update(interval_attribs)
            # edge attributes (fill missing with empty string)
            for k in edge_fields:
                out[k] = edge_attribs.get(k, "")
            w.writerow(out)


def process_file(xml_path: Path, out_dir: Path) -> Path:
    """Convert a single XML into a CSV and return the CSV path."""
    file_id = extract_file_id(xml_path) or ""
    file_meta = {"file_id": file_id, "file_name": xml_path.name}

    # First pass: discover union of edge attribute names
    edge_fields = sorted(discover_edge_fields(xml_path))

    # Second pass: collect rows
    rows: List[Tuple[Dict[str, str], Dict[str, str]]] = list(iter_rows(xml_path))

    csv_path = out_dir / (xml_path.stem + ".csv")
    write_csv(rows, edge_fields, csv_path, file_meta)
    return csv_path


def process_dir(in_dir: Path, out_dir: Path, pattern: str, combine: bool) -> Optional[Path]:
    """Process all matching XMLs. If `combine` True, also write a combined CSV and return its path."""
    xml_files = sorted(in_dir.glob(pattern))
    if not xml_files:
        print(f"No files found matching {pattern} in {in_dir}", file=sys.stderr)
        return None

    combined_rows: List[Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]] = []
    combined_edge_fields: Set[str] = set()

    for xml_path in xml_files:
        # Per-file CSV
        out_csv = process_file(xml_path, out_dir)
        print(f"Wrote: {out_csv}")

        if combine:
            # For combined, we need to gather rows and edge fields with file meta per row
            file_id = extract_file_id(xml_path) or ""
            file_meta = {"file_id": file_id, "file_name": xml_path.name}
            combined_edge_fields.update(discover_edge_fields(xml_path))
            for interval_attribs, edge_attribs in iter_rows(xml_path):
                combined_rows.append((interval_attribs, edge_attribs, file_meta))

    if combine:
        combined_csv = out_dir / "edge_records_combined.csv"
        # Order fields
        edge_fields = sorted(combined_edge_fields)
        # Write combined CSV
        base_cols = ["file_id", "file_name"]
        interval_cols = ["interval_begin", "interval_end", "interval_id"]
        fieldnames = base_cols + interval_cols + edge_fields
        out_dir.mkdir(parents=True, exist_ok=True)
        with combined_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for interval_attribs, edge_attribs, file_meta in combined_rows:
                out: Dict[str, str] = {}
                out.update(file_meta)
                out.update(interval_attribs)
                for k in edge_fields:
                    out[k] = edge_attribs.get(k, "")
                w.writerow(out)
        print(f"Wrote combined: {combined_csv}")
        return combined_csv

    return None


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Convert SUMO meandata edge_records_*.xml to CSV.")
    p.add_argument("--input", type=Path, default=Path("."), help="Input directory containing XML files.")
    p.add_argument("--output", type=Path, default=Path("./csv_out"), help="Output directory for CSV files.")
    p.add_argument("--pattern", type=str, default="edge_records_*.xml", help="Glob pattern for input XML files.")
    p.add_argument("--combine", action="store_true", help="Also write a single combined CSV for all inputs.")
    args = p.parse_args(argv)

    process_dir(args.input, args.output, args.pattern, args.combine)


if __name__ == "__main__":
    main()