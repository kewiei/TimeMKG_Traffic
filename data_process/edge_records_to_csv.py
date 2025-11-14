
#!/usr/bin/env python3
import argparse, os, re, csv
from typing import Dict, List, Set

INTERVAL_OPEN_RE = re.compile(r"<\s*interval\s+([^>/]+?)>")
EDGE_RE = re.compile(r"<\s*edge\s+([^>/]+?)\/\s*>")
ATTR_RE = re.compile(r'(\w+)="([^"]*)"')

def parse_attrs(attr_str: str) -> Dict[str, str]:
    return {k: v for k, v in ATTR_RE.findall(attr_str)}

def filename_id(fname: str) -> str:
    m = re.search(r'(\d+)', os.path.basename(fname))
    return m.group(1) if m else ""

def collect_fields(path: str) -> List[str]:
    fields: Set[str] = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        cur_interval = {}
        for line in f:
            m = EDGE_RE.search(line)
            if m:
                attrs = parse_attrs(m.group(1))
                fields.update(attrs.keys())
    ordered = []
    if "id" in fields:
        ordered.append("id"); fields.remove("id")
    ordered += sorted(fields)
    return ordered

def write_csv(path: str, out_csv: str):
    edge_fields = collect_fields(path)
    prefix = ["file_id","file_name","interval_begin","interval_end","interval_id"]
    header = prefix + edge_fields
    f_id = filename_id(path)
    f_name = os.path.basename(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f, open(out_csv, "w", newline="", encoding="utf-8") as g:
        w = csv.DictWriter(g, fieldnames=header)
        w.writeheader()
        cur_interval = {"interval_begin":"","interval_end":"","interval_id":""}
        for line in f:
            m_int = INTERVAL_OPEN_RE.search(line)
            if m_int:
                ia = parse_attrs(m_int.group(1))
                cur_interval = {
                    "interval_begin": ia.get("begin",""),
                    "interval_end": ia.get("end",""),
                    "interval_id": ia.get("id",""),
                }
            m_edge = EDGE_RE.search(line)
            if m_edge:
                ea = parse_attrs(m_edge.group(1))
                row = {
                    "file_id": f_id,
                    "file_name": f_name,
                    **cur_interval
                }
                for k in edge_fields:
                    row[k] = ea.get(k, "")
                w.writerow(row)

def main():
    ap = argparse.ArgumentParser(description="Robust converter for SUMO meandata edge XML to CSV (regex-based).")
    ap.add_argument("input")
    ap.add_argument("-o","--output")
    args = ap.parse_args()
    out = args.output or (os.path.splitext(args.input)[0] + ".csv")
    write_csv(args.input, out)
    print(out)

if __name__ == "__main__":
    main()
