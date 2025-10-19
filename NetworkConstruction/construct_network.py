# NetworkConstruction/construct_network.py
from pathlib import Path
import pandas as pd, json, csv, string
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.resolve()
BASE_ROOT = PROJECT_ROOT.parent
IN_DIR  = BASE_ROOT / "data" / "SentimentalAnalysis"     # enriched_emails_{YYYY_MM}.csv
OUT_DIR = BASE_ROOT / "data" / "NetworkConstruction"     # network_edges_{X}.csv, network_nodes.ndjson, network_meta.json
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK = 25_000

def parse_list(val: str):
    """Parse semicolon-delimited recipient list into a clean list"""
    if not isinstance(val, str) or not val.strip():
        return []
    return [x for x in (p.strip() for p in val.split(";")) if x]

def main():
    files = sorted(IN_DIR.glob("enriched_emails_*.csv"))
    if not files:
        print(f"No input files in {IN_DIR}")
        return

    # per-partition accumulator
    nodes = set()
    partitions = defaultdict(lambda: defaultdict(lambda: {"w": 0, "sum_c": 0.0}))

    usecols = ["date","sender","recipients","cc","bcc","compound"]

    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f.name}")
        for chunk in pd.read_csv(f, usecols=usecols, chunksize=CHUNK):
            chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")

            for _, row in chunk.iterrows():
                # --- sender ---
                s = str(row.get("sender") or "").strip()
                if not s:
                    continue

                # extract year_month
                d = row.get("date")
                if pd.isna(d):
                    continue
                ym = f"{d.year:04d}_{d.month:02d}"

                # --- recipients ---
                rcpts = (
                    parse_list(str(row.get("recipients") or "")) +
                    parse_list(str(row.get("cc") or "")) +
                    parse_list(str(row.get("bcc") or ""))
                )
                if not rcpts:
                    rcpts = ["null"]

                # --- sentiment + date ---
                try:
                    c = float(row.get("compound") or 0.0)
                except Exception:
                    c = 0.0
                
                nodes.add(s)

                # --- update edges ---
                for t in set(rcpts):
                    t = str(t).strip()
                    if not t:
                        t = "null"
        
                    if t.lower() not in {"null", "nan", "none"}:
                        nodes.add(t)

                    entry = partitions[ym][(s, t)]
                    entry["w"] += 1
                    entry["sum_c"] += c

    # --- Write partitioned edge files ---
    total_edges, total_weight = 0, 0
    meta = {"partitions": []}

    for ym, edges_dict in sorted(partitions.items()):
        out_file = OUT_DIR / f"network_edges_{ym}.csv"
        with open(out_file, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["source", "target", "weight", "sentiment_avg", "year", "month"])
            y, m = ym.split("_")
            for (src, tgt), e in edges_dict.items():
                mean_c = e["sum_c"] / e["w"] if e["w"] else 0.0
                w.writerow([src, tgt, e["w"], f"{mean_c:.6f}", y, m])
                total_edges += 1
                total_weight += e["w"]
        meta["partitions"].append(f"network_edges_{ym}.csv")
        print(f"Saved {out_file.name} ({len(edges_dict)} edges)")

    # --- write nodes (SPEC: network_nodes.ndjson) ---
    with open(OUT_DIR / "network_nodes.ndjson", "w", encoding="utf-8") as nf:
        for n in sorted(nodes):
            nf.write(json.dumps({"node_id": n, "name": ""}) + "\n")

    # --- Write meta ---
    meta.update({
        "total_nodes": len(nodes),
        "total_edges_unique": total_edges,
        "total_edges_weighted": total_weight
    })
    with open(OUT_DIR / "network_meta.json", "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2)

    print("Year–Month partitions and metadata written successfully.")
    print(f"Nodes: {meta['total_nodes']} | Edges (unique): {meta['total_edges_unique']} | Weighted total: {meta['total_edges_weighted']}")


if __name__ == "__main__":
    main()
