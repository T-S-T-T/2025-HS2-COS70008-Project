# from __future__ import annotations
# import argparse
# import csv
# import json
# import sys
# from pathlib import Path
# from typing import Dict, Iterable, List, Optional, Tuple

# import pandas as pd
# ALIASES = {
#     "date": ["date", "message_date", "email_date", "Date", "msg_date", "timestamp", "sent_at"],
#     "compound": ["compound", "sent_compound", "vader_compound", "compound_score", "sentiment", "compoundPolarity"],
#     "node_id": ["node_id", "node", "id", "employee_id", "user_id"],
#     "burnout_prob": ["burnout_prob", "burnout_probability", "burnout_score", "risk_prob"],
#     "burnout_label": ["burnout_label", "risk_label", "label", "burnout_category"],
#     "anomaly_score": ["anomaly_score", "anomaly", "outlier_score", "anomalyIndex"],
#     "x": ["x", "pos_x", "x_coord"],
#     "y": ["y", "pos_y", "y_coord"],
# }

# def normalize_columns(df: pd.DataFrame, required: dict) -> pd.DataFrame:
#     ren = {}
#     for canon, must in required.items():
#         if canon in df.columns:
#             continue
#         for alt in ALIASES.get(canon, []):
#             if alt in df.columns:
#                 ren[alt] = canon
#                 break
#         if must and canon not in set(df.columns) | set(ren.values()):
#             raise ValueError(f"Missing required column '{canon}' (aliases: {ALIASES.get(canon)})")
#     return df.rename(columns=ren)

# def month_key(month: str) -> Tuple[int, int]:
#     """Sort key for 'YYYY_MM' strings."""
#     y, m = month.split("_")
#     return (int(y), int(m))


# def detect_months(base: Path) -> List[str]:
#     insights = (base / "OrganizationalInsight").glob("insights_*.csv")
#     months = {p.stem.split("_")[1] + "_" + p.stem.split("_")[2] for p in insights}
#     if not months:
#         sentiments = (base / "SentimentalAnalysis").glob("sentiment_scores_*.csv")
#         months = {p.stem.split("_")[2] + "_" + p.stem.split("_")[3] for p in sentiments}
#     return sorted(months, key=month_key)


# def ensure_dir(p: Path) -> None:
#     p.mkdir(parents=True, exist_ok=True)
# def generate_time_series(base: Path, out_dir: Path, month: str, chunksize: int = 10_000) -> Path:
#     sa_dir = base / "SentimentalAnalysis"
#     enriched = sa_dir / f"enriched_emails_{month}.csv"
#     scores = sa_dir / f"sentiment_scores_{month}.csv"

#     src = enriched if enriched.exists() else scores
#     if not src.exists():
#         raise FileNotFoundError(f"Missing sentiment monthly file for {month}: {enriched.name} or {scores.name}")

#     agg = {}

#     if src == enriched:
#         usecols = None
#     else:
#         usecols = ["date", "compound"]

#     for chunk in pd.read_csv(src, chunksize=chunksize):
#         chunk = normalize_columns(chunk, {"date": True, "compound": True})

#         dt = pd.to_datetime(chunk["date"], errors="coerce")
#         day = dt.dt.date.astype(str)
#         comp = pd.to_numeric(chunk["compound"], errors="coerce")
#         df = pd.DataFrame({"day": day, "compound": comp}).dropna()

#         grouped = df.groupby("day").agg(sum_comp=("compound", "sum"), cnt=("compound", "size"))
#         for d, row in grouped.iterrows():
#             if d not in agg:
#                 agg[d] = [row["sum_comp"], int(row["cnt"])]
#             else:
#                 agg[d][0] += row["sum_comp"]
#                 agg[d][1] += int(row["cnt"])

#         dt = pd.to_datetime(chunk["date"], errors="coerce")
#         day = dt.dt.date.astype(str)
#         comp = pd.to_numeric(chunk["compound"], errors="coerce")
#         df = pd.DataFrame({"day": day, "compound": comp}).dropna()

#         grouped = df.groupby("day").agg(sum_comp=("compound", "sum"), cnt=("compound", "size"))
#         for d, row in grouped.iterrows():
#             if d not in agg:
#                 agg[d] = [row["sum_comp"], int(row["cnt"])]
#             else:
#                 agg[d][0] += row["sum_comp"]
#                 agg[d][1] += int(row["cnt"])

#     out_rows = []
#     for d in sorted(agg.keys()):
#         sum_comp, cnt = agg[d]
#         avg = sum_comp / cnt if cnt else 0.0
#         out_rows.append({"date": d, "avg_compound": avg, "total_emails": cnt})

#     out = pd.DataFrame(out_rows)
#     out_path = out_dir / f"timeseries_data_{month}.csv"
#     out.to_csv(out_path, index=False)
#     return out_path


# def generate_burnout_bar(base: Path, out_dir: Path, month: str) -> Path:
#     ins = base / "OrganizationalInsight" / f"insights_{month}.csv"
#     if not ins.exists():
#         raise FileNotFoundError(f"Missing insights file for {month}: {ins}")

#     use = pd.read_csv(ins)
#     use = normalize_columns(use, {"node_id": True, "burnout_prob": True, "burnout_label": True})

#     out = use[["node_id", "burnout_prob", "burnout_label"]].copy()
#     out_path = out_dir / f"burnout_bar_data_{month}.csv"
#     out.to_csv(out_path, index=False)
#     return out_path


# def stream_layout_coords(ndjson_path: Path) -> Dict[str, Tuple[float, float]]:
#     """Read NDJSON layout into dict: node_id -> (x, y)."""
#     coords: Dict[str, Tuple[float, float]] = {}
#     with ndjson_path.open("r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             obj = json.loads(line)
#             node_id = None
#             for k in ALIASES["node_id"]:
#                 if k in obj:
#                     node_id = obj[k]; break
#             xv = None
#             for k in ALIASES["x"]:
#                 if k in obj:
#                     xv = obj[k]; break
#             yv = None
#             for k in ALIASES["y"]:
#                 if k in obj:
#                     yv = obj[k]; break
#             if node_id is None or xv is None or yv is None:
#                 continue
#             coords[str(node_id)] = (float(xv), float(yv))
#     return coords


# def generate_network_snapshot(base: Path, out_dir: Path, month: str) -> Path:
#     layout_path = base / "NetworkGraphAnalysis" / f"graph_layout_{month}.ndjson"
#     if not layout_path.exists():
#         raise FileNotFoundError(f"Missing layout NDJSON for {month}: {layout_path}")

#     coords = stream_layout_coords(layout_path)
#     if not coords:
#         raise ValueError(f"No coordinates parsed from {layout_path}")

#     ins = base / "OrganizationalInsight" / f"insights_{month}.csv"
#     ins_df = pd.read_csv(ins) if ins.exists() else pd.DataFrame()
#     if not ins_df.empty:
#         ins_df = normalize_columns(ins_df, {"node_id": False, "burnout_prob": False, "anomaly_score": False})

#     sna_path = base / "NetworkAnalysis" / "sna_metrics.csv"
#     sna_df = pd.read_csv(sna_path) if sna_path.exists() else pd.DataFrame()
#     if not sna_df.empty:
#         sna_df = normalize_columns(sna_df, {"node_id": False, "anomaly_score": False})

#     base_df = pd.DataFrame(
#         [(nid, xy[0], xy[1]) for nid, xy in coords.items()],
#         columns=["node_id", "x", "y"],
#     )

#     if not ins_df.empty and "burnout_prob" in ins_df.columns:
#         base_df = base_df.merge(
#             ins_df[["node_id", "burnout_prob"]], on="node_id", how="left"
#         )
#     else:
#         base_df["burnout_prob"] = pd.NA

#     if not ins_df.empty and "anomaly_score" in ins_df.columns:
#         base_df = base_df.merge(
#             ins_df[["node_id", "anomaly_score"]], on="node_id", how="left"
#         )
#     elif not sna_df.empty and "anomaly_score" in sna_df.columns:
#         base_df = base_df.merge(
#             sna_df[["node_id", "anomaly_score"]], on="node_id", how="left"
#         )
#     else:
#         base_df["anomaly_score"] = pd.NA

#     out_path = out_dir / f"network_snapshot_data_{month}.csv"
#     base_df.to_csv(out_path, index=False)
#     return out_path


# def write_visual_config(out_dir: Path, months: List[str]) -> Path:
#     config = {
#         "months": months,
#         "defaults": {
#             "time_series": {"chart_type": "line"},
#             "burnout_bar": {"chart_type": "bar"},
#             "network_snapshot": {"chart_type": "scatter"},
#         },
#     }
#     out_path = out_dir / "visual_config.json"
#     with out_path.open("w", encoding="utf-8") as f:
#         json.dump(config, f, indent=2)
#     return out_path

# def build_all(base_dir: Path, months: Optional[List[str]] = None, chunksize: int = 10_000, force: bool = False) -> None:
#     base = base_dir.resolve()
#     out_dir = base / "InteractiveVisualization"
#     ensure_dir(out_dir)

#     if not months:
#         months = detect_months(base)
#         if not months:
#             raise RuntimeError("No months detected. Supply --months or add input files.")

#     months = sorted(set(months), key=month_key)

#     written: List[str] = []
#     for month in months:
#         print(f"\n=== Building month {month} ===")
#         ts_path = out_dir / f"timeseries_data_{month}.csv"
#         if force or not ts_path.exists():
#             p = generate_time_series(base, out_dir, month, chunksize=chunksize)
#             print(f"  ✔ time series -> {p}")
#         else:
#             print(f"  ↷ time series exists -> {ts_path}")
#         bb_path = out_dir / f"burnout_bar_data_{month}.csv"
#         if force or not bb_path.exists():
#             p = generate_burnout_bar(base, out_dir, month)
#             print(f"  ✔ burnout bar -> {p}")
#         else:
#             print(f"  ↷ burnout bar exists -> {bb_path}")
#         ns_path = out_dir / f"network_snapshot_data_{month}.csv"
#         if force or not ns_path.exists():
#             p = generate_network_snapshot(base, out_dir, month)
#             print(f"  ✔ network snapshot -> {p}")
#         else:
#             print(f"  ↷ network snapshot exists -> {ns_path}")
#         written.append(month)

#     cfg = write_visual_config(out_dir, written)
#     print(f"\n✔ Wrote visual_config -> {cfg}")

# def parse_args() -> argparse.Namespace:
#     ap = argparse.ArgumentParser(description="Build flat CSVs + index for Power BI visuals")
#     ap.add_argument("--base", type=str, default="../data", help="Base data folder path")
#     ap.add_argument("--months", type=str, default="", help="Comma-separated months e.g. 2025_01,2025_02 (defaults to auto-detect)")
#     ap.add_argument("--chunksize", type=int, default=10_000, help="Chunk size for large CSVs")
#     ap.add_argument("--force", action="store_true", help="Overwrite existing outputs")
#     return ap.parse_args()


# def main() -> None:
#     args = parse_args()
#     base = Path(args.base)
#     months = [m.strip() for m in args.months.split(",") if m.strip()] if args.months else None
#     try:
#         build_all(base, months=months, chunksize=args.chunksize, force=args.force)
#     except Exception as e:
#         print(f"ERROR: {e}", file=sys.stderr)
#         sys.exit(1)


# if __name__ == "__main__":
#     main()

    ,asmdnfbgakejrhnisuecghwrmk
import json
from pathlib import Path

import pandas as pd

# ——— CONFIGURATION ——————————————————————————

BASE_DIR   = Path(__file__).resolve().parent.parent

SA_DIR     = BASE_DIR / "data" / "SentimentalAnalysis"
NA_DIR     = BASE_DIR / "data" / "NetworkAnalysis"
OI_DIR     = BASE_DIR / "data" / "OrganizationalInsight"
NG_DIR     = BASE_DIR / "data" / "NetworkGraphAnalysis"
OUT_DIR    = BASE_DIR / "data" / "InteractiveVisualization"

CHUNK_SIZE = 10_000
DATE_COL   = "date"  # adjust if your enriched_emails uses a different column name

# ——— HELPERS ———————————————————————————————

def list_months():
    """Infer available months from insights files."""
    return [p.stem.replace("insights_", "") for p in sorted(OI_DIR.glob("insights_*.csv"))]

def stream_layout(layout_path: Path):
    """Yield (node_id, x, y) from NDJSON layout file."""
    with open(layout_path, "r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            yield rec["node_id"], float(rec["x"]), float(rec["y"])

# ——— MAIN —————————————————————————————————————

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Optional: load global metrics once (not required for outputs below)
    metrics_path = NA_DIR / "sna_metrics.csv"
    metrics_available = metrics_path.exists()
    if metrics_available:
        df_metrics = pd.read_csv(metrics_path).set_index("node_id")
        # If needed, use df_metrics.loc[node_id, ...] later (not required for current outputs)

    months = list_months()
    visual_config = {
        "months": months,
        "defaults": {
            "time_series":      {"chart_type": "line"},
            "burnout_bar":      {"chart_type": "bar"},
            "network_snapshot": {"chart_type": "scatter"}
        }
    }

    for month in months:
        enriched_path = SA_DIR / f"enriched_emails_{month}.csv"
        insights_path = OI_DIR / f"insights_{month}.csv"
        layout_path   = NG_DIR / f"graph_layout_{month}.ndjson"

        # ——— Time-series CSV ——————————————————
        ts_rows = []
        if not enriched_path.exists():
            raise FileNotFoundError(f"Missing enriched emails for {month}: {enriched_path}")

        # Aggregate by day: avg_compound (mean of compound), total_emails (count)
        day_sums = {}
        day_counts = {}
        for chunk in pd.read_csv(enriched_path, chunksize=CHUNK_SIZE, usecols=[DATE_COL, "compound"]):
            if DATE_COL not in chunk.columns or "compound" not in chunk.columns:
                raise ValueError(f"Expected columns '{DATE_COL}' and 'compound' in {enriched_path}")

            day = pd.to_datetime(chunk[DATE_COL], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
            comp = pd.to_numeric(chunk["compound"], errors="coerce")

            df_tmp = pd.DataFrame({"day": day, "compound": comp}).dropna()
            if df_tmp.empty:
                continue

            # Explicitly name columns to avoid attribute errors
            grp = df_tmp.groupby("day")["compound"].agg(sum_val="sum", count_val="count").reset_index()

            for r in grp.itertuples(index=False):
                prev_sum = day_sums.get(r.day, 0.0)
                prev_cnt = day_counts.get(r.day, 0)
                day_sums[r.day] = prev_sum + float(r.sum_val)
                day_counts[r.day] = prev_cnt + int(r.count_val)

        for day_key in sorted(day_sums.keys()):
            total = day_counts.get(day_key, 0)
            avg_comp = (day_sums[day_key] / total) if total > 0 else 0.0
            ts_rows.append({"date": day_key, "avg_compound": avg_comp, "total_emails": total})

        ts_out = OUT_DIR / f"timeseries_data_{month}.csv"
        pd.DataFrame(ts_rows, columns=["date", "avg_compound", "total_emails"]).to_csv(ts_out, index=False)

        # ——— Burnout-bar CSV —————————————————
        if not insights_path.exists():
            raise FileNotFoundError(f"Missing insights for {month}: {insights_path}")
        df_ins = pd.read_csv(insights_path)
        bb_out = OUT_DIR / f"burnout_bar_data_{month}.csv"
        df_bb = df_ins[["node_id", "burnout_prob", "burnout_label"]].copy()
        df_bb.to_csv(bb_out, index=False)

        # ——— Network-snapshot CSV —————————————
        if not layout_path.exists():
            fallback = NG_DIR / "graph_layout_all.ndjson"
            if fallback.exists():
                layout_path = fallback
            else:
                raise FileNotFoundError(f"Missing layout for {month}: {layout_path} (and fallback {fallback})")

        coords = {nid: (x, y) for nid, x, y in stream_layout(layout_path)}

        df_coords = pd.DataFrame.from_dict(coords, orient="index", columns=["x", "y"]).rename_axis("node_id").reset_index()
        df_ns = df_ins[["node_id", "anomaly_score", "burnout_prob"]].merge(df_coords, on="node_id", how="left")
        df_ns = df_ns[["node_id", "x", "y", "anomaly_score", "burnout_prob"]]

        ns_out = OUT_DIR / f"network_snapshot_data_{month}.csv"
        df_ns.to_csv(ns_out, index=False)

        print(f"[DONE] {month} →")
        print(f"   • {ts_out}")
        print(f"   • {bb_out}")
        print(f"   • {ns_out}")

    # ——— Master index JSON ——————————————————
    vc_path = OUT_DIR / "visual_config.json"
    with open(vc_path, "w", encoding="utf-8") as fh:
        json.dump(visual_config, fh, indent=2)
    print(f"[DONE] visual_config.json → {vc_path}")

if __name__ == "__main__":
    print("\nRunning...\n")
    main()
    print("\nFinished!\n")