#!/usr/bin/env python3
"""
Run an end-to-end demo of the Network Graph Analysis pipeline.

Usage:
  python run_network_graph_demo.py
  # Optional flags:
  python run_network_graph_demo.py \
      --base ./data \
      --subgraph-threshold 50000 \
      --topn 20000 \
      --seed 12345
"""

import os
import csv
import json
import time
import datetime
from typing import Dict, Set, Iterable, Tuple, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import argparse


# ----------------------------
# Args & paths
# ----------------------------
def parse_args():
    BASE_DEFAULT = Path(__file__).resolve().parent.parent / "data"
    parser = argparse.ArgumentParser(description="Network Graph Analysis demo runner")
    parser.add_argument("--base", default=str(BASE_DEFAULT),
                        help="Base data directory containing NetworkConstruction/, NetworkAnalysis/, etc.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for reproducible layouts")
    parser.add_argument("--subgraph-threshold", type=int, default=50000,
                        help="If node count exceeds this, sample a subgraph before layout")
    parser.add_argument("--topn", type=int, default=20000,
                        help="Top-N nodes to keep when sampling by PageRank/degree")
    return parser.parse_args()


# ----------------------------
# Core helpers
# ----------------------------
def stream_node_ids(ndjson_path: str) -> Set[str]:
    node_ids: Set[str] = set()
    with open(ndjson_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                nid = obj.get("node_id")
                if nid:
                    node_ids.add(str(nid))
            except json.JSONDecodeError:
                continue
    return node_ids


def maybe_load_metrics(csv_path: str) -> Dict[str, Dict[str, float]]:
    if not os.path.exists(csv_path):
        return {}
    required = ["node_id", "degree", "pagerank", "betweenness", "clustering_coeff"]
    available = set(pd.read_csv(csv_path, nrows=0).columns)
    usecols = [c for c in required if c in available]
    if "node_id" not in usecols:
        return {}

    df = pd.read_csv(csv_path, usecols=usecols)
    df = df.drop_duplicates(subset=["node_id"]).set_index("node_id")
    return df.to_dict(orient="index")


def stream_edges_to_graph(edge_csv_paths: Iterable[str],
                          node_filter: Optional[Set[str]] = None) -> nx.Graph:
    G = nx.Graph()
    if node_filter:
        G.add_nodes_from(node_filter)

    for path in edge_csv_paths:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                s = row.get("source")
                t = row.get("target")
                if not s or not t or s == t:
                    continue
                if node_filter and (s not in node_filter or t not in node_filter):
                    continue
                try:
                    w = float(row.get("weight", 1) or 1)
                except ValueError:
                    w = 1.0

                if G.has_edge(s, t):
                    G[s][t]["weight"] += w
                else:
                    G.add_edge(s, t, weight=w)
    return G


def sample_subgraph_if_needed(G: nx.Graph,
                              metrics: Dict[str, Dict[str, float]],
                              threshold: int,
                              topn: int) -> Tuple[nx.Graph, bool, str]:
    n = G.number_of_nodes()
    if n <= threshold:
        return G, False, "none"

    if metrics and any("pagerank" in m for m in metrics.values()):
        pr_items = [(nid, metrics.get(nid, {}).get("pagerank", 0.0)) for nid in G.nodes()]
        pr_items.sort(key=lambda x: x[1], reverse=True)
        top_nodes = [nid for nid, _ in pr_items[:topn]]
        SG = G.subgraph(top_nodes).copy()
        if SG.number_of_nodes() > 0:
            return SG, True, "topN_pagerank"

    deg_sorted = sorted(G.degree, key=lambda kv: kv[1], reverse=True)[:topn]
    top_nodes = [nid for nid, _deg in deg_sorted]
    SG = G.subgraph(top_nodes).copy()
    return SG, True, "topN_degree"


def compute_layout(G: nx.Graph, seed: int) -> Dict[str, Tuple[float, float]]:
    n = G.number_of_nodes()
    if n == 0:
        return {}

    if n <= 50000:
        pos = nx.spring_layout(G, seed=seed, weight="weight", iterations=50, k=None)
        algo = "spring_layout"
    else:
        pos = nx.fruchterman_reingold_layout(G, seed=seed, weight="weight", iterations=50)
        algo = "fruchterman_reingold_layout"

    positions = {str(nid): (float(x), float(y)) for nid, (x, y) in pos.items()}
    return positions


def write_layout_ndjson(path: str, positions: Dict[str, Tuple[float, float]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for nid, (x, y) in positions.items():
            f.write(json.dumps({"node_id": nid, "x": x, "y": y}) + "\n")


def write_snapshot_csv(path: str,
                       positions: Dict[str, Tuple[float, float]],
                       metrics: Dict[str, Dict[str, float]]) -> None:
    rows = []
    for nid, (x, y) in positions.items():
        m = metrics.get(nid, {}) if metrics else {}
        rows.append({
            "node_id": nid,
            "x": x,
            "y": y,
            "degree": m.get("degree", np.nan),
            "pagerank": m.get("pagerank", np.nan),
            "betweenness": m.get("betweenness", np.nan),
            "clustering_coeff": m.get("clustering_coeff", np.nan),
        })

    df = pd.DataFrame(rows, columns=[
        "node_id", "x", "y", "degree", "pagerank", "betweenness", "clustering_coeff"
    ])
    df.to_csv(path, index=False)


def write_meta_json(path: str, meta: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()

    BASE = args.base
    INPUT_NODES = os.path.join(BASE, "NetworkConstruction", "network_nodes.ndjson")

    # 🔧 FIX: auto-detect all edge partitions
    INPUT_EDGES_DIR = os.path.join(BASE, "NetworkConstruction")
    edges_glob = sorted((Path(INPUT_EDGES_DIR)).glob("network_edges_*.csv"))
    INPUT_EDGES_PARTS = [str(p) for p in edges_glob]

    INPUT_METRICS = os.path.join(BASE, "NetworkAnalysis", "sna_metrics.csv")

    OUT_GRAPH_DIR = os.path.join(BASE, "NetworkGraphAnalysis")
    OUT_VIZ_DIR = os.path.join(BASE, "InteractiveVisualization")
    os.makedirs(OUT_GRAPH_DIR, exist_ok=True)
    os.makedirs(OUT_VIZ_DIR, exist_ok=True)

    now = datetime.datetime.now()
    MONTH_TAG = f"{now.year}_{now.month:02d}"
    OUT_NDJSON = os.path.join(OUT_GRAPH_DIR, f"graph_layout_{MONTH_TAG}.ndjson")
    OUT_CSV = os.path.join(OUT_VIZ_DIR, f"network_snapshot_{MONTH_TAG}.csv")
    OUT_META = os.path.join(OUT_GRAPH_DIR, f"layout_meta_{MONTH_TAG}.json")

    t0 = time.time()

    node_ids = stream_node_ids(INPUT_NODES)
    if not node_ids:
        raise RuntimeError(f"No nodes found. Check file: {INPUT_NODES}")

    metrics = maybe_load_metrics(INPUT_METRICS)

    if not INPUT_EDGES_PARTS:
        raise RuntimeError(f"No edge CSVs found under {INPUT_EDGES_DIR} (expected network_edges_*.csv)")

    G = stream_edges_to_graph(INPUT_EDGES_PARTS, node_filter=node_ids)

    G_final, used_sampling, sampling_method = sample_subgraph_if_needed(
        G, metrics, threshold=args.subgraph_threshold, topn=args.topn
    )

    positions = compute_layout(G_final, seed=args.seed)

    write_layout_ndjson(OUT_NDJSON, positions)
    write_snapshot_csv(OUT_CSV, positions, metrics)

    meta = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "input_nodes_file": INPUT_NODES,
        "input_edges_files": INPUT_EDGES_PARTS,
        "input_metrics_file": INPUT_METRICS if os.path.exists(INPUT_METRICS) else None,
        "node_count_total": int(G.number_of_nodes()),
        "edge_count_total": int(G.number_of_edges()),
        "node_count_final": int(G_final.number_of_nodes()),
        "edge_count_final": int(G_final.number_of_edges()),
        "used_sampling": bool(used_sampling),
        "sampling_method": sampling_method,
        "layout_algorithm": "spring_layout" if G_final.number_of_nodes() <= 50000
                             else "fruchterman_reingold_layout",
        "seed": args.seed,
        "outputs": {
            "layout_ndjson": OUT_NDJSON,
            "snapshot_csv": OUT_CSV,
            "meta_json": OUT_META
        }
    }
    write_meta_json(OUT_META, meta)

    elapsed = round(time.time() - t0, 3)
    print("\n=== Network Graph Analysis Demo Complete ===")
    print(f"Nodes (total/final): {G.number_of_nodes()} / {G_final.number_of_nodes()}")
    print(f"Edges (total/final): {G.number_of_edges()} / {G_final.number_of_edges()}")
    print(f"Used sampling     : {used_sampling} ({sampling_method})")
    print(f"Layout seed       : {args.seed}")
    print(f"Output NDJSON     : {OUT_NDJSON}")
    print(f"Output CSV        : {OUT_CSV}")
    print(f"Output META       : {OUT_META}")
    print(f"Elapsed           : {elapsed} sec\n")


if __name__ == "__main__":
    main()
