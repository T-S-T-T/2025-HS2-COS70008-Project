"""
Network Analysis — spec-compliant & backward-compatible

Run from repo root:
    python NetworkAnalysis\analyze_network.py
or:
    python -m NetworkAnalysis.analyze_network
"""

import os, json, glob, time
from typing import Dict
import pandas as pd
import networkx as nx

# ----- folders -----
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
NC_DIR = os.path.join(DATA, "NetworkConstruction")
NA_DIR = os.path.join(DATA, "NetworkAnalysis")
os.makedirs(NA_DIR, exist_ok=True)

# spec inputs
NODES_NDJSON = os.path.join(NC_DIR, "network_nodes.ndjson")
EDGES_GLOB   = os.path.join(NC_DIR, "network_edges_*.csv")
# legacy fallback inputs (your old demo)
NODES_CSV_FALLBACK = os.path.join(DATA, "nodes.csv")
EDGES_CSV_FALLBACK = os.path.join(DATA, "edges.csv")

# outputs (per spec)
SNA_METRICS_CSV = os.path.join(NA_DIR, "sna_metrics.csv")
DENSITY_JSON    = os.path.join(NA_DIR, "network_density.json")

# knobs
BETWEENNESS_SAMPLE_EDGE_CUTOFF = 100_000   # was 800_000
BETWEENNESS_SAMPLE_MAX_NODES   = 1000      # was 2000 (even faster)
CLUSTERING_EDGE_CUTOFF         = 400_000   # optional: skip clustering earlier



def load_nodes_ndjson(path: str) -> Dict[str, dict]:
    """Read NDJSON lines with 'node_id'."""
    nodes = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            nid = obj.get("node_id") or obj.get("id")
            if nid:
                nodes[str(nid)] = {k: v for k, v in obj.items() if k not in ("node_id", "id")}
    return nodes


def add_edges_from_partition(G: nx.DiGraph, csv_path: str, chunksize: int = 250_000) -> int:
    """Stream a partition CSV and add weighted edges to G. Returns rows read."""
    total = 0
    usecols = ["source", "target", "weight"]
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        chunk["source"] = chunk["source"].astype(str)
        chunk["target"] = chunk["target"].astype(str)
        chunk["weight"] = pd.to_numeric(chunk["weight"], errors="coerce").fillna(1).astype(int)
        # aggregate within chunk to reduce add_edge calls
        agg = chunk.groupby(["source", "target"], as_index=False)["weight"].sum()
        for s, t, w in agg.itertuples(index=False):
            if G.has_edge(s, t):
                G[s][t]["weight"] += int(w)
            else:
                G.add_edge(s, t, weight=int(w))
        total += len(chunk)
    return total


def build_graph_from_partitions() -> nx.DiGraph:
    nodes_attrs = load_nodes_ndjson(NODES_NDJSON)
    G = nx.DiGraph()
    # add nodes first so isolates are kept
    for nid, attrs in nodes_attrs.items():
        G.add_node(nid, **attrs)
    parts = sorted(glob.glob(EDGES_GLOB))
    if not parts:
        raise FileNotFoundError(f"No edge partitions found matching: {EDGES_GLOB}")
    rows = 0
    for p in parts:
        print(f"[stream] {os.path.basename(p)}")
        rows += add_edges_from_partition(G, p)
    print(f"[info] streamed rows: {rows:,} |V|={G.number_of_nodes():,} |E|={G.number_of_edges():,}")
    return G


def build_graph_from_legacy_csv() -> nx.DiGraph:
    nodes = pd.read_csv(NODES_CSV_FALLBACK)
    edges = pd.read_csv(EDGES_CSV_FALLBACK)
    if "weight" not in edges.columns:
        edges["weight"] = 1
    G = nx.DiGraph()
    for _, r in nodes.iterrows():
        G.add_node(str(r["id"]), **{c: r[c] for c in nodes.columns if c != "id"})
    for _, r in edges.iterrows():
        G.add_edge(str(r["source"]), str(r["target"]), weight=int(r.get("weight", 1)))
    print(f"[info] legacy graph |V|={G.number_of_nodes():,} |E|={G.number_of_edges():,}")
    return G


def compute_metrics(G: nx.DiGraph) -> pd.DataFrame:
    """
    Spec metrics:
      - indegree, outdegree (UNWEIGHTED counts)
      - betweenness (sampled on huge graphs)
      - clustering coefficient (skip if graph enormous)
      - PageRank (weighted)
    """
    t0 = time.time()
    UG = G.to_undirected()
    n, m = G.number_of_nodes(), G.number_of_edges()
    print(f"[metrics] start |V|={n:,} |E|={m:,}")

    indegree  = dict(G.in_degree(weight=None))
    outdegree = dict(G.out_degree(weight=None))
    print("[metrics] degrees done")

    pagerank = nx.pagerank(G, weight="weight", alpha=0.85, tol=1e-6, max_iter=200)
    print("[metrics] pagerank done")

    if m > BETWEENNESS_SAMPLE_EDGE_CUTOFF or n > 100_000:
        k = min(BETWEENNESS_SAMPLE_MAX_NODES, n)
        print(f"[metrics] betweenness sampled (k={k})")
        betw = nx.betweenness_centrality(UG, k=k, seed=42)
    else:
        print("[metrics] betweenness exact")
        betw = nx.betweenness_centrality(UG)
    print("[metrics] betweenness done")

    if m <= CLUSTERING_EDGE_CUTOFF:
        clus = nx.clustering(UG)
        print("[metrics] clustering done")
    else:
        clus = {v: 0.0 for v in UG.nodes()}
        print("[metrics] clustering skipped (graph too large)")

    rows = []
    for v in G.nodes():
        rows.append({
            "node_id": v,
            "indegree": int(indegree.get(v, 0)),
            "outdegree": int(outdegree.get(v, 0)),
            "betweenness": float(betw.get(v, 0.0)),
            "clustering_coeff": float(clus.get(v, 0.0)),
            "pagerank": float(pagerank.get(v, 0.0)),
        })
    print(f"[metrics] all done in {time.time()-t0:.1f}s")
    return pd.DataFrame(rows).sort_values("pagerank", ascending=False, ignore_index=True)


def save_outputs(df: pd.DataFrame, G: nx.DiGraph) -> None:
    df.to_csv(SNA_METRICS_CSV, index=False)
    dens = float(nx.density(G)) if G.number_of_nodes() > 1 else 0.0
    with open(DENSITY_JSON, "w", encoding="utf-8") as f:
        json.dump({"density": dens}, f, indent=2)
    print(f"[ok] wrote {SNA_METRICS_CSV}")
    print(f"[ok] wrote {DENSITY_JSON} (density={dens:.6f})")


def main():
    # choose inputs automatically
    if os.path.exists(NODES_NDJSON) and glob.glob(EDGES_GLOB):
        print("[1/4] using NetworkConstruction inputs (streaming partitions)")
        G = build_graph_from_partitions()
    elif os.path.exists(NODES_CSV_FALLBACK) and os.path.exists(EDGES_CSV_FALLBACK):
        print("[1/4] using legacy CSV inputs (data/nodes.csv, data/edges.csv)")
        G = build_graph_from_legacy_csv()
    else:
        raise FileNotFoundError(
            "No inputs found.\n"
            f"Expected either:\n"
            f"  - {NODES_NDJSON} + {EDGES_GLOB}\n"
            f"or\n"
            f"  - {NODES_CSV_FALLBACK} + {EDGES_CSV_FALLBACK}"
        )

    print("[2/4] computing metrics")
    df = compute_metrics(G)

    print("[3/4] saving outputs")
    save_outputs(df, G)

    print("[DONE] outputs in data/NetworkAnalysis/")

if __name__ == "__main__":
    main()
