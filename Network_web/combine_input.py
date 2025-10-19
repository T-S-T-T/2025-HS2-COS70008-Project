import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()   # folder where this script is located
BASE_ROOT = PROJECT_ROOT.parent   # move one more folder up from PROJECT_ROOT
DATA_ROOT = BASE_ROOT / "data"   # data folder

OUTPUT_DIR = DATA_ROOT / "combine_input"   # data
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)   # create it if it doesn’t exist

# === Paths ===
NETWORK_CONSTRUCTION = DATA_ROOT / "NetworkConstruction"
NETWORK_ANALYSIS = DATA_ROOT / "NetworkAnalysis"

# === Combine all edge partitions into one file ===
edge_files = list(NETWORK_CONSTRUCTION.glob("network_edges_*.csv"))
if not edge_files:
    raise FileNotFoundError("No edge partition files found in NetworkConstruction folder!")

print(f"Found {len(edge_files)} edge files, merging...")

edge_list = []
for f in edge_files:
    df = pd.read_csv(f)

    # Try to extract year/month from filename
    ym = f.stem.replace("network_edges_", "")
    try:
        year, month = ym.split("_")
        df["year"] = int(year)
        df["month"] = int(month)
    except ValueError:
        print(f"[WARN] Could not parse year/month from filename: {f.name}")
        df["year"], df["month"] = None, None

    # Ensure columns are consistent
    if "mean_compound" in df.columns:
        df = df.rename(columns={"mean_compound": "sentiment_avg"})
    elif "sentiment_avg" not in df.columns:
        df["sentiment_avg"] = 0

    # Clean
    df = df.dropna(subset=["source", "target"])
    df = df[df["source"].astype(str).str.strip() != ""]
    df = df[df["target"].astype(str).str.strip() != ""]

    # Keep only relevant columns
    keep_cols = ["source", "target", "weight", "sentiment_avg", "year", "month"]
    df = df[keep_cols]
    edge_list.append(df)

# Combine all edges
edges = pd.concat(edge_list, ignore_index=True)
edges_out = OUTPUT_DIR / "edges.csv"
edges.to_csv(edges_out, index=False)
print(f"Saved combined edges → {edges_out} ({len(edges):,} rows)")


# === Prepare nodes.csv from sna_metrics ===
nodes_file = NETWORK_ANALYSIS / "sna_metrics.csv"
if not nodes_file.exists():
    raise FileNotFoundError("sna_metrics.csv not found in NetworkAnalysis folder!")

nodes = pd.read_csv(nodes_file)

# Clean
nodes = nodes.dropna(subset=["node_id"])
nodes = nodes[nodes["node_id"].astype(str).str.strip() != ""]

# Add year/month if not present
if "year" not in nodes.columns or "month" not in nodes.columns:
    nodes["year"], nodes["month"] = None, None

# Unified degree = indegree + outdegree
if "indegree" in nodes.columns and "outdegree" in nodes.columns:
    nodes["degree"] = nodes["indegree"].fillna(0) + nodes["outdegree"].fillna(0)
elif "degree" not in nodes.columns:
    nodes["degree"] = 0

# Keep relevant columns
keep_cols = ["node_id", "degree", "pagerank", "betweenness", "clustering_coeff", "year", "month"]
existing_cols = [c for c in keep_cols if c in nodes.columns]
nodes = nodes[existing_cols]

# Save
nodes_out = OUTPUT_DIR / "nodes.csv"
nodes.to_csv(nodes_out, index=False)
print(f"Saved cleaned nodes → {nodes_out} ({len(nodes):,} nodes)")

print("\n All combined files are ready in /data/combine_input/")