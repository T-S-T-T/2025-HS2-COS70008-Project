import json
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.linear_model import SGDClassifier

# ——— CONFIGURATION ——————————————————————————

BASE_DIR           = Path(__file__).resolve().parent.parent
EMAIL_DIR          = BASE_DIR / "data" / "SentimentalAnalysis"
METRICS_PATH       = BASE_DIR / "data" / "NetworkAnalysis" / "sna_metrics.csv"
NODE_PATH          = BASE_DIR / "data" / "NetworkConstruction" / "network_nodes.ndjson"
EDGE_DIR           = BASE_DIR / "data" / "NetworkConstruction"
OUTPUT_DIR         = BASE_DIR / "data" / "OrganizationalInsight"

CHUNK_SIZE         = 10_000
DBSCAN_EPS         = 0.5
DBSCAN_MIN_SAMPLES = 5

# Models for incremental burnout prediction
sgd_model      = SGDClassifier(loss="log_loss", max_iter=1, warm_start=True)
first_sgd_fit  = True
xgb_model      = None

# ——— LOAD GLOBAL METRICS ——————————————————————

df_global = pd.read_csv(METRICS_PATH, index_col="node_id")

# ——— COMMUNITY DETECTION ——————————————————————

# Build full undirected graph for Louvain
G_full = nx.Graph()
with open(NODE_PATH, "r", encoding="utf-8") as fh:
    for line in fh:
        nid = json.loads(line)["node_id"]
        G_full.add_node(nid)

for ef in sorted(EDGE_DIR.glob("network_edges_*.csv")):
    df_e = pd.read_csv(ef)
    for _, r in df_e.iterrows():
        G_full.add_edge(r.source, r.target, weight=r.weight)

partition = community_louvain.best_partition(G_full.to_undirected())
num_communities = len(set(partition.values()))

# Determine top influencers by PageRank
pr_thresh       = df_global["pagerank"].quantile(0.90)
top_influencers = set(df_global[df_global["pagerank"] > pr_thresh].index)

# ——— MAIN WORKFLOW ——————————————————————————————————

def main():
    # allow assignment to module-level variables
    global xgb_model, first_sgd_fit

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "total_anomalies": 0,
        "communities":     num_communities,
        "high_burnout":    0,
        "top_influencers": list(top_influencers)
    }

    for email_csv in sorted(EMAIL_DIR.glob("enriched_emails_*.csv")):
        month         = email_csv.stem.replace("enriched_emails_", "")
        insights_path = OUTPUT_DIR / f"insights_{month}.csv"

        # 1. Feature engineering: per-sender volume & avg sentiment
        feats = {}
        for chunk in pd.read_csv(email_csv, chunksize=CHUNK_SIZE):
            grp = chunk.groupby("sender")["compound"] \
                       .agg(volume="count", avg_sentiment="mean") \
                       .reset_index()

            for row in grp.itertuples(index=False):
                sid, vol, avg_s = row.sender, int(row.volume), float(row.avg_sentiment)
                prev = feats.get(sid)
                if prev is None:
                    feats[sid] = {"volume": vol, "avg_sentiment": avg_s}
                else:
                    tot          = prev["volume"] + vol
                    combined_avg = (prev["avg_sentiment"] * prev["volume"] + avg_s * vol) / tot
                    feats[sid]   = {"volume": tot, "avg_sentiment": combined_avg}

        df_feat = (
            pd.DataFrame.from_dict(feats, orient="index")
              .rename_axis("node_id")
              .reset_index()
        )

        # 2. Merge with global SNA metrics
        df_merged = df_feat.merge(
            df_global, how="left", left_on="node_id", right_index=True
        ).fillna(0)

        # 3. Anomaly detection (IsolationForest)
        iso    = IsolationForest(contamination=0.05, random_state=42)
        X_iso  = df_merged[[
            "volume", "avg_sentiment",
            "degree", "betweenness",
            "clustering_coeff", "pagerank"
        ]]
        iso.fit(X_iso)
        df_merged["anomaly_score"] = iso.decision_function(X_iso)
        df_merged["anomaly_label"] = iso.predict(X_iso)
        summary["total_anomalies"] += int((df_merged["anomaly_label"] == -1).sum())

        # 4. DBSCAN outliers
        db    = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
        X_db  = df_merged[["volume", "avg_sentiment"]]
        df_merged["dbscan_label"] = db.fit_predict(X_db)

        # 5. Community & influence
        df_merged["community_id"]   = df_merged["node_id"] \
                                        .map(partition) \
                                        .fillna(-1) \
                                        .astype(int)
        df_merged["influence_flag"] = df_merged["node_id"].isin(top_influencers)

        # 6. Burnout prediction (SGDClassifier + XGBoost)
        feature_cols = [
            "volume", "avg_sentiment",
            "degree", "betweenness",
            "clustering_coeff", "pagerank",
            "anomaly_score"
        ]
        X_feat   = df_merged[feature_cols].values
        y_target = (df_merged["anomaly_label"] == -1).astype(int).values

        # 6a. LogisticRegression via partial_fit
        if first_sgd_fit:
            sgd_model.partial_fit(X_feat, y_target, classes=[0, 1])
            first_sgd_fit = False
        else:
            sgd_model.partial_fit(X_feat, y_target)
        prob_lr = sgd_model.predict_proba(X_feat)[:, 1]

        # 6b. XGBoost incremental training
        if xgb_model is None:
            xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
            xgb_model.fit(X_feat, y_target)
        else:
            xgb_model.fit(X_feat, y_target, xgb_model=xgb_model)
        prob_xgb = xgb_model.predict_proba(X_feat)[:, 1]

        # Combine probabilities
        df_merged["burnout_prob"]  = (prob_lr + prob_xgb) / 2.0
        df_merged["burnout_label"] = (df_merged["burnout_prob"] > 0.5).astype(int)
        summary["high_burnout"]   += int(df_merged["burnout_label"].sum())

        # 7. Write monthly insights
        out_cols = [
            "node_id", "anomaly_score", "dbscan_label",
            "community_id", "influence_flag",
            "burnout_prob", "burnout_label"
        ]
        df_merged[out_cols].to_csv(insights_path, index=False)
        print(f"[DONE] Insights for {month} → {insights_path}")

    # 8. Write summary JSON
    with open(OUTPUT_DIR / "insight_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"[DONE] Summary → {OUTPUT_DIR/'insight_summary.json'}")

if __name__ == "__main__":
    print("\nRunning...\n")
    main()
    print("\nFinished!\n")