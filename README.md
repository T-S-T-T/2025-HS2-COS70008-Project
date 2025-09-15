# 2025-HS2-COS70008-Project
SNA project for Unit COS70008 using Python

The project consists of 7 modules

## Data Processing

## Sentimental Analysis

## Network Construction

## Network Analysis

## Network Graph Analysis

## Organizational Insights

## Interactive Visualization


## Draft
1. Data Processing
Purpose: Stream and parse 2.3 GB of raw Maildir email blobs into monthly CSVs and a line-delimited index without ever loading all files into RAM.

Functions:
Recursively traverse data/maildir/ and treat every file as an email.
Parse RFC-822 headers (Message-ID, Date, From, To, Cc, Bcc, Subject).
Extract and concatenate text/plain body parts.
Normalize dates to ISO 8601 and addresses to semicolon-delimited lists.
Partition output by month, appending each record to processed_emails_{YYYY_MM}.csv.
Emit a newline-delimited JSON index mapping message_id → relative path.

Python libraries:
os, pathlib
email (stdlib)
csv, json
datetime, email.utils, regex

Input files:
data/maildir/ (arbitrary nesting; each file is an email) Memory: streaming via rglob, one file at a time.

Output files (under data/DataProcessing/):
processed_emails_{YYYY_MM}.csv
Columns: message_id, date, sender, recipients, cc, bcc, subject, body
One file per month (e.g. 2000_07), ~10–100 MB each

email_index.ndjson
Each line: {"message_id":"<..>", "path":"maildir/.../filename"}

Memory strategy:
Open each monthly CSV in append mode, write header only once.
Write one JSON line per email to index file.
Never accumulate more than one record in memory.


2. Sentiment Analysis
Purpose: Compute VADER sentiment scores in a chunked fashion and produce enriched partitioned outputs.

Functions:
Read each processed_emails_{YYYY_MM}.csv in chunks.
Apply VADER to body text to get compound, neg, neu, pos scores.
Assign a sentiment label based on compound thresholds.
Append scores to sentiment_scores_{YYYY_MM}.csv.
Merge original columns + sentiment into enriched_emails_{YYYY_MM}.csv.

Python libraries:
pandas
nltk.sentiment.vader or vaderSentiment
numpy

Input files:
data/DataProcessing/processed_emails_{YYYY_MM}.csv Memory: pd.read_csv(..., chunksize=10_000).

Output files (under data/SentimentalAnalysis/):
sentiment_scores_{YYYY_MM}.csv
Columns: message_id, compound, neg, neu, pos, sentiment_label

enriched_emails_{YYYY_MM}.csv
All original columns + the five sentiment columns

Memory strategy:
Process one chunk at a time, write each chunk’s results to CSV in append mode.
Discard chunk after writing.


3. Network Construction
Purpose: Stream enriched emails to build node and edge partitions for large-scale graph construction.

Functions:
Read enriched_emails_{YYYY_MM}.csv by chunk.
Emit unique nodes (senders + recipients + cc + bcc) to network_nodes.ndjson.
For each chunk, generate edges (sender → each recipient/cc/bcc) with weight=1.
Aggregate edge counts by partitioning on first character of source email.
Write partitioned CSVs: network_edges_{partition}.csv.
Record graph metadata (list of partitions) to network_meta.json.

Python libraries:
pandas
networkx (optional for validation)
json

Input files:
data/SentimentalAnalysis/enriched_emails_{YYYY_MM}.csv Memory: chunked read, one partition file open per letter/digit.

Output files (under data/NetworkConstruction/):
network_nodes.ndjson
Each line: {"node_id":"email@example.com","name":""}

network_edges_{A-Z0-9}.csv
Columns: source, target, weight (incremental counts)

network_meta.json
Fields: {"partitions":[...],"total_nodes":N,"total_edges":M}

Memory strategy:
Stream each month’s enriched file, update counts in small in-memory dicts per partition, flush periodically.
Emit nodes in NDJSON as discovered (use a small in-memory set to dedupe within a run).

4. Network Analysis
Purpose: Incrementally construct the full graph and compute core SNA metrics without loading all edge data at once.

Functions:
Load network_nodes.ndjson (small) into NetworkX graph nodes.
For each network_edges_{p}.csv partition:
Stream rows and add weighted edges to the graph.
After graph assembly, compute:
Degree centrality
Betweenness centrality
Clustering coefficient
PageRank
Network density
Write node metrics to sna_metrics.csv.
Write global density to network_density.json.

Python libraries:
networkx
pandas
json

Input files:
data/NetworkConstruction/network_nodes.ndjson
data/NetworkConstruction/network_edges_{*}.csv

Output files (under data/NetworkAnalysis/):
sna_metrics.csv
Columns: node_id, degree, betweenness, clustering_coeff, pagerank, density

network_density.json
{"density": float}

Memory strategy:
Graph object kept in memory (fits if edge count <1 M).
Edge partitions streamed one at a time.


5. Network Graph Analysis
Purpose: Compute and store lightweight layout coordinates and package a Python script for Power BI.

Functions:
Stream network_nodes.ndjson to collect node IDs.
Optionally stream sna_metrics.csv for styling metadata.
Read edge partitions one by one to build a sparse adjacency for layout.
Compute positions via NetworkX’s spring_layout or iterative Barnes-Hut on partitions.
Emit graph_layout.ndjson: each line {"node_id":"...","x":X,"y":Y}.
Generate graph_visualization.py, a self-contained Python script that:
Loads partitioned nodes, edges, and layout.
Renders an interactive Plotly network.

Python libraries:
networkx
plotly or matplotlib
json

Input files:
data/NetworkConstruction/network_nodes.ndjson
data/NetworkConstruction/network_edges_{*}.csv
data/NetworkAnalysis/sna_metrics.csv (optional)

Output files (under data/NetworkGraphAnalysis/):

graph_layout.ndjson
graph_visualization.py

Memory strategy:
Build and discard adjacency from each edge partition.
Compute layout incrementally or on a downsampled subgraph if node count is very large.

6. Organizational Insights
Purpose: Derive anomaly scores, detect silos/communities, map influence, and predict burnout in a streaming, partitioned ML workflow.

Functions:
Stream-merge monthly enriched_emails_{YYYY_MM}.csv and sna_metrics.csv by node_id.
Feature engineering: communication volume, centrality, sentiment aggregates.
Anomaly detection: train and apply Isolation Forest on feature windows per month.
Clustering: apply DBSCAN to detect outlier communication patterns.
Community detection: use Louvain algorithm on the full graph to assign community_id.
Influence mapping: threshold PageRank to flag top influencers.
Burnout prediction: train XGBoost and Logistic Regression on historical features; output probability.
Write per-month insights to insights_{YYYY_MM}.csv.
Summarize global stats to insight_summary.json.

Python libraries:
pandas
scikit-learn (IsolationForest, DBSCAN, LogisticRegression)
python-louvain
xgboost
networkx
numpy
json

Input files:
data/SentimentalAnalysis/enriched_emails_{YYYY_MM}.csv
data/NetworkAnalysis/sna_metrics.csv

Output files (under data/OrganizationalInsight/):
insights_{YYYY_MM}.csv
Columns: node_id, anomaly_score, dbscan_label, community_id, influence_flag, burnout_prob, burnout_label

insight_summary.json
e.g. {"total_anomalies":X,"communities":Y,"high_burnout":Z,"top_influencers": […]}

Memory strategy:
Chunked feature assembly per month.
Incremental model updates: partial_fit for IsolationForest and logistic regression.
Community detection once on full graph (in memory).
Write each month’s insights then free memory.

7. Interactive Visualization
Purpose: Stream-partitioned data and scripts for Power BI Python visuals without overflowing RAM.

Functions:
Load sentiment_scores_{YYYY_MM}.csv, sna_metrics.csv, and insights_{YYYY_MM}.csv in chunks.
Aggregate per-node time-series, summary tables, and network snapshots per month.
Generate Plotly/Seaborn chart definitions in powerbi_dashboard.py.
Emit dashboard_data_{YYYY_MM}.csv combining node_id with all metrics for that month.
Create visual_config.json for filter defaults and layout settings.

Python libraries:
pandas
plotly
seaborn
json

Input files:
data/SentimentalAnalysis/sentiment_scores_{YYYY_MM}.csv
data/NetworkAnalysis/sna_metrics.csv
data/OrganizationalInsight/insights_{YYYY_MM}.csv

Output files (under data/InteractiveVisualization/):
dashboard_data_{YYYY_MM}.csv
Columns: node_id, compound, degree, anomaly_score, burnout_prob, etc.

powerbi_dashboard.py

visual_config.json

Memory strategy:
Generate each month’s dashboard CSV independently with chunksize.
Keep chart-generation code streaming, reading only required slices of CSV.
Leverage Power BI’s paging to load visuals on demand.