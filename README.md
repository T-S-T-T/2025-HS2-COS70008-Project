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
Purpose: Parse raw email files into unified, machine-readable records.
Functions:
Traverse folder and read each email file.
Parse RFC-822 headers and extract fields (ID, date, from, to, cc, bcc, subject).
Extract and clean body text.
Normalize timestamps and address lists.
Export records into tabular form.


Python libraries:
os, glob
email (stdlib)
pandas
datetime, regex


Input files:
Folder raw_emails/ containing many .txt or .eml files, each with full headers and body.


Output files:

processed_emails.csv
Columns: message_id, date, sender, recipients (semicolon-delimited), cc (semicolon-delimited), bcc (semicolon-delimited), subject, body

email_index.json
Mapping: message_id → original filename and folder path



2. Sentiment Analysis
Purpose: Enrich each email with standardized sentiment scores.
Functions:
Load processed_emails.csv.
Apply VADER (or equivalent) to body.
Compute compound, neg, neu, pos scores.
Assign a sentiment label (e.g., positive/neutral/negative).


Python libraries:
pandas
nltk.sentiment.vader or vaderSentiment
numpy


Input files:
processed_emails.csv


Output files:

sentiment_scores.csv
Columns: message_id, compound, neg, neu, pos, sentiment_label

enriched_emails.csv
All columns from processed_emails.csv + appended sentiment columns



3. Network Construction
Purpose: Build a graph of who communicates with whom, capturing intensity.
Functions:
Read enriched_emails.csv.
Create node list from unique senders and recipients.
Generate edges (sender → each recipient/cc/bcc).
Compute edge weights as count of emails.
Serialize nodes and edges for downstream use.


Python libraries:
pandas
networkx
json


Input files:
enriched_emails.csv


Output files:

network_nodes.csv
Columns: node_id (email), name (if available)

network_edges.csv
Columns: source (email), target (email), weight

network_graph.json
JSON with nodes and edges arrays



4. Network Analysis
Purpose: Compute core social-network metrics for each participant.
Functions:
Load graph from network_graph.json (or CSVs).
Calculate degree centrality, betweenness, clustering coefficient, PageRank, and network density.
Export metrics to support anomaly detection and influence mapping.


Python libraries:
networkx
pandas


Input files:
network_graph.json


Output files:

sna_metrics.csv
Columns: node_id, degree, betweenness, clustering_coeff, pagerank, network_density



5. Network Graph Analysis
Purpose: Prepare Python scripts for embedding an interactive graph in Power BI.
Functions:
Load network_nodes.csv and network_edges.csv.
Compute layout (e.g., spring layout).
Generate plotting code using Plotly or Matplotlib.
Package as a .py script ready for Power BI’s Python visual.


Python libraries:
networkx
plotly or matplotlib
json


Input files:
network_nodes.csv
network_edges.csv
(Optional) sna_metrics.csv for styling


Output files:

graph_visualization.py
Contains code to read CSVs, compute layout, and render the network

graph_layout.json (optional)
Precomputed node positions



6. Organizational Insights
Purpose: Use ML and graph algorithms to uncover anomalies, silos, influence, and burnout risk.
Functions:
Merge enriched_emails.csv and sna_metrics.csv.
Anomaly detection via Isolation Forest and DBSCAN.
Community detection with Louvain algorithm.
Influence mapping via PageRank thresholds.
Burnout prediction using XGBoost and logistic regression on communication features.
Summarize node-level insights.


Python libraries:
pandas
scikit-learn
python-louvain (community)
xgboost
networkx
numpy


Input files:
enriched_emails.csv
sna_metrics.csv


Output files:

insights.csv
Columns: node_id, anomaly_score, community_id, influence_score, burnout_prob, burnout_label

insight_summary.json
Aggregate stats (e.g., number of anomalies, top 10 influencers, high-risk burnout count)



7. Interactive Visualization
Purpose: Produce final data bundles and Python scripts for Power BI dashboards.
Functions:
Aggregate sentiment_scores.csv, sna_metrics.csv, and insights.csv.
Create chart objects (time series, bar charts, network snapshots) via Plotly or Seaborn.
Bundle data and visualization code into files that Power BI can ingest as Python visuals.


Python libraries:
pandas
plotly
seaborn


Input files:
sentiment_scores.csv
sna_metrics.csv
insights.csv


Output files:

dashboard_data.csv
Combined table: one row per node_id with sentiment, SNA metrics, insights

powerbi_dashboard.py
Python script defining all visuals; reads dashboard_data.csv and renders charts

visual_config.json (optional)
Layout settings and filter defaults for Power BI

## Draft 2

1. Data Processing
Purpose: Parse 2.3 GB of raw maildir files into partitioned, tabular records to minimize in-memory footprint.

Input files:
data/maildir/ (extension-less email blobs; arbitrary nesting)

Output files:
processed_emails_{YYYY_MM}.csv
Columns: message_id, date, sender, recipients, cc, bcc, subject, body
One file per month (e.g. processed_emails_2000_07.csv), ~10–100 MB each

email_index.ndjson
Newline-delimited JSON; each line: {"message_id":"...", "path":"relative/path/to/file"}

Memory strategy:
Stream rglob + parser;
Immediately append each record to its month’s CSV;
Emit index entries as individual JSON lines.


2. Sentiment Analysis
Purpose: Enrich each parsed email with VADER scores in a streaming, chunked workflow.

Input files:
processed_emails_{YYYY_MM}.csv

Output files:
sentiment_scores_{YYYY_MM}.csv
Columns: message_id, compound, neg, neu, pos, sentiment_label

enriched_emails_{YYYY_MM}.csv
All original columns + appended sentiment columns

Memory strategy:
Read source CSV in chunks (e.g. chunksize=10_000);
Compute scores on each chunk;
Append to output files without ever loading an entire month in RAM.

3. Network Construction
Purpose: Build node and edge lists without materializing the full graph in memory.

Input files:
enriched_emails_{YYYY_MM}.csv

Output files:
network_nodes.ndjson
Each line: {"node_id":"email@example.com", "name":""}

network_edges_{A-Z0-9_}.csv
Partitioned by first character of source (e.g. network_edges_A.csv)
Columns: source, target, weight

network_meta.json
Metadata: list of edge partitions, total nodes, total edge weight

Memory strategy:
Stream each monthly file;
Maintain an on-disk CSV per partition;
Emit unique nodes periodically to NDJSON.

4. Network Analysis
Purpose: Compute core SNA metrics by streaming in edge partitions.

Input files:
network_nodes.ndjson
network_edges_{*}.csv

Output files:
sna_metrics.csv
Columns: node_id, degree, betweenness, clustering_coeff, pagerank, density

network_density.json
Single value: global network density

Memory strategy:
Load node list (small) into NetworkX;
Iterate over edge partitions one at a time to build or update the graph;
Compute and write metrics in one pass.

5. Network Graph Analysis
Purpose: Prepare lightweight layout and Python code for Power BI visuals without dumping the entire graph.

Input files:
network_nodes.ndjson

network_edges_{*}.csv
(Optional) sna_metrics.csv

Output files:

graph_visualization.py
Reads partitioned NDJSON/CSVs to render network via Plotly or Matplotlib

graph_layout.ndjson
Each line: {"node_id":"...", "x":float, "y":float}

Memory strategy:
Compute layout on subgraphs or via iterative coordinate updates;
Stream results directly to NDJSON.

6. Organizational Insights
Purpose: Derive ML-driven insights using chunked feature sets to avoid memory spikes.

Input files:
enriched_emails_{YYYY_MM}.csv

sna_metrics.csv
Output files:

insights_{YYYY_MM}.csv
Columns: node_id, anomaly_score, community_id, influence_score, burnout_prob, burnout_label

insight_summary.json
Aggregate stats: e.g. total_anomalies, top_influencers, high_burnout_count

Memory strategy:
Process each month’s enriched file in chunks;
Incrementally update Isolation Forest, DBSCAN, Louvain, XGBoost;
Append per-month insights to partitioned CSVs.

7. Interactive Visualization
Purpose: Produce partitioned data bundles and config for Power BI Python visuals.

Input files:
sentiment_scores_{YYYY_MM}.csv

sna_metrics.csv
insights_{YYYY_MM}.csv

Output files:
dashboard_data_{YYYY_MM}.csv
Combined per-node table for each month

visual_config.json
Filter defaults, layout settings

powerbi_dashboard.py
Python script reading month-files and rendering Plotly/Seaborn charts

Memory strategy:
Merge chunked inputs per partition on disk;
Keep each dashboard CSV small enough for streaming into Power BI.