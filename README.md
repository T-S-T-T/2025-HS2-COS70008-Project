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
Purpose: Stream and parse raw Maildir emails into monthly CSV partitions and a line-delimited index for downstream modules.

Functions
Recursively traverse data/maildir/; treat every file as an email blob.

Parse RFC-822 headers: Message-ID, Date, From, To, Cc, Bcc, Subject.

Extract and concatenate text/plain body parts.

Normalize dates to ISO 8601; normalize address lists (semicolon-delimited).

Partition by email date (YYYY_MM) and append each record to processed_emails_{YYYY_MM}.csv.

Emit one JSON line per email to email_index.ndjson (message_id → relative path).

Python libraries
pathlib, os

email (stdlib), email.utils

csv, json

datetime, regex

Input files
Folder: data/maildir/ • Structure: arbitrary nesting; each file is one email • Memory: streaming via Path.rglob()

Output files (under data/DataProcessing/)
processed_emails_{YYYY_MM}.csv • Format: CSV; headers = message_id,date,sender,recipients,cc,bcc,subject,body • Partition: one file per month (e.g. processed_emails_2000_07.csv) • Size: ~10–100 MB each

email_index.ndjson • Format: newline-delimited JSON; each line = {"message_id":"<..>","path":"maildir/.../file"}

Memory strategy
File-by-file streaming: only one email is parsed in memory at a time.

Append-only writes: open each monthly CSV in append mode; write header once.

NDJSON index: write one line per email; no accumulation in RAM.

2. Sentiment Analysis
Purpose: Enrich parsed emails with VADER sentiment scores via chunked processing.

Functions
Read each processed_emails_{YYYY_MM}.csv in fixed-size chunks (e.g. 10 000 rows).

Apply VADER to body → produce compound, neg, neu, pos.

Assign sentiment_label based on compound thresholds.

Append scores to sentiment_scores_{YYYY_MM}.csv.

Merge original columns + new sentiment columns → enriched_emails_{YYYY_MM}.csv.

Python libraries
pandas

vaderSentiment or nltk.sentiment.vader

numpy

Input files (under data/DataProcessing/)
processed_emails_{YYYY_MM}.csv

Output files (under data/SentimentalAnalysis/)
sentiment_scores_{YYYY_MM}.csv • CSV headers = message_id,compound,neg,neu,pos,sentiment_label

enriched_emails_{YYYY_MM}.csv • All original fields + appended sentiment columns

Memory strategy
Chunked CSV reads: pd.read_csv(..., chunksize=10000) to limit DataFrame size.

Incremental write: each chunk’s results are written to the output CSV in append mode, then discarded.

3. Network Construction
Purpose: Build node/edge lists by streaming enriched data and partitioning edge files for scale.

Functions
Stream each enriched_emails_{YYYY_MM}.csv in chunks.

Extract unique participants (sender + recipients + cc + bcc); emit each as one NDJSON line.

For each email chunk, generate edges: one row per (sender → each recipient/cc/bcc).

Partition edges by first character of source (A–Z, 0–9) into separate CSVs.

Maintain in-memory counters per partition; flush periodically.

Write network_meta.json listing partitions and global counts.

Python libraries
pandas

networkx (for validation)

json

Input files (under data/SentimentalAnalysis/)
enriched_emails_{YYYY_MM}.csv

Output files (under data/NetworkConstruction/)
network_nodes.ndjson • Each line: {"node_id":"email@example.com","name":""}

network_edges_{X}.csv (where X ∈ [A–Z0–9]) • CSV headers = source,target,weight

network_meta.json • {"partitions":[...],"total_nodes":N,"total_edges":M}

Memory strategy
Streaming reads: one chunk at a time.

Partitioned in-memory counters: small dict per edge partition, flushed when it reaches a threshold.

NDJSON nodes: dedupe via a small in-memory set (cleared periodically).

4. Network Analysis
Purpose: Assemble the full graph from partitions and compute core SNA metrics.

Functions
Load network_nodes.ndjson (small) to add nodes to a NetworkX graph.

For each network_edges_{X}.csv partition: stream rows, add weighted edges.

Compute metrics:

Degree centrality

Betweenness centrality

Clustering coefficient

PageRank

Global network density

Write node metrics to sna_metrics.csv.

Write density to network_density.json.

Python libraries
networkx

pandas

json

Input files (under data/NetworkConstruction/)
network_nodes.ndjson

network_edges_{X}.csv

Output files (under data/NetworkAnalysis/)
sna_metrics.csv • Headers = node_id,degree,betweenness,clustering_coeff,pagerank

network_density.json • {"density":float}

Memory strategy
Graph in memory: fits if ≤1 million edges.

Partition streaming: load edges one partition at a time.

5. Network Graph Analysis
Purpose: Compute and export layout coordinates plus a ready-to-use Python visual for Power BI.

Functions
Stream network_nodes.ndjson for node IDs.

Optionally stream sna_metrics.csv for styling metadata.

Build adjacency incrementally by streaming each network_edges_{X}.csv.

Compute layout via networkx.spring_layout (or Barnes-Hut optimizations).

If node count ≫100 k, sample a subgraph or apply incremental layout updates.

Emit graph_layout.ndjson: each line = {"node_id":"...","x":X,"y":Y}.

Generate graph_visualization.py with code to read partitions & layout, and render an interactive Plotly graph in Power BI.

Python libraries
networkx

plotly or matplotlib

json

Input files (under data/NetworkConstruction/ & data/NetworkAnalysis/)
network_nodes.ndjson

network_edges_{X}.csv

sna_metrics.csv (optional)

Output files (under data/NetworkGraphAnalysis/)
graph_layout.ndjson

graph_visualization.py

Memory strategy
Streaming adjacency: process each partition sequentially.

Subgraph sampling for very large graphs to bound memory use.

6. Organizational Insights
Purpose: Derive anomalies, detect silos, map influence, and predict burnout via partitioned ML workflows.

Functions
Stream-merge each month’s enriched_emails_{YYYY_MM}.csv with global sna_metrics.csv by node_id.

Feature engineering: volume, centrality, sentiment aggregates.

Anomaly detection:

Train/apply IsolationForest in a rolling-window fashion.

Apply DBSCAN per month to flag communication outliers.

Community detection: run Louvain once on the full graph to assign community_id.

Influence mapping: threshold PageRank for top influencers.

Burnout prediction: train/apply XGBoost + LogisticRegression via partial_fit on streaming feature batches.

Append per-month insights to insights_{YYYY_MM}.csv.

Summarize across all months in insight_summary.json.

Python libraries
pandas

scikit-learn (IsolationForest, DBSCAN, LogisticRegression)

python-louvain

xgboost

networkx

numpy, json

Input files (under data/SentimentalAnalysis/ & data/NetworkAnalysis/)
enriched_emails_{YYYY_MM}.csv

sna_metrics.csv

Output files (under data/OrganizationalInsight/)
insights_{YYYY_MM}.csv • Headers = node_id,anomaly_score,dbscan_label,community_id,influence_flag,burnout_prob,burnout_label

insight_summary.json • e.g. {"total_anomalies":X,"communities":Y,"high_burnout":Z,"top_influencers":[...]}

Memory strategy
Chunked merges: pd.read_csv(..., chunksize=…).

Incremental model training: use partial_fit where available.

Full-graph Louvain: load graph once, then discard.

7. Interactive Visualization
Purpose: Package partitioned data and visuals for Power BI, including network layout, insights, and charts.

Functions
Read monthly sentiment_scores_{YYYY_MM}.csv, global sna_metrics.csv, monthly insights_{YYYY_MM}.csv, graph_layout.ndjson.

Merge per-node metrics into dashboard_data_{YYYY_MM}.csv.

Generate powerbi_dashboard.py with Plotly/Seaborn visuals: time series, bar charts, network snapshots.

Produce visual_config.json for default filters, layout settings, and visual metadata.

Python libraries
pandas

plotly

seaborn

json

Input files
data/SentimentalAnalysis/sentiment_scores_{YYYY_MM}.csv

data/NetworkAnalysis/sna_metrics.csv

data/OrganizationalInsight/insights_{YYYY_MM}.csv

data/NetworkGraphAnalysis/graph_layout.ndjson

Output files (under data/InteractiveVisualization/)
dashboard_data_{YYYY_MM}.csv • Headers = node_id,compound,degree,anomaly_score,burnout_prob,…

powerbi_dashboard.py

visual_config.json

Memory strategy
Partitioned merges: stitch one month at a time with chunksize.

On-demand loading in Power BI: let Power BI page through large CSVs rather than preloading everything.

## Draft 2

i am creating a Social Network analysis for my project. in the project design, I have multiple modules in the workflow to go from a folder with multiple email files to a Power BI presentation. Here are the current modules: 

Data Processing: 
This is the first module in the project. this module takes in a folder of multiple emails and produces one or multiple files for the next modules 
Next module: Sentimental Analysis 

Sentimental Analysis: 
Previous module: Data Processing Sentiment Analysis 
applies VADER to assign tone and sentiment scores, enriching communication records. Next modules: Network Construction, Organizational Insights Network Construction: Previous module: Sentimental Analysis Network construction generates a graph consisting of nodes, edges, and weighted edges. 
Next modules: Network Analysis, Network Graph Analysis 

Network Analysis: 
Previous module: Network Construction 
Network analysis computes key SNA metrics, including degree, betweenness, clustering coefficient, PageRank, and density, aligning with the requirement for computation of core SNA measures. These metrics support an insights layer that enables anomaly detection, collaboration gap identification, influence mapping, and burnout risk monitoring, aligning directly with client-specified outcomes. 
Next modules: Organizational Insights Network Graph Analysis: Previous module: Network Construction use data from Network Construction to generate Python plugins for Power BI Next modules: Interactive Visualization 

Organizational Insights: 
Previous module: Sentimental Analysis, Network Analysis 
Machine learning models and graph algorithms generate higherlevel insights, including anomaly detection (Isolation Forest, DBSCAN), silo identification (Louvain algorithm), influence mapping (PageRank), and burnout prediction (XGBoost, Logistic Regression). 
Next modules: Interactive Visualization 

Interactive Visualization: 
Previous modules: Organizational Insights, Network Graph Analysis 
this is the final output of the project. this should be one or multiple files that can be put into Power BI for better visualization 

Check to see if this design is valid for the project. i don't want to add more function to the project itself, but it can use different Python library. Also, I only want to use Python for this project, including the final output which will be put into Power BI. change, reduce, merge, split, increase the modules if needed. the most important thing to remember is that each module will have input and output files which will be used by other modules.