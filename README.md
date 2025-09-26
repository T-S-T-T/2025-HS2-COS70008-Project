# 2025-HS2-COS70008-Project
SNA project for Unit COS70008 using Python

The project consists of 7 modules

## Data Processing
python -m DataProcessing.process_emails
## Sentimental Analysis
python -m SentimentalAnalysis.analyze_sentiment
## Network Construction
python -m NetworkConstruction.construct_network
## Network Analysis
python -m NetworkAnalysis.analyze_network
## Network Graph Analysis
python -m NetworkGraphAnalysis.analyze_graph
## Organizational Insights
python -m OrganizationalInsight.derive_insights
## Interactive Visualization
python -m InteractiveVisualization.visualize_dashboard
## Power BI notes

## Draft

Data Processing: 
This is the first module in the project. this module takes in a folder of multiple emails and produces one or multiple files for the next modules 
Next module: Sentimental Analysis 

Sentimental Analysis: 
Previous module: Data Processing Sentiment Analysis 
applies VADER to assign tone and sentiment scores, enriching communication records. 
Next modules: Network Construction, Organizational Insights 

Network Construction: 
Previous module: Sentimental Analysis 
Network construction generates a graph consisting of nodes, edges, and weighted edges. 
Next modules: Network Analysis, Network Graph Analysis 

Network Analysis: 
Previous module: Network Construction 
Network analysis computes key SNA metrics, including degree, betweenness, clustering coefficient, PageRank, and density, aligning with the requirement for computation of core SNA measures. These metrics support an insights layer that enables anomaly detection, collaboration gap identification, influence mapping, and burnout risk monitoring, aligning directly with client-specified outcomes. 
Next modules: Organizational Insights 

Network Graph Analysis: 
Previous module: Network Construction 
use data from Network Construction to generate Python plugins for Power BI 
Next modules: Interactive Visualization 

Organizational Insights: 
Previous module: Sentimental Analysis, Network Analysis 
Machine learning models and graph algorithms generate higherlevel insights, including anomaly detection (Isolation Forest, DBSCAN), silo identification (Louvain algorithm), influence mapping (PageRank), and burnout prediction (XGBoost, Logistic Regression). 
Next modules: Interactive Visualization 

Interactive Visualization: 
Previous modules: Organizational Insights, Network Graph Analysis 
this is the final output of the project. this should be one or multiple files that can be put into Power BI for better visualization 

Each module's code must be inside its designated folder. For example, Sentimental Analysis:
The code must be inside the folder "/SentimentalAnalysis". Then input files for the module can be found in "../data/DataProcessing". The output files must be put into "../data/SentimentalAnalysis"


## Draft 2


1. Data Processing
Purpose: Stream and parse raw Maildir emails into monthly CSV partitions and a line-delimited index for downstream modules.

Functions
Recursively traverse ../data/maildir/; treat every file as an email blob.
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
Folder: ../data/maildir/ 
• Structure: arbitrary nesting; each file is one email 
• Memory: streaming via Path.rglob()

Output files (under ../data/DataProcessing/)
processed_emails_{YYYY_MM}.csv 
• Format: CSV; headers = message_id,date,sender,recipients,cc,bcc,subject,body 
• Partition: one file per month (e.g. processed_emails_2000_07.csv) 
• Size: ~10–100 MB each

email_index.ndjson 
• Format: newline-delimited JSON; each line = {"message_id":"<..>","path":"maildir/.../file"}

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

Input files (under ../data/DataProcessing/)
processed_emails_{YYYY_MM}.csv

Output files (under ../data/SentimentalAnalysis/)
sentiment_scores_{YYYY_MM}.csv 
• CSV headers = message_id,compound,neg,neu,pos,sentiment_label

enriched_emails_{YYYY_MM}.csv 
• All original fields + appended sentiment columns

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
Write network_meta.json listing partitions and both edge counts:
- total_edges_unique = number of distinct (source, target) pairs.
- total_edges_weighted = sum of all edge weights (total emails).

Python libraries
pandas
networkx (for validation) (optional)
json

Input files (under ../data/SentimentalAnalysis/)
enriched_emails_{YYYY_MM}.csv

Output files (under ../data/NetworkConstruction/)
network_nodes.ndjson 
• Each line: {"node_id":"email@example.com","name":""}

network_edges_{X}.csv (where X ∈ [A–Z0–9]) 
• CSV headers = source,target,weight

network_meta.json 
• {"partitions":[...],"total_nodes":N,"total_edges":M,"total_edges_weighted": W}

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
• Degree centrality
• Betweenness centrality
• Clustering coefficient
• PageRank
• Global network density
Write node metrics to sna_metrics.csv.
Write density to network_density.json.

Python libraries
networkx
pandas
json

Input files (under ../data/NetworkConstruction/)
network_nodes.ndjson
network_edges_{X}.csv

Output files (under ../data/NetworkAnalysis/)
sna_metrics.csv 
• Headers = node_id,degree,betweenness,clustering_coeff,pagerank

network_density.json 
• {"density":float}

Memory strategy
Graph in memory: fits if ≤1 million edges.
Partition streaming: load edges one partition at a time.


5. Network Graph Analysis
Purpose Compute stable layout coordinates for each node and emit flat data files that Power BI can ingest directly as network‐snapshot visuals.

Functions
Stream node list 
• Read network_nodes.ndjson one line at a time 
• Collect node_id values into a lightweight index (no full DataFrame)

Optionally load styling metadata 
• If sna_metrics.csv is present, read only its centrality columns (degree, pagerank, etc.) into a dict[node_id → metrics]

Build adjacency incrementally 
• For each partition file network_edges_{X}.csv, stream rows and add weighted edges to a NetworkX graph instance 
• Avoid loading all partitions at once—add edges partition by partition

Subgraph sampling for scale 
• If total node count exceeds a configurable threshold (e.g. 50 000), select a representative subset by:
--Top-N by PageRank, or
--Random sample stratified by community, or
--Snowball‐expand a seed set until size OK 
--• Restrict layout computation to that subgraph

Compute layout 
• Use networkx.spring_layout(G, k=…, iterations=…) for ≤50 k nodes 
• For larger graphs, switch to Barnes-Hut force‐directed (via nx.layout.fruchterman_reingold_layout with threshold) 
• Fix random seed for reproducibility

Emit NDJSON per month 
• For each node in the final graph (or sampled subgraph), write one line to graph_layout_{YYYY_MM}.ndjson: {"node_id":"alice@example.com","x":0.123,"y":-0.456}
• Flush writes incrementally—no full JSON in memory

Produce flat snapshot CSV 
• Immediately after layout, join each node’s x,y with any loaded metrics into a small pandas DataFrame 
• Write network_snapshot_{YYYY_MM}.csv under ../data/InteractiveVisualization/ with headers: node_id,x,y,degree,pagerank,betweenness,clustering_coeff
• This CSV is tailored for Power BI’s Scatter chart visual

Python libraries
networkx
numpy
pandas (for metrics‐merge & final CSV)
json

Input Files
../data/NetworkConstruction/network_nodes.ndjson
../data/NetworkConstruction/network_edges_{X}.csv (one file per partition)
../data/NetworkAnalysis/sna_metrics.csv (optional styling metadata)

Output Files (under ../data/NetworkGraphAnalysis/ and InteractiveVisualization)
../data/NetworkGraphAnalysis/graph_layout_{YYYY_MM}.ndjson 
• One NDJSON line per node: {"node_id":…,"x":…,"y":…}

../data/InteractiveVisualization/network_snapshot_{YYYY_MM}.csv 
• Flat CSV for Power BI scatter: node_id,x,y,degree,pagerank,betweenness,clustering_coeff

../data/NetworkGraphAnalysis/layout_meta_{YYYY_MM}.json 
• Metadata about layout run: node count, subgraph flag, algorithm, timestamp

Memory strategy
Streaming adjacency: Process one network_edges_{X}.csv at a time to keep peak memory low.
Dedicated index: Maintain only a small dict or set of node IDs and metrics, not a giant DataFrame.
Incremental writes: Write each NDJSON line and CSV chunk immediately—never accumulate the full layout in RAM.
Subgraph sampling: For very large graphs, reduce node count before layout computation to bound memory and CPU.


6. Organizational Insights
Purpose: Derive anomalies, detect silos, map influence, and predict burnout via partitioned ML workflows.

Functions
Stream-merge each month’s enriched_emails_{YYYY_MM}.csv with global sna_metrics.csv by node_id.
Feature engineering: volume, centrality, sentiment aggregates.
Anomaly detection:
• Train/apply IsolationForest in a rolling-window fashion.
• Apply DBSCAN per month to flag communication outliers.
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

Input files (under ../data/SentimentalAnalysis/ & ../data/NetworkAnalysis/)
enriched_emails_{YYYY_MM}.csv
sna_metrics.csv

Output files (under ../data/OrganizationalInsight/)
insights_{YYYY_MM}.csv 
• Headers = node_id,anomaly_score,dbscan_label,community_id,influence_flag,burnout_prob,burnout_label

insight_summary.json 
• e.g. {"total_anomalies":X,"communities":Y,"high_burnout":Z,"top_influencers":[...]}

Memory strategy
Chunked merges: pd.read_csv(..., chunksize=…).
Incremental model training: use partial_fit where available.
Full-graph Louvain: load graph once, then discard.

7. Interactive Visualization
Purpose: Produce flat, per-month CSV data files and a JSON index that Power BI can load directly into native visuals—no in-report Python scripting required.

Functions
Load source data per month
-Stream sentiment scores: ../data/SentimentalAnalysis/sentiment_scores_{YYYY_MM}.csv
-Load global network metrics: ../data/NetworkAnalysis/sna_metrics.csv (only once)
-Read organizational insights: ../data/OrganizationalInsight/insights_{YYYY_MM}.csv
-Stream layout coordinates: ../data/NetworkGraphAnalysis/graph_layout_{YYYY_MM}.ndjson
Generate time-series CSV (timeseries_data_{YYYY_MM}.csv)
-Read sentiment_scores_{YYYY_MM}.csv in chunks (e.g. 10 000 rows)
-Convert each email’s date to YYYY-MM-DD
-Aggregate by day:
--date
--avg_compound = mean of compound scores
--total_emails = count of messages
-Write one flat CSV under ../data/InteractiveVisualization/
Generate burnout-bar CSV (burnout_bar_data_{YYYY_MM}.csv)
-Read insights_{YYYY_MM}.csv entirely (small per-month file)
-Select per-node columns:
--node_id
--burnout_prob
--burnout_label
-Write flat CSV for Power BI bar chart
Generate network-snapshot CSV (network_snapshot_data_{YYYY_MM}.csv)
-Stream layout NDJSON line by line into a dict of node_id → (x, y)
-Read insights_{YYYY_MM}.csv and/or sna_metrics.csv to fetch:
--anomaly_score
--burnout_prob
-Combine into a pandas DataFrame with columns:
--node_id
--x, y
--anomaly_score
--burnout_prob
-Write flat CSV optimized for Power BI scatter plot
Write master index JSON (visual_config.json)
-List all available months in chronological order
-Specify default chart types and filter settings:
{
  "months": ["2025_01", "2025_02", "..."],
  "defaults": {
    "time_series":      {"chart_type": "line"},
    "burnout_bar":      {"chart_type": "bar"},
    "network_snapshot": {"chart_type": "scatter"}
  }
}

Python libraries
pandas (streaming, aggregation, CSV I/O)
json (write visual_config.json)

Input files
../data/SentimentalAnalysis/sentiment_scores_{YYYY_MM}.csv
../data/NetworkAnalysis/sna_metrics.csv
../data/OrganizationalInsight/insights_{YYYY_MM}.csv
../data/NetworkGraphAnalysis/graph_layout_{YYYY_MM}.ndjson

Output files (under ../data/InteractiveVisualization/)
timeseries_data_{YYYY_MM}.csv
• Headers = date,avg_compound,total_emails

burnout_bar_data_{YYYY_MM}.csv
• Headers = node_id,burnout_prob,burnout_label

network_snapshot_data_{YYYY_MM}.csv
• Headers = node_id,x,y,anomaly_score,burnout_prob

visual_config.json

Memory strategy
Chunked reads: Use pd.read_csv(..., chunksize=10000) for large sentiment files.
Streaming NDJSON: Read graph_layout_{YYYY_MM}.ndjson line by line to avoid loading entire layout.
Lightweight merges: Only join minimal columns for each output, then discard intermediate DataFrames.
Flat outputs: Emit lean CSVs tailored to each Power BI visual—no post-processing needed in Power BI.