# python -m streamlit run Network_web/network_web.py

import streamlit as st
import pandas as pd
from pyvis.network import Network
from pathlib import Path
import math
import time

PROJECT_ROOT = Path(__file__).parent.resolve()   # folder where this script is located
BASE_ROOT = PROJECT_ROOT.parent   # move one more folder up from PROJECT_ROOT
DATA_ROOT = BASE_ROOT / "data"   # data folder

INPUT_DIR = DATA_ROOT / "combine_input"   # data
OUTPUT_DIR = DATA_ROOT / "network_web" 
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- File paths ---
NODES_FILE = INPUT_DIR / "nodes.csv"
EDGES_FILE = INPUT_DIR / "edges.csv"
OUTPUT_HTML = OUTPUT_DIR / "network_graph.html"

# --- Streamlit page setup ---
st.set_page_config(page_title="Email Network Graph", layout="wide")
st.title("Interactive Email Communication Network")

@st.cache_data
def load_data():
    nodes = pd.read_csv(NODES_FILE)
    edges = pd.read_csv(EDGES_FILE)
    return nodes, edges

# --- Custom spinner HTML ---
spinner_html = """
<div style="display: flex; align-items: center; justify-content: center; height: 150px;">
  <div style="border: 6px solid #f3f3f3; border-top: 6px solid #3498db; border-radius: 50%;
              width: 40px; height: 40px; animation: spin 1s linear infinite;"></div>
  <p style="margin-left: 15px; font-size: 18px; color: #333;">Please wait... Loading network graph</p>
</div>
<style>
@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
</style>
"""

# --- Display spinner WHILE loading ---
with st.container():
    spinner_placeholder = st.empty()
    spinner_placeholder.markdown(spinner_html, unsafe_allow_html=True)

    # load data
    nodes, edges = load_data()

    # Optional: simulate a short pause to show the animation
    time.sleep(1)

    spinner_placeholder.empty()

# --- Sidebar filters ---
st.sidebar.header("Filters")
years = sorted(edges["year"].dropna().unique())
selected_year = st.sidebar.selectbox("Select Year", years)

months = sorted(edges.loc[edges["year"] == selected_year, "month"].dropna().unique())
selected_month = st.sidebar.selectbox("Select Month", months)

# Filter by selected period
edges_filtered = edges[(edges["year"] == selected_year) & (edges["month"] == selected_month)]
nodes_filtered = nodes[(nodes["year"] == selected_year) & (nodes["month"] == selected_month)]

if nodes_filtered.empty or edges_filtered.empty:
    st.warning("No data available for the selected period.")
    st.stop()

# --- PyVis setup ---
net = Network(
    height="850px",
    width="100%",
    bgcolor="#ffffff",
    font_color="#333333",
    directed=False
)

net.force_atlas_2based(
    gravity=-30,
    central_gravity=0.002,
    spring_length=230,
    spring_strength=0.045,
    damping=0.8,
    overlap=0
)

# --- Identify top PageRank node ---
top_pagerank_node = nodes_filtered.loc[nodes_filtered["pagerank"].idxmax(), "node_id"]

# --- Add nodes ---
for _, row in nodes_filtered.iterrows():
    node_id = row["node_id"]
    degree = row.get("degree", 1)
    pagerank = row.get("pagerank", 0)
    betweenness = row.get("betweenness", 0)
    clustering = row.get("clustering_coeff", 0)

    # Node size proportional to degree
    size = 6 + math.log1p(degree) * 8

    # Color by degree
    if degree >= 10:
        color = "rgb(50,142,110)"      # darker green for high-degree nodes
    else:
        color = "rgb(144,198,124)"     # lighter green for low-degree nodes

    # Keep special color for top PageRank node
    if node_id == top_pagerank_node:
        color = "rgb(205,92,0)"        # orange-brick

    tooltip = (
        f"{node_id}\n"
        f"Degree: {degree}\n"
        f"Pagerank: {pagerank:.6f}\n"
        f"Betweenness: {betweenness:.6f}\n"
        f"Clustering: {clustering:.6f}"
    )

    net.add_node(
        node_id,
        title=tooltip,
        size=size,
        color={"border": "rgb(50,142,110)", "background": color},
        borderWidthSelected=6
    )

# --- Add edges ---
for _, row in edges_filtered.iterrows():
    src, tgt = row["source"], row["target"]
    weight = row.get("weight", 1)
    sentiment = row.get("sentiment_avg", 0)

    # Edge thickness proportional to weight
    width = 1 + math.log1p(weight)

    # Edge color based on sentiment
    if sentiment < -0.2:
        color = "rgb(202,120,66)"         # dark-orange (negative)
    elif sentiment > 0.2:
        color = "rgb(50,142,110)"        # dark blue-green (positive)
    else:
        color = "rgb(144,198,124)"      # light blue-green (neutral)

    net.add_edge(
        src,
        tgt,
        value=weight,
        width=width,
        color=color,
        title=f"Weight: {weight}, Sentiment: {sentiment:.2f}"
    )

# --- Render and save to data/network_web ---
net.save_graph(str(OUTPUT_HTML))

with open(OUTPUT_HTML, "r", encoding="utf-8") as f:
    html_content = f.read()

st.components.v1.html(html_content, height=600, scrolling=False)


# --- Legend ---
st.markdown("""
### Understanding the Email Network

| **Visual Element** | **What It Represents** | **Explanation** |
|---------------------|------------------------|--------------------------|
| **Node size** | Level of activity (connections) | Larger circles show people who send or receive more emails, they are more active in communication. |
| **Node color** | Activity intensity | Darker green circles mean higher email activity, while lighter green circles show less active participants. |
| **Top node (orange)** | Most connected person | The orange-brick circle highlights the person who interacts most frequently across the network, a key communicator. |
| **Edge color** | Tone or mood of communication | **Dark orange** = negative tone, **dark green** = positive tone, **light green** = neutral or balanced mood. |
| **Edge thickness** | Number of emails exchanged | Thicker lines indicate more emails sent between two people, showing stronger communication links. |
""")
