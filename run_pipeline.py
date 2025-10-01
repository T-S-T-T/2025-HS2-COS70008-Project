import subprocess

# List of modules to run in order
modules = [
    "DataProcessing.process_emails",
    "SentimentalAnalysis.analyze_sentiment",
    "NetworkConstruction.construct_network",
    "NetworkAnalysis.analyze_network",
    "NetworkGraphAnalysis.network_graph_analysis_1",
    "OrganizationalInsight.derive_insights",
    "InteractiveVisualization.interactive_visualization",
]

def run_pipeline():
    for module in modules:
        print(f"\n=== Running {module} ===")
        try:
            # Run the module as a blocking subprocess
            subprocess.run(["python", "-m", module], check=True)
            print(f"--- Finished {module} ---\n")
        except subprocess.CalledProcessError as e:
            print(f"Error while running {module}: {e}")
            break  # Stop the pipeline if one module fails

if __name__ == "__main__":
    run_pipeline()