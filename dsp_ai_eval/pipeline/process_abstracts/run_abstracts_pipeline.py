from pathlib import Path
import subprocess
import os

script_parent = Path(__file__).parent
pipeline_env = os.environ.copy()
pipeline_env["OMP_NUM_THREADS"] = "1"

subprocess.run(f"python {script_parent / 'embed_scite_abstracts.py'}", shell=True)
subprocess.run(f"python {script_parent / 'cluster_abstracts.py'}", shell=True)
subprocess.run(
    f"python {script_parent / 'cluster_summarization_pipeline.py'}",
    shell=True,
    env=pipeline_env,
)  # some interdependencies here causing coredump segfault
subprocess.run(
    f"python {script_parent / 'clean_cluster_summaries.py'}",
    shell=True,
    env=pipeline_env,
)  # some interdependencies here causing coredump segfault
subprocess.run(
    f"python {script_parent / 'plot_abstract_clusters.py'}",
    shell=True,
    env=pipeline_env,
)  # some interdependencies here causing coredump segfault
