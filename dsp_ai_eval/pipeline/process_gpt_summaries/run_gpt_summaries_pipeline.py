from pathlib import Path
import subprocess
import os

script_parent = Path(__file__).parent
pipeline_env = os.environ.copy()
pipeline_env["OMP_NUM_THREADS"] = "1"

subprocess.run(
    f"python {script_parent / 'clean_themes.py'}", shell=True, env=pipeline_env
)
subprocess.run(
    f"python {script_parent / 'embed_and_cluster_themes.py'}",
    shell=True,
    env=pipeline_env,
)
subprocess.run(
    f"python {script_parent / 'summarize_clusters.py'}", shell=True, env=pipeline_env
)
subprocess.run(
    f"python {script_parent / 'clean_cluster_summaries.py'}",
    shell=True,
    env=pipeline_env,
)
subprocess.run(
    f"python {script_parent / 'plot_clusters.py'}", shell=True, env=pipeline_env
)
