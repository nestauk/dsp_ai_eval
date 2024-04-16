import subprocess

subprocess.run("python clean_themes.py", shell=True)
subprocess.run("python embed_and_cluster_themes.py", shell=True)
subprocess.run("python summarize_clusters.py", shell=True)
subprocess.run("python clean_cluster_summaries.py", shell=True)
subprocess.run("python plot_clusters.py", shell=True)
