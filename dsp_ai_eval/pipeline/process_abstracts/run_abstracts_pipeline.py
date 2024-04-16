import subprocess

subprocess.run("python embed_scite_abstracts.py", shell=True)
subprocess.run("python cluster_abstracts.py", shell=True)
subprocess.run("python cluster_summarization_pipeline.py", shell=True)
subprocess.run("python clean_cluster_summaries.py", shell=True)
subprocess.run("python plot_abstract_clusters.py", shell=True)
