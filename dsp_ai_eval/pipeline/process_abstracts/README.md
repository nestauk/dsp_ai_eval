# Themes in the research landscape: pipeline for clustering research abstracts

## Introduction

We will cluster research papers based on their abstracts, and then summarize the most representative papers from each cluster. This will give us an idea of the main themes in the research landscape, and the most important papers in each theme.

## Usage

1. Navigate to this subdirectory.
2. Run `python run_abstracts_pipeline.py` to run all the steps in sequence.

Note that the final step, `plot_abstract_clusters.py`, saves plots **locally** rather than to s3, so check `outputs/figures/` for the outputs from this script.

## Pipeline steps

1. `embed_scite_abstracts.py`: create vector representations of title + abstract for each of the research papers, using a sentence transformer model defined as `embedding_model` in the config. Clean the data by deduplicating and removing papers with few citations.

2. `cluster_abstracts.py`: cluster the research papers based on their vector representations, using [BERTopic](https://maartengr.github.io/BERTopic/getting_started/quickstart/quickstart.html).

3. `cluster_summarization_pipeline.py`: extract the most representative paper from each cluster and pass these through the OpenAI API to get a summary for each.

4. `clean_cluster_summaries.py`: do some minor cleaning on the GPT-generated cluster summaries. (This is in a separate script from `cluster_summarization_pipeline.py` just so that if we want to modify the cleaning steps, we don't have to regenerate the summaries, as doing so comes with a cost.)

5. `plot_abstract_clusters.py`: visualize the clusters that we have created!

CHECKLIST

- Update file paths in base.yaml
- Update output path for figs in plot_abstract_clusters.py and add corresponding output file in finder
- Update number of args in get_abstracts in scite.py, and in embed_scite_abstracts (?)
- Put data from scite in S3
