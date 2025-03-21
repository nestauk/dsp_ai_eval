# Steps

`python dsp_ai_eval/analysis/racial_bias_org_change/extract_openalex.py` Extract and concat abstracts for RQ1/RQ2

`python dsp_ai_eval/analysis/racial_bias_org_change/filter_rq1.py --package-suffixes=.txt,.py,.yaml --datastore=s3 run` (cleaning and filtering abstracts using regex)

`dsp_ai_eval openalex clustering run-pipeline` (clustering and summarisation)

# RQ1

Hyperparams (these are also recorded in the config):

- openalex_rmin: 0

- minimum citations: ">0"

```
embedding_model: all-miniLM-L6-v2
seed: 42
rq_prefix: racial_bias_org_change/rq1
RQ: <your research question>
openalex_user: <your email>
OMP_NUM_THREADS: 1
cluster_colours: tableau20

oa_abstracts_pipeline:
  path_raw_data: inputs/openalex/data/works_raw.parquet
  path_filtered_data: inputs/openalex/data/works_filtered.parquet
  path_bm25_filtered_data: inputs/openalex/data/works_bm25_filtered.parquet
  path_cleaned_data_w_embeddings: inputs/openalex/data/works_cleaned_w_embeddings.parquet
  openalex_rmin: 10
  min_cites: ">0"
  hdbscan_min_cluster_size: 10
  tfidf_ngram_min: 1
  tfidf_ngram_max: 3
  umap_n_neighbors: 15
  umap_n_components: 50
  reduce_noise: True
  dir_topic_model: outputs/models/openalex/bertopic_abstracts_model
  path_probs: outputs/openalex/data/bertopic_abstracts_model_probs.npy
  path_topics: outputs/openalex/data/bertopic_abstracts_model_topics.pkl
  path_repr_docs: outputs/openalex/data/bertopic_abstracts_representative_docs.pkl
  path_summaries: outputs/openalex/data/abstracts_cluster_summaries.json
  path_summaries_cleaned: outputs/openalex/data/abstracts_cluster_summaries_cleaned.parquet
  path_vis_data: outputs/openalex/data/visualization_data.parquet
  cluster_colours: tableau20
```
