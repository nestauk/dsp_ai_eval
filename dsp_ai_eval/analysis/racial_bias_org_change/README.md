Hyperparams (these are also recorded in the config):

- openalex_rmin: 10

- minimum citations: ">0"

Steps:

`python dsp_ai_eval/analysis/racial_bias_org_change/extract_openalex.py` Extract and concat abstracts

`dsp_ai_eval openalex works process` (cleaning and filtering abstracts)

`dsp_ai_eval openalex clustering run-pipeline` (clustering and summarisation)
