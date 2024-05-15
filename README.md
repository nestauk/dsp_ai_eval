# dsp_ai_eval

## Setup

- Meet the data science cookiecutter [requirements](http://nestauk.github.io/ds-cookiecutter/quickstart), in brief:
  - Install: `direnv` and `conda`
- Run `make install` to configure the development environment:
  - Setup the conda environment
  - Configure `pre-commit`
- Make sure you have a `.env` file with the following keys:
```
OPENAI_API_KEY = 'YOUR-KEY-HERE'
```

## Repo structure

- Key hyperparameters are stored in `dsp_ai_eval/config/base.yaml`. This file contains: the research question for the project; hyperparameters for the topic modelling and GPT prompting; plus paths to relevant files in the S3 bucket.
- `dsp_ai_eval/getters/` contains functions for obtaining different raw and processed datasets and artefacts. This is a Nesta DS convention.
- There are three pipelines in `dsp_ai_eval/pipeline/`:
  - `generate_themes_with_gpt/`: pipeline for obtaining repeated GPT answers to the research question.
  - `process_abstracts/`: pipeline for performing text clustering on research abstracts
  - `process_gpt_summaries/`: pipeline for performing text clustering on the summaries obtained with the `generate_themes_with_gpt/` pipeline

## Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>
