import typer
import questionary
from typing import Optional, Any, Dict
from typing_extensions import Annotated

import json
from pathlib import Path

# this is only used for base default config, not for user config
from dsp_ai_eval import _base_config_path, get_yaml_config


# user config should be in the Path: ~/.config/dsp_ai_eval/<config_type>.json
# there are 3 types of configs: env vars, secrets, and pipeline configs
# there should only be one env vars, one secrets, and multiple pipeline configs

# secrets: this function will read the Path: ~/.config/dsp_ai_eval/secrets.json

app = typer.Typer()

default_config: Dict[Any, Any] = get_yaml_config(_base_config_path)
config_base_path = Path("~/.config/dsp_ai_eval/").expanduser()

if not config_base_path.exists():
    config_base_path.mkdir(parents=True)


def read_secrets(
    secrets_path: Path = config_base_path / "secrets.json",
) -> Dict[str, Any]:
    # TODO: add pydantic schema validation for secrets
    with open(secrets_path, "r") as f:
        secrets = json.load(f)
        if isinstance(secrets, dict):
            return secrets
        else:
            raise ValueError(
                "Secrets must be parsable as a dictionary, could not read secrets"
            )


# env vars: this function will read the Path: ~/.config/dsp_ai_eval/env.json
def read_env_vars(env_path: Path = config_base_path / "env.json") -> Dict[str, Any]:
    # TODO: add pydantic schema validation for env vars, unsure that output is a dictionary
    with open(env_path, "r") as f:
        env = json.load(f)
        if isinstance(env, dict):
            return env
        else:
            raise ValueError(
                "Env vars must be parsable as a dictionary, could not read env vars"
            )


# secrets: this function will create or overwrite existing secrets at the Path: ~/.config/dsp_ai_eval/secrets.json
@app.command()
def set_secrets(
    openai_key: Annotated[
        str, typer.Option(prompt="Enter your OpenAI API key", hide_input=True)
    ],
    langfuse_host: Annotated[str, typer.Option(prompt="Enter the Langfuse Host")],
    langfuse_public_key: Annotated[
        str, typer.Option(prompt="Enter your Langfuse Public Key", hide_input=True)
    ],
    langfuse_secret_key: Annotated[
        str, typer.Option(prompt="Enter your Langfuse Secret Key", hide_input=True)
    ],
):
    with open(config_base_path / "secrets.json", "w") as f:
        secrets = {
            "OPENAI_API_KEY": openai_key,
            "LANGFUSE_PUBLIC_KEY": langfuse_public_key,
            "LANGFUSE_SECRET_KEY": langfuse_secret_key,
            "LANGFUSE_HOST": langfuse_host,
        }
        json.dump(secrets, f, indent=4)
        del secrets  # remove secrets from memory as an extra security measure


# env vars: this function will create or overwrite existing env vars at the Path: ~/.config/dsp_ai_eval/env.json
@app.command()
def set_env_vars(
    openalex_user: Annotated[
        str, typer.Option(prompt="Enter your OpenAlex User Email")
    ],
    omp_num_threads: Annotated[
        int, typer.Option(prompt="Enter the number of threads for OpenMP")
    ],
    current_abstracts_config: Annotated[Optional[Path], typer.Option()],
    current_reclustering_config: Annotated[Optional[Path], typer.Option()],
):
    if (current_abstracts_config is not None) and (
        current_reclustering_config is not None
    ):
        env_vars = {
            "OPENALEX_USER": openalex_user,
            "OMP_NUM_THREADS": omp_num_threads,
            "CURRENT_ABSTRACTS_CONFIG": current_abstracts_config.as_posix(),
            "CURRENT_RECLUSTERING_CONFIG": current_reclustering_config.as_posix(),
        }

        with open(config_base_path / "env.json", "w") as f:
            json.dump(env_vars, f, indent=4)
    else:
        typer.secho(
            "Please provide a path for both the current abstracts and reclustering configs",
            fg=typer.colors.RED,
        )


# sets the current pipeline config to use
@app.command()
def set_current_abstracts_config():
    config_files = [
        f for f in config_base_path.iterdir() if f.name.startswith("abstracts_")
    ]
    pipeline_choice = questionary.select(
        "Select an Abstracts pipeline config to use",
        choices=[questionary.Choice(title=f.name) for f in config_files],
    ).ask()
    typer.secho(f"Selected {pipeline_choice}", fg=typer.colors.GREEN)

    current_config: Dict[Any, Any] = read_env_vars()
    if "CURRENT_ABSTRACTS_CONFIG" in current_config:
        current_config["CURRENT_ABSTRACTS_CONFIG"] = pipeline_choice

    set_env_vars(
        openalex_user=current_config["OPENALEX_USER"],
        omp_num_threads=current_config["OMP_NUM_THREADS"],
        current_abstracts_config=Path(pipeline_choice),
        current_reclustering_config=Path(current_config["CURRENT_RECLUSTERING_CONFIG"]),
    )


@app.command()
def set_current_reclustering_config():
    config_files = [
        f for f in config_base_path.iterdir() if f.name.startswith("reclustering_")
    ]
    pipeline_choice = questionary.select(
        "Select a Reclustering pipeline config to use",
        choices=[questionary.Choice(title=f.name) for f in config_files],
    ).ask()
    typer.secho(f"Selected {pipeline_choice}", fg=typer.colors.GREEN)

    current_config: Dict[Any, Any] = read_env_vars()
    if "CURRENT_RECLUSTERING_CONFIG" in current_config:
        current_config["CURRENT_RECLUSTERING_CONFIG"] = pipeline_choice

    set_env_vars(
        openalex_user=current_config["OPENALEX_USER"],
        omp_num_threads=current_config["OMP_NUM_THREADS"],
        current_abstracts_config=Path(current_config["CURRENT_ABSTRACTS_CONFIG"]),
        current_reclustering_config=Path(pipeline_choice),
    )


@app.command()
def create_abstracts_config(
    path_raw_data: Annotated[
        str, typer.Option(prompt="Enter the path to the raw data")
    ] = default_config["oa_abstracts_pipeline"]["path_raw_data"],
    path_filtered_data: Annotated[
        str, typer.Option(prompt="Enter the path to the filtered data")
    ] = default_config["oa_abstracts_pipeline"]["path_filtered_data"],
    path_bm25_filtered_data: Annotated[
        str, typer.Option(prompt="Enter the path to the BM25 filtered data")
    ] = default_config["oa_abstracts_pipeline"]["path_bm25_filtered_data"],
    path_cleaned_data_w_embeddings: Annotated[
        str, typer.Option(prompt="Enter the path to the cleaned data with embeddings")
    ] = default_config["oa_abstracts_pipeline"]["path_cleaned_data_w_embeddings"],
    dir_topic_model: Annotated[
        str, typer.Option(prompt="Enter the directory to the topic model")
    ] = default_config["oa_abstracts_pipeline"]["dir_topic_model"],
    path_topics: Annotated[
        str, typer.Option(prompt="Enter the path to the topics")
    ] = default_config["oa_abstracts_pipeline"]["path_topics"],
    path_probs: Annotated[
        str, typer.Option(prompt="Enter the path to the probabilities")
    ] = default_config["oa_abstracts_pipeline"]["path_probs"],
    path_repr_docs: Annotated[
        str, typer.Option(prompt="Enter the path to the representative documents")
    ] = default_config["oa_abstracts_pipeline"]["path_repr_docs"],
    path_summaries: Annotated[
        str, typer.Option(prompt="Enter the path to the summaries")
    ] = default_config["oa_abstracts_pipeline"]["path_summaries"],
    path_summaries_cleaned: Annotated[
        str, typer.Option(prompt="Enter the path to the cleaned summaries")
    ] = default_config["oa_abstracts_pipeline"]["path_summaries_cleaned"],
    path_vis_data: Annotated[
        str, typer.Option(prompt="Enter the path to the visualization data")
    ] = default_config["oa_abstracts_pipeline"]["path_vis_data"],
    embedding_model: Annotated[
        str, typer.Option(prompt="Enter the embedding model to use")
    ] = default_config["embedding_model"],
    seed: Annotated[int, typer.Option(prompt="Enter the seed to use")] = default_config[
        "seed"
    ],
    cluster_colours: Annotated[
        str, typer.Option(prompt="Enter the cluster colour scheme")
    ] = default_config["cluster_colours"],
    rq_prefix: Annotated[
        str, typer.Option(prompt="Enter a prefix for your research question")
    ] = default_config["rq_prefix"],
    RQ: Annotated[
        str, typer.Option(prompt="Enter your research question")
    ] = default_config["RQ"],
):
    abstracts_config = {
        "path_raw_data": path_raw_data,
        "path_filtered_data": path_filtered_data,
        "path_bm25_filtered_data": path_bm25_filtered_data,
        "path_cleaned_data_w_embeddings": path_cleaned_data_w_embeddings,
        "dir_topic_model": dir_topic_model,
        "path_topics": path_topics,
        "path_probs": path_probs,
        "path_repr_docs": path_repr_docs,
        "path_summaries": path_summaries,
        "path_summaries_cleaned": path_summaries_cleaned,
        "path_vis_data": path_vis_data,
        "embedding_model": embedding_model,
        "seed": seed,
        "rq_prefix": rq_prefix,
        "cluster_colours": cluster_colours,
        "research_question": RQ,
    }

    with open(config_base_path / f"abstracts_{RQ}.json", "w") as f:
        json.dump(abstracts_config, f, indent=4)


@app.command()
def create_reclustering_config(
    zeroshot_min_similarity: Annotated[
        float,
        typer.Option(prompt="Enter the minimum similarity for zero-shot clustering"),
    ] = default_config["oa_reclustering_pipeline"]["zeroshot_min_similarity"],
    min_topic_size: Annotated[
        int, typer.Option(prompt="Enter the minimum topic size")
    ] = default_config["oa_reclustering_pipeline"]["min_topic_size"],
    dir_topic_model: Annotated[
        str, typer.Option(prompt="Enter the directory to the reclustering topic model")
    ] = default_config["oa_reclustering_pipeline"]["dir_topic_model"],
    path_topics: Annotated[
        str, typer.Option(prompt="Enter the path to the reclustered topics")
    ] = default_config["oa_reclustering_pipeline"]["path_topics"],
    path_probs: Annotated[
        str, typer.Option(prompt="Enter the path to the reclustered probabilities")
    ] = default_config["oa_reclustering_pipeline"]["path_probs"],
    path_repr_docs: Annotated[
        str,
        typer.Option(
            prompt="Enter the path to the reclustered representative documents"
        ),
    ] = default_config["oa_reclustering_pipeline"]["path_repr_docs"],
    path_summaries: Annotated[
        str, typer.Option(prompt="Enter the path to the reclustered summaries")
    ] = default_config["oa_reclustering_pipeline"]["path_summaries"],
    path_summaries_cleaned: Annotated[
        str, typer.Option(prompt="Enter the path to the cleaned reclustered summaries")
    ] = default_config["oa_reclustering_pipeline"]["path_summaries_cleaned"],
    path_vis_data: Annotated[
        str, typer.Option(prompt="Enter the path to the reclustered visualization data")
    ] = default_config["oa_reclustering_pipeline"]["path_vis_data"],
):
    reclustering_config = {
        "zeroshot_min_similarity": zeroshot_min_similarity,
        "min_topic_size": min_topic_size,
        "dir_topic_model": dir_topic_model,
        "path_topics": path_topics,
        "path_probs": path_probs,
        "path_repr_docs": path_repr_docs,
        "path_summaries": path_summaries,
        "path_summaries_cleaned": path_summaries_cleaned,
        "path_vis_data": path_vis_data,
    }

    with open(config_base_path / "reclustering_base.json", "w") as f:
        json.dump(reclustering_config, f, indent=4)


@app.command()
def setup_config():
    # creates blank config files if they don't exist, TODO change to create default config from base.yaml

    # create the env vars file if it doesn't exist
    set_env_vars(
        openalex_user="",
        omp_num_threads=1,
        current_abstracts_config=config_base_path / "abstracts_base.json",
        current_reclustering_config=config_base_path / "reclustering_base.json",
    )

    # create the secrets file if it doesn't exist
    set_secrets(
        openai_key="",
        langfuse_host="",
        langfuse_public_key="",
        langfuse_secret_key="",
    )

    # create the abstracts_base file if it doesn't exist
    create_abstracts_config()

    # create the reclustering_base file if it doesn't exist
    create_reclustering_config()


def current_abstracts_config_path() -> Path:
    try:
        current_abstracts_name: str = read_env_vars()["CURRENT_ABSTRACTS_CONFIG"]
    except Exception as e:
        setup_config()
        current_abstracts_name: str = read_env_vars()["CURRENT_ABSTRACTS_CONFIG"]

    return Path(current_abstracts_name)


def current_reclustering_config_path() -> Path:
    try:
        current_reclustering_name: str = read_env_vars()["CURRENT_RECLUSTERING_CONFIG"]
    except Exception as e:
        setup_config()
        current_reclustering_name: str = read_env_vars()["CURRENT_RECLUSTERING_CONFIG"]

    return Path(current_reclustering_name)


def read_abstracts_config(
    abstracts_config_path: Path = current_abstracts_config_path(),
    current: bool = True,
) -> Dict[str, Any]:
    # TODO: add pydantic schema validation for abstracts config
    current_abstracts_config = (
        config_base_path / abstracts_config_path
        if current
        else config_base_path / "abstracts_base.json"
    )

    with open(current_abstracts_config, "r") as f:
        config = json.load(f)
        if isinstance(config, dict):
            return config
        else:
            raise ValueError("Error reading abstracts config")


def read_reclustering_config(
    reclustering_config_path: Path = current_reclustering_config_path(),
    current: bool = True,
):
    # TODO: add pydantic schema validation for reclustering config
    current_reclustering_config = (
        reclustering_config_path
        if current
        else config_base_path / "reclustering_base.json"
    )

    with open(current_reclustering_config, "r") as f:
        return json.load(f)


def get_all_configs() -> Dict[str, Any]:
    return {
        "secrets": {**read_secrets()},
        "env_vars": {**read_env_vars()},
        "oa_abstracts_pipeline": {**read_abstracts_config()},
        "reclustering_pipeline": {**read_reclustering_config()},
    }


def update_secrets():
    secrets = read_secrets()

    update_choice = questionary.select(
        message="Choose the secret you want to update",
        choices=[questionary.Choice(k) for k in secrets.keys()],
    ).ask()

    new_value = questionary.password(f"Enter a new value for {update_choice}").ask()

    secrets[update_choice] = new_value

    set_secrets(
        openai_key=secrets["OPENAI_KEY"],
        langfuse_host=secrets["LANGFUSE_HOST"],
        langfuse_public_key=secrets["LANGFUSE_PUBLIC_KEY"],
        langfuse_secret_key=secrets["LANGFUSE_SECRET_KEY"],
    )


def update_env_vars():
    env_vars = read_env_vars()

    update_choice = questionary.select(
        message="Choose the env var you want to update",
        choices=[questionary.Choice(k) for k in env_vars.keys()],
    ).ask()

    new_value: str = typer.prompt(f"Enter a new value for {update_choice}", type=str)

    env_vars[update_choice] = new_value

    set_env_vars(
        openalex_user=env_vars["OPENALEX_USER"],
        omp_num_threads=env_vars["OMP_NUM_THREADS"],
        current_abstracts_config=Path(env_vars["CURRENT_ABSTRACTS_CONFIG"]),
        current_reclustering_config=Path(env_vars["CURRENT_RECLUSTERING_CONFIG"]),
    )


def update_abstracts_pipeline():
    abstracts_config = read_abstracts_config()

    update_choice = questionary.select(
        message="Which abstracts pipeline config value do you want to update",
        choices=[questionary.Choice(k) for k in abstracts_config.keys()],
    ).ask()

    new_value: str = typer.prompt(f"Enter a new value for {update_choice}", type=str)

    abstracts_config[update_choice] = new_value

    create_abstracts_config(
        path_raw_data=abstracts_config["path_raw_data"],
        path_filtered_data=abstracts_config["path_filtered_data"],
        path_bm25_filtered_data=abstracts_config["path_bm25_filtered_data"],
        path_cleaned_data_w_embeddings=abstracts_config[
            "path_cleaned_data_w_embeddings"
        ],
        dir_topic_model=abstracts_config["dir_topic_model"],
        path_topics=abstracts_config["path_topics"],
        path_probs=abstracts_config["path_probs"],
        path_repr_docs=abstracts_config["path_repr_docs"],
        path_summaries=abstracts_config["path_summaries"],
        path_summaries_cleaned=abstracts_config["path_summaries_cleaned"],
        path_vis_data=abstracts_config["path_vis_data"],
        embedding_model=abstracts_config["embedding_model"],
        seed=abstracts_config["seed"],
        cluster_colours=abstracts_config["cluster_colours"],
        rq_prefix=abstracts_config["rq_prefix"],
        RQ=abstracts_config["research_question"],
    )


def update_reclustering_pipeline():
    reclustering_config = read_reclustering_config()

    update_choice = questionary.select(
        message="Which abstracts pipeline config value do you want to update",
        choices=[questionary.Choice(k) for k in reclustering_config.keys()],
    ).ask()

    new_value = typer.prompt(f"Enter a new value for {update_choice}", type=str)

    reclustering_config[update_choice] = new_value

    create_reclustering_config(
        zeroshot_min_similarity=reclustering_config["zeroshot_min_similarity"],
        min_topic_size=reclustering_config["min_topic_size"],
        dir_topic_model=reclustering_config["dir_topic_model"],
        path_topics=reclustering_config["path_topics"],
        path_probs=reclustering_config["path_probs"],
        path_repr_docs=reclustering_config["path_repr_docs"],
        path_summaries=reclustering_config["path_summaries"],
        path_summaries_cleaned=reclustering_config["path_summaries_cleaned"],
        path_vis_data=reclustering_config["path_vis_data"],
    )


@app.command()
def update():
    update_choice: str = questionary.select(
        message="Which config you want to update",
        choices=[
            "Secrets",
            "Environment Variables",
            "Abstracts Pipeline Config",
            "Reclustering Pipeline Config",
        ],
    ).ask()

    update_choice = update_choice.lower().replace(" ", "_")

    match update_choice:
        case "secrets":
            update_secrets()
        case "environment_variables":
            update_env_vars()
        case "abstracts_pipeline_config":
            update_abstracts_pipeline()
        case "reclustering_pipeline_config":
            update_reclustering_pipeline()


@app.command()
def show():
    show_choice: str = questionary.select(
        message="Which config you want to view",
        choices=[
            "Secrets",
            "Environment Variables",
            "Abstracts Pipeline Config",
            "Reclustering Pipeline Config",
        ],
    ).ask()

    show_choice = show_choice.lower().replace(" ", "_")

    match show_choice:
        case "secrets":
            secrets = read_secrets()
            # blank out secrets from view
            blanked_secrets = {
                k: "****hidden****" for k in secrets.keys() if k != "LANGFUSE_HOST"
            }
            blanked_secrets["LANGFUSE_HOST"] = secrets["LANGFUSE_HOST"]
            typer.echo(blanked_secrets)
        case "environment_variables":
            typer.echo(read_env_vars())
        case "abstracts_pipeline_config":
            typer.echo(read_abstracts_config())
        case "reclustering_pipeline_config":
            typer.echo(read_reclustering_config())


if __name__ == "__main__":
    app()
