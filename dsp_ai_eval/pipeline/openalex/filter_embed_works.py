import subprocess, sys

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "--upgrade", "setuptools", "pip"]
)
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-warn-conflicts",
        "--disable-pip-version-check",
        "-qr",
        "requirements.txt",
    ]
)

from metaflow import FlowSpec, step, batch, Parameter


class FilterEmbedFlow(FlowSpec):

    model_name = Parameter(
        "model_name",
        help="Name of the SentenceTransformer model to use for embedding",
        default="all-miniLM-L6-v2",  # "nomic-ai/nomic-embed-text-v1.5",
        type=str,
    )

    score_threshold = Parameter(
        "score_threshold",
        help="Minimum relevance score for works to be included",
        default=10,
        type=int,
    )

    bm25_threshold = Parameter(
        "bm25_threshold",
        help="Threshold for number of docs to return from bm25 retriever",
        default=1000,
        type=int,
    )

    @step
    def start(self):
        self.next(self.filter_works)

    @step
    def filter_works(self):
        import pandas as pd

        from utils import num_tokens_from_string

        from dsp_ai_eval import config, S3_BUCKET, logger
        from dsp_ai_eval.utils import text_cleaning as tc
        from dsp_ai_eval.getters.utils import save_to_s3
        from dsp_ai_eval.getters.openalex import get_works_raw

        rq_prefix = config["rq_prefix"]

        FILTERED_OUT_PATH = config["oa_abstracts_pipeline"]["path_filtered_data"]
        FILTERED_S3_KEY = f"{rq_prefix}/{FILTERED_OUT_PATH}"

        # printing these would be better since Metaflow overrides prints to stdout
        df = get_works_raw()
        logger.info(f"Total number of works: {len(df)}")

        df = df[df["relevance_score"] > self.score_threshold]
        logger.info(
            f"Number of works remaining after filtering for relevance_score > {self.score_threshold}: {len(df)}"
        )

        works_with_min_cites = len(
            df[df["cited_by_count"] == df["cited_by_count"].min()]
        )
        logger.info(
            f"Number of works with minimum number of citations: {works_with_min_cites}"
        )

        # unnest some columns for desired data
        biblio = pd.json_normalize(df["biblio"])
        pmid = pd.json_normalize(df["ids"])[["pmid"]]
        primary_location = pd.json_normalize(df["primary_location"]).rename(
            columns={
                "source.display_name": "journal",
                "source.host_organization_name": "publisher",
                "source.issn_l": "issn_l",
            }
        )[
            ["journal", "publisher", "issn_l"]
        ]  # issn_l, source.display_name, source.host_organization_name
        primary_topics = pd.json_normalize(df["primary_topic"]).rename(
            columns=lambda x: f"primary_topic.{x}"
        )
        df = pd.concat([df, biblio, pmid, primary_location, primary_topics], axis=1)
        df = df.dropna(
            subset=[
                "doi",
                "volume",
                "issue",
                "first_page",
                "last_page",
                "pmid",
                "journal",
                "publisher",
                "issn_l",
            ]
        )
        logger.info(
            f"Number of works remaining after dropping NA values on a subset: {len(df)}"
        )

        df["abstract_clean"] = df["abstract"].apply(tc.clean_abstract)

        df = tc.clean_title_and_abstract(
            df, "abstract_clean", "title"
        )  # adds title_abstract column

        df = df.drop_duplicates("doi")
        logger.info(f"Number of works remaining after deduplicating on doi: {len(df)}")
        df = df.drop_duplicates("title")
        logger.info(
            f"Number of works remaining after deduplicating on titles: {len(df)}"
        )

        # count tokens in title_abstract
        df["num_tokens"] = df["title_abstract"].apply(num_tokens_from_string)

        save_to_s3(S3_BUCKET, df, FILTERED_S3_KEY)

        print(
            "Number of documents more than 20000 tokens",
            len(df[df["num_tokens"] > 20000]),
        )

        df = df[df["is_retracted"] == False]
        logger.info(
            f"Number of works remaining after filtering out retracted papers: {len(df)}"
        )

        self.df = df

        self.next(self.bm25_filter)

    @step
    def bm25_filter(self):
        from langchain_community.retrievers import BM25Retriever
        from langchain_community.document_loaders import DataFrameLoader

        from dsp_ai_eval import config, S3_BUCKET
        from dsp_ai_eval.getters.utils import save_to_s3, get_bm25_scores
        from utils import lc_docs_to_df

        rq_prefix = config["rq_prefix"]

        BM25_FILTERED_OUT_PATH = config["oa_abstracts_pipeline"][
            "path_bm25_filtered_data"
        ]
        BM25_FILTERED_S3_KEY = f"{rq_prefix}/{BM25_FILTERED_OUT_PATH}"

        self.df["bm25_score"] = get_bm25_scores(
            config["RQ"], self.df["title_abstract"].tolist()
        )

        loader = DataFrameLoader(self.df, page_content_column="title_abstract")
        docs = loader.load()
        retriever = BM25Retriever.from_documents(docs, k=self.bm25_threshold)
        results = retriever.invoke(config["RQ"])
        res_df = lc_docs_to_df(results)
        print(len(res_df))

        save_to_s3(S3_BUCKET, res_df, BM25_FILTERED_S3_KEY)

        self.df = res_df

        self.next(self.embed_works)

    # @batch(cpu=30, memory=100000)
    @step
    def embed_works(self):
        """This can potentially replace the entire filtering step"""
        from more_itertools import chunked
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.model_name)

        batches = list(
            chunked(self.df["title_abstract"].tolist(), n=10)
        )  # one of the batches in the middle has a longer sequences causing high memory allocation requests

        print(f"Number of chunks: {len(batches)}")

        embeddings = []

        for batch in batches:
            vecs = model.encode(
                batch,
                show_progress_bar=True,
            )
            embeddings.extend(vecs)

        self.df["embeddings"] = [vec.tolist() for vec in embeddings]

        self.next(self.end)

    @step
    def end(self):
        from dsp_ai_eval import config, S3_BUCKET, logger
        from dsp_ai_eval.getters.utils import save_to_s3

        rq_prefix = config["rq_prefix"]
        EMBEDDINGS_OUT_PATH = config["oa_abstracts_pipeline"][
            "path_cleaned_data_w_embeddings"
        ]
        EMBEDDINGS_S3_KEY = f"{rq_prefix}/{EMBEDDINGS_OUT_PATH}"

        save_to_s3(S3_BUCKET, self.df, EMBEDDINGS_S3_KEY)
        logger.info("Filter and embed works complete.")

        pass


if __name__ == "__main__":
    FilterEmbedFlow()
