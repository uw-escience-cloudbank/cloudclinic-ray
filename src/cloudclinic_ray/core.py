import argparse
import os
from itertools import batched
from time import perf_counter
from typing import Any

import nltk
import ray
import spacy
from nltk.corpus import brown
from ray.util import ActorPool
from rich.console import Console
from rich.table import Column, Table

WORKER_BATCH_SIZE = 8


class CorpusAnalyzer:
    """Analyze corpus documents using a spaCy English model."""

    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_sm", exclude=["ner", "lemmatizer"])

    @staticmethod
    def analyze_document(doc: Any) -> dict:
        """Count the tokens and passive constructions in a parsed document."""
        tokens = [
            token
            for token in doc
            if not (
                token.is_space or token.is_punct or token.pos_ in {"PUNCT", "SPACE"}
            )
        ]
        # ❗Use dependency labels to identify likely passive constructions.
        passives = [
            token
            for token in tokens
            if token.pos_ == "VERB"
            and any(
                child.dep_ in {"nsubjpass", "csubjpass", "auxpass"}
                for child in token.children
            )
        ]

        return {
            "tokens": len(tokens),
            "passives": len(passives),
            "example": passives[0].sent.text if passives else None,
        }

    def analyze(self, documents: tuple) -> list[dict]:
        """Analyze a batch of documents in one process."""
        return [
            {**metadata, **self.analyze_document(doc)}
            for doc, metadata in self.nlp.pipe(
                documents, as_tuples=True, batch_size=WORKER_BATCH_SIZE, n_process=1
            )
        ]


def get_documents() -> list:
    """Load Brown corpus documents with corresponding genre and file ID."""
    try:
        nltk.data.find("corpora/brown")
    except LookupError:
        nltk.download("brown", quiet=True, raise_on_error=True)

    return [
        (
            " ".join(brown.words(file_id)),
            {"genre": brown.categories(file_id)[0], "file_id": file_id},
        )
        for file_id in sorted(brown.fileids())
    ]


def aggregate(rows: list[dict]) -> dict:
    """Combine resulting counts by genre."""
    results: dict = {}
    for row in sorted(rows, key=lambda r: r["file_id"]):
        stats = results.setdefault(
            row["genre"], {"tokens": 0, "passives": 0, "example": None}
        )
        stats["tokens"] += row["tokens"]
        stats["passives"] += row["passives"]

        if stats["example"] is None and row["example"]:
            stats["example"] = (row["file_id"], row["example"])

    return results


def print_results(results: dict, docs: int, elapsed: float) -> None:
    tokens = sum(stats["tokens"] for stats in results.values())
    ranked = sorted(
        results.items(),
        key=lambda item: item[1]["passives"] / max(item[1]["tokens"], 1),
        reverse=True,
    )
    table = Table(
        "Genre",
        Column("Tokens", justify="right"),
        Column("Passives", justify="right"),
        Column("Per 1,000 tokens", justify="right"),
        title="Number of passive constructions",
    )
    for genre, stats in ranked:
        table.add_row(
            genre,
            f"{stats['tokens']:,}",
            f"{stats['passives']:,}",
            f"{1000 * stats['passives'] / max(stats['tokens'], 1):.2f}",
        )

    print()
    console = Console(markup=False, highlight=False)
    console.print(
        table,
        f"{docs:,} documents · {tokens:,} tokens\n"
        f"Elapsed: {elapsed:.2f}s · {tokens / elapsed:,.0f} tokens/s",
    )
    for genre, stats in ranked[:3]:
        if stats["example"] is not None:
            file_id, text = stats["example"]
            console.print(f"\n{genre} [{file_id}]", style="bold")
            console.print(text)


def run(num_workers: int) -> dict:
    """Analyze the Brown corpus in parallel with a pool of Ray actors."""
    t_started = perf_counter()
    documents = get_documents()

    # ❗Connect to the cluster at RAY_ADDRESS or start Ray locally.
    with ray.init(address=os.getenv("RAY_ADDRESS", "local")):
        # Reserve a CPU per actor using SPREAD to schedule actors among available nodes.
        analyzer = ray.remote(CorpusAnalyzer).options(
            num_cpus=1, scheduling_strategy="SPREAD"
        )
        # Start one actor per worker.
        actors = [analyzer.remote() for _ in range(num_workers)]

        rows = []
        pool = ActorPool(actors)

        # ❗map_unordered() yields completed results as they become available.
        # https://docs.ray.io/en/latest/ray-core/patterns/ray-get-submission-order.html
        for batch_results in pool.map_unordered(
            # ❗remote() is asynchronous and returns a future (ObjectRef) to the result.
            lambda actor, batch: actor.analyze.remote(batch),
            list(batched(documents, WORKER_BATCH_SIZE, strict=False)),
        ):
            rows.extend(batch_results)

        results = aggregate(rows)

    t_finished = perf_counter()
    print_results(results, len(documents), t_finished - t_started)

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-workers", type=int, default=1, help="Number of workers")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run(args.n_workers)


if __name__ == "__main__":
    main()
