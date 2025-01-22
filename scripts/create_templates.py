import os
from pathlib import Path
from typing import List, TypedDict, Dict, Tuple, Optional, Literal
import argparse
import itertools
import random
import json
import uuid

import markdown
import pandas as pd
import numpy as np
from jinja2 import Environment, FileSystemLoader


class Question(TypedDict):
    name: str
    text: str
    description: str
    options: Dict[str, str]
    justification: bool


class ItemResponse(TypedDict):
    query_id: str
    query_text: str
    response_id: str
    response_text: str


class ItemStatement(TypedDict):
    query_id: str
    query_text: str
    response_id: str
    statement_id: str
    statement_text: str
    reference_id: str
    reference_text: str


class ItemPairwise(TypedDict):
    query_id: str
    query_text: str
    response_id_a: str
    response_id_b: str
    response_text_a: str
    response_text_b: str


def get_questionnaire_data(path: str) -> Tuple[str, Literal["response", "statement", "pairwise"], str, List[Question]]:
    with open(path) as f:
        questionnaire = json.load(f)
    return (
        Path(path).stem,
        questionnaire["type"],
        questionnaire["instructions"],
        [Question(**q) for q in questionnaire["questions"]]
    )


def stratified_batching(idx: List[int], weights: List[float], batch_size: int):
    # Cut weights into B evenly sized strata
    strata = pd.qcut(weights, q=batch_size, labels=list(range(batch_size)))
    # Assign stratum to each idx
    stratum_with_idx = list(zip(idx, weights, strata))
    # Group idx by stratum
    idx_by_stratum = {k: [(idx, weight) for idx, weight, stratum in stratum_with_idx if stratum == k] for k in range(batch_size)}
    # Whats the smallest group?
    evenly_sized_batches = min([len(b) for b in idx_by_stratum.values()])
    # Distribute until smallest group is empty
    batches = []
    for i in range(evenly_sized_batches):
        b = []
        for j in range(batch_size):
            item = idx_by_stratum[j].pop()
            b.append(item)
        batches.append(b)

    # Join remaining (idx, weight) pairs into one list
    remaining_items = []
    _ = list(map(remaining_items.extend, idx_by_stratum.values()))
    # Assign highest remaining idx by weight to lowest weighted batch, repeat until none left
    remaining_items = sorted(remaining_items, key=lambda x: x[-1], reverse=True)
    for item in remaining_items:
        # Find lowest-weighted batch
        batch_idx = np.argmin([sum(x[1] for x in b) for b in batches])
        # Assign to it
        batches[batch_idx].append(item)
    # Flatten to (idx, batch) tuples
    res = [(index, i) for i, b in enumerate(batches) for (index, _) in b]
    # Sort by index to get original order
    res = sorted(res, key=lambda x: x[0], reverse=False)
    # Only return batch ids
    return [b for (_, b) in res]


def get_template():
    env = Environment(loader=FileSystemLoader("../data/questionnaires"))
    template = env.get_template(f"questionnaire.html")
    return template


def cast_batch(type: Literal["response", "statement", "pairwise"], batch: pd.DataFrame):
    match type:
        case "response":
            return [
                ItemResponse(
                    query_id=row["topic"],
                    query_text=row["query"],
                    response_id=row["response"],
                    response_text=markdown.markdown(row["cleaned_text"])
                ) for _, row in batch.sample(frac=1).iterrows()
            ]
        case "statement":
            return [
                ItemStatement(
                    query_id=row["topic"],
                    query_text=row["query"],
                    response_id=row["response"],
                    statement_id=row["statement_id"],
                    statement_text=markdown.markdown(row["statement_text"]),
                    reference_id=row["reference_id"],
                    reference_text=markdown.markdown(row["reference_text"]) if row["reference_text"] is not None else None
                ) for _, row in batch.sample(frac=1).iterrows()
            ]
        case "pairwise":
            return [
                ItemPairwise(
                    query_id=row["topic"],
                    query_text=row["query"],
                    response_id_a=row["response_id_a"],
                    response_id_b=row["response_id_b"],
                    response_text_a=markdown.markdown(row["response_text_a"]),
                    response_text_b=markdown.markdown(row["response_text_b"]),
                ) for _, row in batch.sample(frac=1).iterrows()
            ]


def match_reference(row) -> Optional[str]:
    try:
        index = row["references_ids"].index(row["reference_id"])
    except ValueError:
        return None
    return row["references_texts"][index]


def load_items(type, input_path_responses, batch_size):
    items = []
    match type:
        case "response":
            df = pd.read_json(input_path_responses, lines=True, orient="records")
            topics = pd.read_csv("../data/base/study2_subsampled_topics.csv", header=None, names=["topic"])[
                "topic"].values.tolist()
            df = df[df["topic"].isin(topics)]
            items = (
                df
                .assign(length=lambda df: df["cleaned_text"].apply(len))
                .assign(batch=lambda df: stratified_batching(df.index.values, df["length"].values, batch_size))
                .drop(columns=["length"])
            )
        case "statement":
            df = pd.read_json(input_path_responses, lines=True, orient="records")
            topics = pd.read_csv("../data/base/study2_subsampled_topics.csv", header=None, names=["topic"])[
                "topic"].values.tolist()
            df = df[df["topic"].isin(topics)]
            items = (
                df
                .explode("statements")
                .assign(
                    statement_text=lambda df: df["statements"].apply(lambda s: s["text"].strip()),
                    reference_id=lambda df: df["statements"].apply(lambda s: s["citations"])
                )
                .explode("reference_id")
                .assign(
                    reference_text=lambda df: df.apply(lambda row: match_reference(row), axis=1)
                )
                .assign(statement_id=lambda df: df.groupby("response").cumcount())
                .loc[:,
                ["response", "topic", "query", "statement_id", "statement_text", "reference_id", "reference_text"]]
                .assign(length=lambda df: df["statement_text"].apply(lambda x: len(x) if x is not None else 0) + df[
                    "reference_text"].apply(lambda x: len(x) if x is not None else 0))
                .assign(batch=lambda df: stratified_batching(df.index.values, df["length"].values, batch_size))
                .drop(columns=["length"])
            )
        case "pairwise":
            items = (
                pd.read_json(input_path_responses, lines=True, orient="records")
                .loc[:, ["response", "topic", "query", "cleaned_text"]]
                .groupby("topic")
                .apply(lambda group: pd.Series({
                    "topic": group["topic"].unique()[0],
                    "query": group["query"].unique()[0],
                    "responses": list(map(
                        lambda x: sorted(x, key=lambda p: random.random()),
                        itertools.combinations(zip(group["response"], group["cleaned_text"]), 2)
                    ))
                }))
                .explode("responses")
                .assign(
                    response_a=lambda df: df["responses"].apply(pd.Series)[0],
                    response_b=lambda df: df["responses"].apply(pd.Series)[1]
                )
                .assign(
                    response_id_a=lambda df: df["response_a"].apply(pd.Series)[0],
                    response_id_b=lambda df: df["response_b"].apply(pd.Series)[0],
                    response_text_a=lambda df: df["response_a"].apply(pd.Series)[1],
                    response_text_b=lambda df: df["response_b"].apply(pd.Series)[1],
                )
                .dropna()
                .drop(columns=["responses", "response_a", "response_b", "topic"])
                .reset_index()
                .assign(length=lambda df: df["response_text_a"].apply(lambda x: len(x) if x is not None else 0) + df[
                    "response_text_b"].apply(lambda x: len(x) if x is not None else 0))
                .assign(batch=lambda df: stratified_batching(df.index.values, df["length"].values, batch_size))
                .drop(columns=["length"])
            )
    return items


def main(
    input_path_study = "study2a",
    input_path_responses="../data/artifacts/responses.jsonl.gz",
    input_path_topics="../data/raw/study2_subsampled_topics.csv",
    output_path="./templates",
    batch_size = 15,
    backend_url="https://collector-backend-api.web.webis.de",
    frontend_url="https://webis-rag-annotation-study.pages.dev"
):
    study, type, instructions, questions = get_questionnaire_data(input_path_study)
    template = get_template()
    items = load_items(type, input_path_responses, batch_size)

    url_path = f"{output_path.rstrip("/")}/{study}/urls.csv"
    os.makedirs(os.path.dirname(url_path), exist_ok=True)
    with open(url_path, "w") as url_file:
        for name, batch in items.groupby("batch"):
            submission_id = str(uuid.uuid4())
            batch = cast_batch(type, batch)
            parsed_template = template.render(
                study=study,
                type=type,
                instructions=markdown.markdown(instructions),
                questions=questions,
                items=batch,
                submission_id=submission_id,
                backend_url=backend_url
            )
            url_file.write(f'{frontend_url}/{study}/{submission_id}.html\n')
            with open(f"{output_path.rstrip("/")}/{study}/{submission_id}.html", "w") as fh:
                fh.write(parsed_template)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path_study", "-s", required=False, type=str, default="")
    parser.add_argument("--input_path_responses", "-i", required=False, type=str, default="../data/artifacts/responses.jsonl.gz")
    parser.add_argument("--input_path_topics", "-t", required=False, type=str, default="../data/raw/study2_subsampled_topics.csv")
    parser.add_argument("--output_path", "-o", required=False, type=str, default="../data/studies")
    parser.add_argument("--batch_size", "-b", required=False, type=int, default=15)
    parser.add_argument("--frontend_url", "-f", required=False, type=str, default="https://webis-rag-annotation-study.pages.dev")
    parser.add_argument("--backend_url", "-u", required=False, type=str, default="https://collector-backend-api.web.webis.de")
    args = parser.parse_args()

    main(
        input_path_study=args.input_path_study,
        input_path_responses=args.input_path_responses,
        input_path_topics=args.input_path_topics,
        output_path=args.output_path,
        batch_size=args.batch_size,
        frontend_url=args.frontend_url,
        backend_url=args.backend_url
    )