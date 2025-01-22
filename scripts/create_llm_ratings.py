import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import uuid

DIMS = ["correctness_topical", "coherence_logical", "coherence_stylistic", "coverage_broad", "coverage_deep", "consistency_internal", "quality_overall"]

NAMESPACE_UUID = uuid.UUID("0e8b439f-cc41-4468-81c1-f2cf514851d9")

BASE_PROMPT_INDIVIDUAL = """You are an evaluation agent for Retrieval-Augmented Generation. 
You are given a search request and two alternative responses, prefixed with 'Response A:' and 'Response B:', respectively.
Your task is to rate the quality of both responses comparatively, rating which one is better. 
For the questions below, indicate your rating either with 'A' (if response A is better), 'B' (if response B is better) or 'N' (neutral, if both are equal).
For each rating, provide a reasoning. Reply only in JSON format.
The JSON should be structured: for each question, include one key `rating` with the rating as value, and one key `justification` with the justification as value. 
Do not include any other formatting or markdown, only reply with the raw JSON string.
"""

QUESTION_PROMPT_INDIVIDUAL = {
    "correctness_topical": "Which response better answers the query? A good response gives clear information that directly addresses the question. A poor response might talk about other things or suggest asking a different question instead.",
    "coherence_logical": "Which response is easier to follow? A good response presents information in a way that makes sense, and has a clear structure. A poor response jumps between different ideas in a confusing way.",
    "coherence_stylistic": "Which response has a more consistent writing style? A good response uses the same type of language throughout - for example, either all formal or all casual. A response text mixes different styles - for example, switching between very simple and very complex language.",
    "coverage_broad": "Which response looks at more aspects of the topic? A good response discusses different parts of the topic and shows how they connect. A poor response only talks about one small part of the topic.",
    "coverage_deep": "Which response explains things more in detail? A good response gives clear examples and explains ideas completely. A poor response only superficially mentions things without explaining them well.",
    "consistency_internal": "Which response is clearer about how different views fit together? A good response clearly shows when it is comparing different views (like 'some people think X, while others believe Y'). A poor response mixes up different views without explaining how they relate to each other.",
    "quality_overall": "Which response is better overall? Think about your answers to all questions above and decide which response is better in general.",
}


BASE_PROMPT_COMBINED = """You are an evaluation agent for Retrieval-Augmented Generation.
You are given a search request and two alternative responses, prefixed with 'Response A:' and 'Response B:', respectively.
Your task is to rate the quality of both responses comparatively, rating which one is better.
For each of the questions below, indicate your rating either with 'A' (if response A is better), 'B' (if response B is better) or 'N' (neutral, if both are equal).
For each rating, provide a reasoning.

Q1: Which response better answers the query? A good response gives clear information that directly addresses the question. A poor response might talk about other things or suggest asking a different question instead.
Q2: Which response is easier to follow? A good response presents information in a way that makes sense, and has a clear structure. A poor response jumps between different ideas in a confusing way.
Q3: Which response has a more consistent writing style? A good response uses the same type of language throughout - for example, either all formal or all casual. A response text mixes different styles - for example, switching between very simple and very complex language.
Q4: Which response looks at more aspects of the topic? A good response discusses different parts of the topic and shows how they connect. A poor response only talks about one small part of the topic.
Q5: Which response explains things more in detail? A good response gives clear examples and explains ideas completely. A poor response only superficially mentions things without explaining them well.
Q6: Which response is clearer about how different views fit together? A good response clearly shows when it is comparing different views (like 'some people think X, while others believe Y'). A poor response mixes up different views without explaining how they relate to each other.
Q7: Which response is better overall? Think about your answers to all questions above and decide which response is better in general.

Reply only in JSON format.
The JSON should be structured: for each question, include one key with the rating as value, and one key with the justification as value, i.e., `Q1_rating` and `Q1_justification` for question `Q1`.
 Do not include any other formatting or markdown, only reply with the raw JSON string.
"""
QUERY_PROMPT = "Query: {query}\n"
DOC_A_PROMPT = "Response A:´ {doc}\n"
DOC_B_PROMPT = "Response B: {doc}\n"


def get_system_prompt_combined() -> str:
    return BASE_PROMPT_COMBINED + "\n\n"

def get_system_prompt_individual(dim: str) -> str:
    return BASE_PROMPT_INDIVIDUAL  + "\n\nQuestion:" + QUESTION_PROMPT_INDIVIDUAL[dim] + "\n\n"

def get_user_prompt(query: str, doc_a: str, doc_b: str) -> str:
    return (
        QUERY_PROMPT.format(query=query) + "\n"
        + DOC_A_PROMPT.format(doc=doc_a) + "\n"
        + DOC_B_PROMPT.format(doc=doc_b) + "\n"
    )

def get_response_combined(query: str, doc_a: str, doc_b: str, client: OpenAI=None) -> str:
    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": get_system_prompt_combined()},
            {"role": "user", "content": get_user_prompt(query, doc_a, doc_b)},
        ],
        model="gpt-4o",
        top_p=0,
        temperature=0,
        response_format={"type": "json_object"}
    )
    return str(completion.choices[0].message.content)

def get_response_individual(query: str, doc_a: str, doc_b: str, dim: str, client: OpenAI=None) -> str:
    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": get_system_prompt_individual(dim)},
            {"role": "user", "content": get_user_prompt(query, doc_a, doc_b)},
        ],
        model="gpt-4o",
        top_p=0,
        temperature=0,
        response_format={"type": "json_object"}
    )
    return str(completion.choices[0].message.content)

def get_ratings(client, query_text, response_a_text, response_b_text, combined=False):
    if combined:
        ratings = get_response_combined(
            query=query_text,
            doc_a=response_a_text,
            doc_b=response_b_text,
            client=client
        )
        ratings = json.loads(ratings)
        q_lookup = {k: v for k,v in zip(DIMS, [f"Q{i+1}" for i in range(len(DIMS))])}
        ratings = {q_lookup[k]: v for k,v in ratings.items()}
    else:
        ratings = {}
        for dim in DIMS:
            dim_rating = get_response_individual(
                query=query_text,
                doc_a=response_a_text,
                doc_b=response_b_text,
                dim=dim,
                client=client
            )
            dim_rating = json.loads(dim_rating)
            ratings[dim+"_rating"] = dim_rating["rating"]
            ratings[dim+"_justification"] = dim_rating["justification"]
    return ratings


def get_ratings_with_backoff(client, query_text, response_a_text, response_b_text, combined=False, max_retries=3):
    """Wrapper for get_ratings with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return get_ratings(client, query_text, response_a_text, response_b_text, combined)
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                raise e

            # Calculate backoff with jitter
            backoff_time = (2 ** attempt) + random.random()
            time.sleep(backoff_time)


def process_data_parallel(input_data, output_path, client, combined=False, max_workers=4):
    """Process data in parallel with backoff handling"""

    def process_row(row):
        try:
            # One direction
            ratings1 = get_ratings_with_backoff(
                client,
                row.query_text,
                row.response_a_text,
                row.response_b_text,
                combined
            )

            # Other direction
            ratings2 = get_ratings_with_backoff(
                client,
                row.query_text,
                row.response_b_text,
                row.response_a_text,
                combined
            )

            return [
                {
                    "query_id": row.query_id,
                    "response_a": row.response_a,
                    "response_b": row.response_b,
                    "ratings": ratings1
                },
                {
                    "query_id": row.query_id,
                    "response_a": row.response_b,
                    "response_b": row.response_a,
                    "ratings": ratings2
                }
            ]

        except Exception as e:
            print(f"Error processing query_id {row.index}: {str(e)}")
            return []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_row, row) for _, row in input_data.iterrows()]

        with open(output_path, "a") as f:
            for future in tqdm(futures, total=len(futures)):
                for result in future.result():
                    f.write(json.dumps(result) + "\n")

def main(
    input_path_ratings="../data/artifacts/ratings.jsonl.gz",
    input_path_responses="../data/artifacts/responses.jsonl.gz",
    output_path="../data/artifacts/llm_ratings_individual.jsonl",
    combined=False,
    api_key=None,
    max_workers=8
):
    client = OpenAI(api_key=api_key)

    input_data = (
        pd.read_json(input_path_ratings, lines=True)
        .loc[:, ["query_id", "response_a", "response_b"]]
        .merge(
            (
                pd.read_json(input_path_responses, lines=True)
                .loc[:, ["response", "raw_text"]]
                .drop_duplicates()
            ),
            left_on="response_a",
            right_on="response",
            how="left",
        )
        .drop(columns="response")
        .rename(columns={"raw_text": "response_a_text"})
        .merge(
            (
                pd.read_json(input_path_responses, lines=True)
                .loc[:, ["response", "raw_text"]]
                .drop_duplicates()
            ),
            left_on="response_b",
            right_on="response",
            how="left",
        )
        .drop(columns="response")
        .rename(columns={"raw_text": "response_b_text"})
        .merge(
            (
                pd.read_json(input_path_responses, lines=True)
                .loc[:, ["topic", "query"]]
                .drop_duplicates()
            ),
            left_on="query_id",
            right_on="topic",
            how="left",
        )
        .drop(columns="topic")
        .rename(columns={"query": "query_text"})
        .assign(
            pair_1=lambda df: df["response_a"] + "_" + df["response_b"],
            pair_2=lambda df: df["response_b"] + "_" + df["response_a"]
        )
        .sample(frac=1)
    )
    input_data = (
        input_data
        .loc[input_data["pair_1"].isin(input_data["pair_2"]), :]
    )

    input_data = (
        input_data
        .merge(
            pd.read_json(output_path, lines=True),
            on=["query_id", "response_a", "response_b"],
            how="left",
            indicator=True
        )
        .query("_merge == 'left_only'")
        .loc[:, ["query_id", "response_a", "response_b", "query_text", "response_a_text", "response_b_text"]]
    )

    process_data_parallel(input_data, output_path, client, combined, max_workers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", required=True, type=str)
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--input_path_ratings", type=str, required=False, default="../data/artifacts/ratings.jsonl.gz"),
    parser.add_argument("--input_path_responses", type=str, required=False, default="../data/artifacts/responses.jsonl.gz"),
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--combined", type=bool, required=False, default=False)
    args = parser.parse_args()

    main(
        input_path_ratings=args.input_path_ratings,
        input_path_responses=args.input_path_responses,
        output_path=args.output_path,
        combined=args.combined,
        api_key=args.api_key,
        max_workers=args.max_workers
    )

