import argparse

import pandas as pd
from typing import List, Dict
import re
import string


def split_text_by_reference(text: str, references: List[str], type: str):
    # Regular expression to match text followed by one or more references in brackets
    pattern = r'([^\[]+)(\[[^\]]+\](?:\[[^\]]+\])*)'
    # Find all matches in the text
    iterator = re.finditer(pattern, text)
    # Process each match and clean up the references
    result = []
    end_prev = 0
    for match in iterator:
        # Extract the text and references
        start, end = match.span(2)
        text_part = text[end_prev:start]
        end_prev = end
        # Remove brackets and split references into a list of numbers
        ref_numbers = re.findall(r'\d+', match[2])
        ref_numbers = list(map(int, ref_numbers))
        citations = []
        for i in ref_numbers:
            try:
                citations.append(references[i] )
            except IndexError:
                pass
        # Append to result list as a dictionary
        result.append({'text': text_part, 'citations': citations})
    if end_prev < len(text):
        result.append({'text': text[end_prev:], 'citations': []})

    for i in range(len(result)):
        prefix = result[i]['text']
        if i == len(result) - 1:
            suffix = "."
        else:
            suffix = result[i + 1]['text']

        split_chars = list(string.punctuation)
        if type == "bullet":
            split_chars.remove("-")
        if suffix.startswith(tuple(split_chars)):
            prefix = prefix.strip()
            if prefix != "" and prefix[-1] != suffix[0]:
                prefix += suffix[0]
            suffix = suffix[1:].strip()
        if type == "bullet":
            result[i]["text"] = "\n" + prefix.rstrip()
        else:
            result[i]["text"] = prefix.rstrip()

        if i != len(result) - 1:
            result[i + 1]["text"] = suffix.rstrip()

    result = [x for x in result if len(x["text"].strip()) > 0 or len(x["citations"]) > 0]
    return result

def cleanup_text(raw: List[Dict[str, str]]) -> str:
    return (
        " ".join(x["text"] for x in raw)
        .replace("- ", " - ")
        .replace("  ", " ")
        .replace(" \n", "\n")
        .replace("\n ", "\n")
        .strip("\n")
        .strip()
    )

def main(
    input_path_human: str,
    input_path_llm: str,
    input_path_tasks: str,
    input_path_retrieval: str,
    output_path: str
):

    (
        pd.concat([
            (
                pd.read_json(input_path_human, lines=True)
                .assign(kind="human")
                .sort_values("timestamp", ascending=True)
                .groupby(["response"])
                .last()
                .drop(columns=["timestamp"])
                .reset_index()
                .loc[:, ["response", "topic", "style", "kind", "text"]]
            ),
            (
                pd.read_json(input_path_llm, lines=True)
                .assign(kind="llm")
                .reset_index()
                .loc[:, ["response", "topic", "style", "kind", "text"]]
            )
        ])
        .merge(
            pd.read_json(input_path_retrieval, lines=True),
            on=["topic"],
            how="left"
        )
        .dropna()
        .sort_values(by=["response"], ascending=True)
        .assign(statements=lambda df: df.apply(
            lambda row: split_text_by_reference(row["text"], row["references_ids"], row["style"]), axis=1))
        .assign(
            raw_text=lambda df: df["text"],
            cleaned_text=lambda df: df["statements"].apply(cleanup_text)
        )
        .drop(columns=["text"])
        .loc[:, ["response", "topic", "style", "kind", "query", "references_ids", "references_texts", "raw_text", "cleaned_text", "statements"]]
        .sort_values(["topic", "style", "kind"])
        .to_json(output_path, orient="records", lines=True, compression="gzip")
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path_human", required=False, type=str, default="../data/raw/study1_responses_human.jsonl.gz")
    parser.add_argument("--input_path_llm", required=False, type=str, default="../data/raw/study1_responses_llm.jsonl.gz")
    parser.add_argument("--input_path_tasks", required=False, type=str, default="../data/raw/study1_tasks.jsonl.gz")
    parser.add_argument("--input_path_retrieval", required=False, type=str, default="../data/raw/study1_retrieval.jsonl.gz")
    parser.add_argument("--output_path", required=False, type=str, default="../data/artifacts/responses.jsonl.gz")
    args = parser.parse_args()

    main(
        input_path_human=args.input_path_human,
        input_path_llm=args.input_path_llm,
        input_path_tasks=args.input_path_tasks,
        input_path_retrieval=args.input_path_retrieval,
        output_path=args.output_path
    )