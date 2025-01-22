import argparse
from typing import List
import pandas as pd

from openai import OpenAI
from tqdm import tqdm
import uuid

NAMESPACE_UUID = uuid.UUID("0e8b439f-cc41-4468-81c1-f2cf514851d9")

BASE_PROMPT = """You are an agent for Retrieval-Augmented Generation. 
You are given a search request ('topic') and a list of reference documents, just like a search engine would return them.
Compose a response text that answers the search request, explains the topic, and cites the respective references.
Cite claims you take from the references. If many references make very similar claims, cite them all. Not all references need be cited. 
Claims without reference can be added without citation. Quotations "..." are allowed.
Add the reference ID in brackets: [0]. Add citations at the end of a sentence before the full-stop. Combine multiple citations within the same brackets separated by commas.
Adopt a formal tone. Avoid first or second person ("I", "you").
The response should be short, around 250 words. Do not use markdown, only emit plain text.
"""

PROMPTS = {
    "essay": BASE_PROMPT + "The response should be written in the style of an essay: Start with a clear thesis, provide arguments, finish with a conclusion.",
    "bullet": BASE_PROMPT + "The response should be written in the style of a bullet point list: List all the relevant points to the answer in a single-level bullet point list.",
    "news": BASE_PROMPT + "The response should be written in the style of a news article following the inverted pyramid scheme: Start with the lead, then provide the important details, lastly add background information."
}

QUERY_PROMPT = "Query: {query}\n"
DOC_PROMPT = "[{ref_number}] {ref_text}\n"
REFERENCE_PROMPT = "References:\n"

def get_system_prompt(style: str) -> str:
    return PROMPTS[style] + "\n\n"

def get_user_prompt(query: str, references: List[str]) -> str:
    references = list(map(lambda x: x.replace("\n", " "), references))
    return (
        QUERY_PROMPT.format(query=query) + "\n"
        + REFERENCE_PROMPT
        + "".join([DOC_PROMPT.format(ref_number=i, ref_text=text) for i, text in enumerate(references)])
    )

def get_response(query: str, references: List[str], style="essay", client: OpenAI=None) -> str:
    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": get_system_prompt(style)},
            {"role": "user", "content": get_user_prompt(query, references)}
        ],
        model="gpt-4o"
    )
    return str(completion.choices[0].message.content)


def main(input_path="../data/raw/study1_retrieval.jsonl.gz", output_path="../data/raw/study1_responses_llm.jsonl.gz", api_key=None):
    client = OpenAI(api_key=api_key)
    df = pd.read_json(input_path, lines=True)
    responses = []
    pbar = tqdm(total=len(df) * 3)
    with open("responses.txt", "w") as file:
        for index, row in df.iterrows():
            for style in ["bullet", "essay", "news"]:
                response = get_response(row["query"], row["references_texts"], style, client)
                responses.append((row["topic"], style, response))
                file.write(response.encode("unicode_escape").decode("utf-8") + "\n")
                pbar.update()

    (
        pd.DataFrame(responses, columns=["topic", "style", "text"])
        .assign(
            response=lambda df: df.apply(lambda row: str(uuid.uuid3(NAMESPACE_UUID, row["topic"]+row["style"]+"llm")), axis=1)
        )
        .loc[:, ["response", "topic", "style", "text"]]
        .to_json(
            output_path,
            lines=True,
            orient="records",
            compression="gzip"
        )
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", required=True, type=str)
    parser.add_argument("--input_path", required=False, type=str, default="../data/raw/study1_retrieval.jsonl.gz")
    parser.add_argument("--output_path", required=False, type=str, default="../data/raw/study1_responses_llm.jsonl.gz")
    args = parser.parse_args()

    main(
        input_path=args.input_path,
        output_path=args.output_path,
        api_key=args.api_key
    )
