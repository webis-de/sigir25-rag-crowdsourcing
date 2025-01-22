import argparse

import pandas as pd

def main(
    input_path_responses: str = "../data/artifacts/responses.jsonl.gz",
    input_path_tasks: str = "../data/raw/study1_tasks.jsonl.gz",
    output_path: str = "../data/raw/study2_subsampled_topics.csv"
):
    (
        pd.read_json(input_path_responses, lines=True)
        .merge(
            pd.read_json(input_path_tasks, lines=True)
            .loc[:, ["response", "worker"]]
        )
        .query("kind == 'human' & style == 'essay'")
        .loc[:, ["response", "topic", "worker"]]
        .sample(frac=1, random_state=SEED)
        .groupby("worker")
        .nth([0, 1, 2, 3])  # Take the first four topics for each worker
        .sample(frac=1, random_state=SEED)
        .head(50)  # We have slightly more than 50 from previous step, so randomly choose 50
        ["topic"]
        .to_csv(output_path, index=False, header=False)
    )

if __name__ == '__main__':
    SEED = 2847525549  # Randomly generated 64-bit integer as seed, for reproducibility

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path_responses", required=False, type=str,
                        default="../data/artifacts/responses.jsonl.gz")
    parser.add_argument("--input_path_tasks", required=False, type=str,
                        default="../data/raw/study1_tasks.jsonl.gz")
    parser.add_argument("--output_path", required=False, type=str,
                        default="../data/raw/study2_subsampled_topics.csv")
    args = parser.parse_args()

    main(
        input_path_responses=args.input_path_responses,
        input_path_tasks=args.input_path_tasks,
        output_path=args.output_path
    )