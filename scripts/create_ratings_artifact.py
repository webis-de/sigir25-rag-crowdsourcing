import pandas as pd
import numpy as np

from src import MACE


def main(input_path: str = '../data/raw/study2_responses.jsonl.gz', output_path: str = "../data/artifacts/ratings.jsonl.gz"):
    cols =  [
        "correctness_topical",
        "coherence_logical",
        "coherence_stylistic",
        "coverage_broad",
        "coverage_deep",
        "consistency_internal",
        "quality_overall"
    ]

    spam_data = []
    label_data = []

    df = (
        pd.read_json(input_path, lines=True)
        .assign(item=lambda df: df["response_a"]+"_"+df["response_b"])
        .rename(columns={"validity": "correctness_topical"})
    )

    for col in cols:
        print(col)
        with pd.option_context("future.no_silent_downcasting", True):
            X = (
                df
                .pivot(index="item", columns="worker", values=col)
                .replace({"A": 0, "B": 1, "N": 2})
                .fillna(-1)
            )

        m = MACE(n_iter=100, n_restarts=20, verbose=True)
        m.fit(X.values.astype(int))
        spam_data.append(
            pd.DataFrame(list(zip(X.columns, m.get_annotator_stats()["spamming_probs"])), columns=["worker", col])
            .explode(col)
            .drop_duplicates("worker", keep="first")
            .sort_values("worker")
            .set_index("worker")
        )
        if col == "quality_overall":
            label_data.append(
                pd.DataFrame(list(zip(X.index, m.predict_proba(X.values.astype(int)))), columns=["id", col])
                .dropna()
                .assign(id=lambda df: df["id"].apply(lambda row: row.split("_")))
                .assign(
                    response_a=lambda df: df["id"].apply(lambda row: row[0]),
                    response_b=lambda df: df["id"].apply(lambda row: row[1]),
                    val_p_a=lambda df: df[col].apply(lambda val: val[0]),
                    val_p_b=lambda df: df[col].apply(lambda val: val[1]),
                    gold=lambda df: df[col].apply(lambda val: ["a", "b"][np.argmax(val)]),
                )
                .drop(columns=["id", col])
                .rename({"val_p_a": col+"_p_a", "val_p_b":  col+"_p_b", "gold": col}, axis=1)
                .sort_values(["response_a", "response_b"])
                .set_index(["response_a", "response_b"])
            )
        else:
            label_data.append(
                pd.DataFrame(list(zip(X.index, m.predict_proba(X.values.astype(int)))), columns=["id", col])
                .dropna()
                .assign(id=lambda df: df["id"].apply(lambda row: row.split("_")))
                .assign(
                    response_a=lambda df: df["id"].apply(lambda row: row[0]),
                    response_b=lambda df: df["id"].apply(lambda row: row[1]),
                    val_p_a=lambda df: df[col].apply(lambda val: val[0]),
                    val_p_b=lambda df: df[col].apply(lambda val: val[1]),
                    val_p_n=lambda df: df[col].apply(lambda val: val[2]),
                    gold=lambda df: df[col].apply(lambda val: ["a", "b", "n"][np.argmax(val)]),
                )
                .drop(columns=["id", col])
                .rename({"val_p_a": col+"_p_a", "val_p_b":  col+"_p_b", "val_p_n":  col+"_p_n", "gold": col}, axis=1)
                .sort_values(["response_a", "response_b"])
                .set_index(["response_a", "response_b"])
            )

    final = (
        df
        .drop(columns="item")
        .merge(
            pd.concat(spam_data, axis=1),
            on=["worker"],
            how="left",
            suffixes=("_vote", "_spam_probability"),
        )
        .groupby(["response_a", "response_b"])
        # A handful of pairs have 1 extra vote due to a questionnaire being included in 2 batches of voting,
        # thus being annotated twice; the first 5 are the intended ones, only keep them.
        # For the others were only 5 votes are present in the first place the indexing has no effect
        .agg({
            'submission_id': lambda x: list(set(x))[0],
            'query_id': lambda x: list(set(x))[0],
            'worker': lambda x: x.tolist()[:5],
            'correctness_topical_vote': lambda x: x.tolist()[:5],
            'coherence_logical_vote': lambda x: x.tolist()[:5],
            'coherence_stylistic_vote': lambda x: x.tolist()[:5],
            'coverage_broad_vote': lambda x: x.tolist()[:5],
            'coverage_deep_vote': lambda x: x.tolist()[:5],
            'consistency_internal_vote': lambda x: x.tolist()[:5],
            'quality_overall_vote': lambda x: x.tolist()[:5],
            'correctness_topical_spam_probability': lambda x: x.tolist()[:5],
            'coherence_logical_spam_probability': lambda x: x.tolist()[:5],
            'coherence_stylistic_spam_probability': lambda x: x.tolist()[:5],
            'coverage_broad_spam_probability': lambda x: x.tolist()[:5],
            'coverage_deep_spam_probability': lambda x: x.tolist()[:5],
            'consistency_internal_spam_probability': lambda x: x.tolist()[:5],
            'quality_overall_spam_probability': lambda x: x.tolist()[:5]
        })
        .reset_index()
        .merge(
            pd.concat(label_data, axis=1).reset_index(),
            on=["response_a", "response_b"]
        )
        .rename(columns={
            "correctness_topical": "correctness_topical_gold",
            "coherence_logical": "coherence_logical_gold",
            "coherence_stylistic": "coherence_stylistic_gold",
            "coverage_broad": "coverage_broad_gold",
            "coverage_deep": "coverage_deep_gold",
            "consistency_internal": "consistency_internal_gold",
            "quality_overall": "quality_overall_gold"
        })
        .loc[
            :,
            [
                'submission_id',
                'query_id',
                'response_a',
                'response_b',
                'worker',
                # Topical correctness voting data
                'correctness_topical_vote',
                'correctness_topical_spam_probability',
                'correctness_topical_p_a',
                'correctness_topical_p_n',
                'correctness_topical_p_b',
                'correctness_topical_gold',
                # Logical coherence voting data
                'coherence_logical_vote',
                'coherence_logical_spam_probability',
                'coherence_logical_p_a',
                'coherence_logical_p_n',
                'coherence_logical_p_b',
                'coherence_logical_gold',
                # Stylistic coherence voting data
                'coherence_stylistic_vote',
                'coherence_stylistic_spam_probability',
                'coherence_stylistic_p_a',
                'coherence_stylistic_p_n',
                'coherence_stylistic_p_b',
                'coherence_stylistic_gold',
                # Broad coverage voting data
                'coverage_broad_vote',
                'coverage_broad_spam_probability',
                'coverage_broad_p_a',
                'coverage_broad_p_n',
                'coverage_broad_p_b',
                'coverage_broad_gold',
                # Deep coverage voting data
                'coverage_deep_vote',
                'coverage_deep_spam_probability',
                'coverage_deep_p_a',
                'coverage_deep_p_n',
                'coverage_deep_p_b',
                'coverage_deep_gold',
                # Internal consistency voting data
                'consistency_internal_vote',
                'consistency_internal_spam_probability',
                'consistency_internal_p_a',
                'consistency_internal_p_n',
                'consistency_internal_p_b',
                'consistency_internal_gold',
                # Overall quality voting data
                'quality_overall_vote',
                'quality_overall_spam_probability',
                'quality_overall_p_a',
                'quality_overall_p_b',
                'quality_overall_gold'
            ]
        ]
    )

    #final.columns = pd.MultiIndex.from_tuples((
    #    ('meta', 'submission_id'),
    #    ('meta', 'query_id'),
    #    ('meta', 'response_a'),
    #    ('meta', 'response_b'),
    #    ('meta', 'workers'),
    #    # Topical correctness voting data
    #    ('correctness_topical', 'votes'),
    #    ('correctness_topical', 'spam_probabilities'),
    #    ('correctness_topical', 'prob_a'),
    #    ('correctness_topical', 'prob_n'),
    #    ('correctness_topical', 'prob_b'),
    #    ('correctness_topical', 'gold'),
    #    # Logical coherence voting data
    #    ('coherence_logical', 'votes'),
    #    ('coherence_logical', 'spam_probabilities'),
    #    ('coherence_logical', 'prob_a'),
    #    ('coherence_logical', 'prob_n'),
    #    ('coherence_logical', 'prob_b'),
    #    ('coherence_logical', 'gold'),
    #    # Stylistic coherence voting data
    #    ('coherence_stylistic', 'votes'),
    #    ('coherence_stylistic', 'spam_probabilities'),
    #    ('coherence_stylistic', 'p_a'),
    #    ('coherence_stylistic', 'p_n'),
    #    ('coherence_stylistic', 'p_b'),
    #    ('coherence_stylistic', 'gold'),
    #    # Broad coverage voting data
    #    ('coverage_broad', 'votes'),
    #    ('coverage_broad', 'spam_probabilities'),
    #    ('coverage_broad', 'prob_a'),
    #    ('coverage_broad', 'prob_n'),
    #    ('coverage_broad', 'prob_b'),
    #    ('coverage_broad', 'gold'),
    #    # Deep coverage voting data
    #    ('coverage_deep', 'votes'),
    #    ('coverage_deep', 'spam_probabilities'),
    #    ('coverage_deep', 'prob_a'),
    #    ('coverage_deep', 'prob_n'),
    #    ('coverage_deep', 'prob_b'),
    #    ('coverage_deep', 'gold'),
    #    # Internal consistency voting data
    #    ('consistency_internal', 'votes'),
    #    ('consistency_internal', 'spam_probabilities'),
    #    ('consistency_internal', 'prob_a'),
    #    ('consistency_internal', 'prob_n'),
    #    ('consistency_internal', 'prob_b'),
    #    ('consistency_internal', 'gold'),
    #    # Overall quality voting data
    #    ('quality_overall', 'votes'),
    #    ('quality_overall', 'spam_probabilities'),
    #    ('quality_overall', 'prob_a'),
    #    ('quality_overall', 'prob_b'),
    #    ('quality_overall', 'gold')
    #))
    final.to_json(output_path, lines=True, orient='records', compression='gzip')

if __name__ == '__main__':
    main()