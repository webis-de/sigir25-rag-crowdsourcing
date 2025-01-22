# Data Documentation

#### [responses.jsonl.gz](responses.jsonl.gz)

| Key                | Description                                                                                  | Source             | Values                    |
|--------------------|:---------------------------------------------------------------------------------------------|--------------------|:--------------------------|
| `response`         | The UUID of the response / task.                                                             | Task Specification | UUID                      |
| `topic`            | The topic ID of this task.                                                                   | Task Specification | String ID                 |
| `style`            | The text style of the response written for this task.                                        | Task Specification | `essay`, `news`, `bullet` |
| `kind`             | Whether this text was written by an LLM or a human                                           | Writing Study      | `llm`, `human`            |
| `query`            | The query text of this topic.                                                                | TREC RAG           | String value              |
| `references_ids`   | The IDs of the 20 sources retrieved for this topics' query. Aligned with `references_texts`. | TREC RAG           | List of string IDs        |
| `references_texts` | The texts of the 20 sources retrieved for this topics' query. Aligned with `references_ids`. | TREC-RAG           | List of string values     |
| `text`             | The text as written by the human author or LLM.                                              | Writing Survey     | String                    |
| `cleaned_text`     | Text as cleaned by our preprocessing pipeline, without reference markers.                    | Writing Survey     | String                    |
| `statements`       | Text parsed into individual statements, each with the corresponding `reference_ids` cited.   | Writing Survey     | List of Dictionaries      |

#### [ratings.jsonl.gz](ratings.jsonl.gz)

Ratings on pairwise response utility as given by crowd workers. The columns prefixed `{dimension}` below are included once for each possible dimension (`correctness_topical`, `coherence_logical`, `coherence_stylistic`, `coverage_broad`, `coverage_deep`, `consistency_internal`, `quality_overall`).

| Key                            | Description                                                                                                  | Source                 | Values                                      |
|:-------------------------------|:-------------------------------------------------------------------------------------------------------------|:-----------------------|---------------------------------------------|
| `submission_id`                | The UUID of the questionnaire this response pair was rated by.                                               | Task Specification     | UUID                                        |
| `query_id`                     | The topic id this response pair belongs to.                                                                  | TREC RAG               | String id                                   |
| `response_a`                   | The UUID of the first response in this pair (displayed lefthand side).                                       | Task Specification     | UUID                                        |
| `response_b`                   | The UUID of the second response in this pair (displayed righthand side).                                     | Task Specification     | UUID                                        |
| `worker`                       | The UUIDs of the 5 workers completing this questionnaire.                                                    | Task Specification     | List of UUID                                |
| `{dimension}_vote`             | The individual votes for the specified dimension by the 5 workers.                                           | Prolific Crowd Workers | List of string, each entry `a`, `n`, or `b` |
| `{dimension}_spam_probability` | The individual spam probabilities associated with each vote for the specified dimension.                     | Prolific Crowd Workers | List of float, each entry between 0 and 1   |
| `{dimension}_p_a`              | The probability of the gold label being `a` for the specified dimension (first response better than second). | Prolific Crowd Workers | float                                       |
| `{dimension}_p_n`              | The probability of the gold label being `n` for the specified dimension (both responses equal).              | Prolific Crowd Workers | float                                       |
| `{dimension}_p_b`              | The probability of the gold label being `b` for the specified dimension (second response better than first). | Prolific Crowd Workers | float                                       |
| `{dimension}_gold`             | The gold label with highest probability for the specified dimension.                                         | Prolific Crowd Workers | `a`, `n`, or `b`                            |


#### [llm_ratings.jsonl.gz](llm_ratings.jsonl.gz)

Ratings on pairwise response utility as given by an LLM. The column named `{dimension}` below is included once for each possible dimension (`correctness_topical`, `coherence_logical`, `coherence_stylistic`, `coverage_broad`, `coverage_deep`, `consistency_internal`, `quality_overall`).

| Key           | Description                                                                   | Source             | Values                     |
|:--------------|:------------------------------------------------------------------------------|:-------------------|:---------------------------|
| `query_id`    | The topic is this response pair belongs to.                                   | TREC RAG           | String ID                  |
| `response_a`  | The UUID of the first response in this pair (included first in the prompt).   | Task Specification | UUID                       |
| `response_b`  | The UUID of the second response in this pair (included second in the prompt). | Task Specification | UUID                       |
| `inference`   | The inference mode the judgments were collected with.                         | Task Specification | `combined` or `individual` |
| `{dimension}` | The rating given by the LLM for this `{dimension}`                            | LLM Inference      | `a`, `n`, or `b`           |

#### [grades.jsonl.gz](grades.jsonl.gz)

Pointwise, per-topic ranked grades as inferred by a Bradley-Terry probabilistic model. Not to be used as absolute values across their topic context!

| Key                    | Description                                      | Source                                    | Values                                                     |
|------------------------|:-------------------------------------------------|-------------------------------------------|:-----------------------------------------------------------|
| `response`             | The UUID of the response.                        | Task Specification                        | UUID                                                       |
| `correctness_topical`  | The topical correctness grade of this response.  | Pairwise Inference w. Bradley-Terry Model | Integer, 1-6, per topic relative ranks, higher is better.  |
| `coherence_logical`    | The logical coherence grade of this response.    | Pairwise Inference w. Bradley-Terry Model | Integer, 1-6, per topic relative ranks, higher is better.  |
| `coherence_stylistic`  | The stylistics coherence grade of this response. | Pairwise Inference w. Bradley-Terry Model | Integer, 1-6, per topic relative ranks, higher is better.  |
| `coverage_broad`       | The broad coverage grade of this response.       | Pairwise Inference w. Bradley-Terry Model | Integer, 1-6, per topic relative ranks, higher is better.  |
| `coverage_deep`        | The deep coverage grade of this response.        | Pairwise Inference w. Bradley-Terry Model | Integer, 1-6, per topic relative ranks, higher is better.  |
| `consistency_internal` | The internal consistency grade of this response. | Pairwise Inference w. Bradley-Terry Model | Integer, 1-6, per topic relative ranks, higher is better.  |
| `quality_overall`      | The overall quality grade of this response.      | Pairwise Inference w. Bradley-Terry Model | Integer, 1-6, per topic relative ranks, higher is better.  |
