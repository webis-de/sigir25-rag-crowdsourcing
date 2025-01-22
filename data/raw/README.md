# Data Documentation

This file lists information on the structure and content of raw data files, organized by filename.

#### [study1_tasks.jsonl.gz](study1_tasks.jsonl.gz)

Contains data on the different tasks conducted in study 1, including worker responses on entry and exit surveys.

| Key                   | Description                                                                                                                     | Source             | Values                                       |
|-----------------------|:--------------------------------------------------------------------------------------------------------------------------------|--------------------|:---------------------------------------------|
| `response`            | The UUID of the response / task.                                                                                                | Task Specification | UUID                                         |
| `worker`              | The UUID of the worker assigned to this task.                                                                                   | Task Specification | UUID                                         |
| `topic`               | The topic ID of this task.                                                                                                      | TREC RAG           | String ID                                    |
| `style`               | The text style of the response written for this task.                                                                           | Task Specification | `essay`, `news`, `bullet`                    |
| `query_answerable`    | Whether there is a definitive answer to the query.                                                                              | Entry Survey       | Rating `1` (True), `0` (Maybe), `-1` (False) |
| `query_controversial` | Whether the question posed by the query is controversial.                                                                       | Entry Survey       | Rating `1` (True), `0` (Maybe), `-1` (False) |
| `worker_knowledge`    | Self-assessed knowledge of the worker on the topic.                                                                             | Entry Survey       | Scale `1` (low) to `5` (high)                |
| `worker_expertise`    | Self-assessed experties of the worker on the topic.                                                                             | Entry Survey       | Scale `1` (low) to `5` (high)                |
| `worker_satisfaction` | If the worker was satisfied was with the answer they wrote.                                                                     | Exit Survey        | Rating `1` (True), `0` (Maybe), `-1` (False) |
| `references_quality`  | The overall quality of the references given for this query.                                                                     | Exit Survey        | Scale `1` (low) to `5` (high)                |
| `references_extra`    | How much extra information from worker experience or memory contributed to the response that is not provided in the references. | Exit Survey        | Scale `1` (low) to `5` (high)                |
| `references_omission` | Whether workers omitted information given in the references when answering the query.                                           | Exit Survey        | Rating `1` (True), `0` (Maybe), `-1` (False) |

#### [study1_retrieval.jsonl.gz](study1_retrieval.jsonl.gz)

Contains data on the retrieval data (topics and documents) for study 1.

| Key                | Description                                                                                  | Source   | Values                | 
|:-------------------|:---------------------------------------------------------------------------------------------|:---------|:----------------------|
| `topic`            | The ID of this topic.                                                                        | TREC-RAG | String ID             |
| `query`            | The query text of this topic.                                                                | TREC-RAG | String value          |
| `references_ids`   | The IDs of the 20 sources retrieved for this topics' query. Aligned with `references_texts`. | TREC-RAG | List of string IDs    |
| `references_texts` | The texts of the 20 sources retrieved for this topics' query. Aligned with `references_ids`. | TREC-RAG | List of string values |

#### [study1_responses_human.jsonl.gz](study1_responses_human.jsonl.gz)

Contains the writing log for each response, with timestamped snapshots of the writing process taken roughly every 300ms.

| Key         | Description                                                 | Source             | Values                    | 
|:------------|:------------------------------------------------------------|:-------------------|:--------------------------|
| `response`  | The UUID of this response.                                  | Task Specification | UUID                      |
| `topic`     | The ID of the topic this response was written for.          | TREC-RAG           | String ID                 |
| `style`     | The text style of the response written for this task.       | Task Specification | `essay`, `news`, `bullet` |
| `timestamp` | The timeshot the snapshot of text was taken at.             | Writing Survey     | Timestamp                 |
| `text`      | The text as written by the worker as of the time specified. | Writing Survey     | String                    |

#### [study1_responses_llm.jsonl.gz](study1_responses_human.jsonl.gz)

Contains baseline LLM responses, written by GPT4o with the same instructions as crowd workers.

| Key         | Description                                                 | Source             | Values                    | 
|:------------|:------------------------------------------------------------|:-------------------|:--------------------------|
| `response`  | The UUID of this response.                                  | Task Specification | UUID                      |
| `topic`     | The ID of the topic this response was written for.          | TREC-RAG           | String ID                 |
| `style`     | The text style of the response written for this task.       | Task Specification | `essay`, `news`, `bullet` |
| `text`      | The text as written by the worker as of the time specified. | Writing Survey     | String                    |
