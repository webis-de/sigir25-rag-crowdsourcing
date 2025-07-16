# The Viability of Crowdsourcing for RAG Evaluation

Repository containing code and data for the paper "The Viability of Crowdsourcing for RAG Evaluation".

Just interested in the data? Check out the [data/artifacts](data/artifacts) directory, or the [Zenodo mirror](TODO).

## 💡Summary

How good are humans at writing and rating responses in retrieval-augmented generation scenarios (RAG)? 
To answer this question, we investigate the efficacy of crowdsourcing for RAG through two complementary studies: response writing and judgment of response utility. 
We present the Crowd RAG Corpus 2025 (CRAGC-25). It consists of:
- RAG Responses:
  - across all 301 topics of the TREC~2024 RAG track
  - across three different response styles: 📋Bulleted lists, 📝 Essays, 📰News-style articles
  - total of 903 human-written and 903 LLM-generated responses
- Pairwise response judgments:
  - across 65 topics
  - across 7 quality dimensions (e.g., coverage and coherence)
  - total of 47,320 human judgments and 10,556 LLM judgments
  
Our analyses give insights into human writing behavior for RAG, show the viability of crowdsourced RAG judgment, and reveal that evaluation based on human-written reference responses fails to effectively capture key quality dimensions, while LLM-based judgment fails to reproduce human gold labels.

## 📊 Repository Structure

The repository contains:
```
├── README.md                     -> This README
├── data                          -> All data produced in this study
│   ├── artifacts                 -> Final processed data
│   │   ├── README.md             -> Detailed data documentation
│   │   ├── grades.jsonl.gz       -> Processed pointwise grade data from human preferences
│   │   ├── llm_ratings.jsonl.gz  -> Pairwise preference data from LLMs
│   │   ├── ratings.jsonl.gz      -> Pairwise preference data from humans
│   │   └── responses.jsonl.gz    -> Written response data, includes both human and LLM responses
│   ├── questionnaires            -> HTML Questionnaire templates
│   ├── raw                       -> Raw data as collected in crowdsourcing studies
│   └── studies                   -> Crowdsourcing study configuration files
├── notebooks                     -> Data analysis notebooks to generate tables/plots from the paper
├── scripts                       -> Invokable scripts (e.g., to create questionnaire templates, ...)
└── src                           -> Source code
    ├── aggregation               -> Implementation of Bradley-Terry vote aggregation
    ├── api                       -> Implementation of our crowdsourcing backend
    └── mace                      -> Implementation of the MACE algorithm


```

## 📝 Citation

If you use the data or code in your research, please cite:

```bib
@InProceedings{gienapp:2025a,
  author =                   {Lukas Gienapp and Tim Hagen and Maik Fr{\"o}be and Matthias Hagen and Benno Stein and Martin Potthast and Harrisen Scells},
  booktitle =                {48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2025)},
  doi =                      {10.1145/3726302.3730093},
  isbn =                     {979-8-4007-1592-1/2025/07},
  month =                    jul,
  numpages =                 11,
  pages =                    {159 -- 169},
  publisher =                {ACM},
  site =                     {Padua, Italy},
  title =                    {{The Viability of Crowdsourcing for RAG Evaluation}},
  year =                     2025
}
```


## 📄 License
This repository is licensed under the [MIT License](LICENSE), except the [data](data) directory.
All its contents are licensed under the [CC-BY 4.0 International](data/artifacts/LICENSE) license.

---
Made with 🔬 by the Webis Group
