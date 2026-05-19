# ScholarQuest / Paper Search Benchmark

This repository contains the code and data used to build **ScholarQuest**, a benchmark for evaluating paper-search agents. The benchmark asks systems to retrieve sets of relevant arXiv papers for realistic scholarly search queries, rather than answer single factoid questions.

The repository is organized around three things:

- the released benchmark dataset,
- the benchmark construction pipeline,
- the Lewen academic search API used to search and inspect papers.

## Repository Layout

```text
PaperBench_Code/
├── datasets/
│   ├── ScholarQuest.jsonl
│   └── query_metadata.jsonl
├── BenchmarkConstruction/
│   ├── SeedGeneration/
│   ├── QueryGeneration/
│   ├── AnswerFinding/
│   ├── AnswerFinding_v2/
│   ├── AnswerFilters/
│   ├── benchmark/
│   ├── configs/
│   ├── scripts/
│   └── src/paperbench/
└── Lewen-API/
```

## Data Files

### `datasets/ScholarQuest.jsonl`

The final benchmark dataset. It contains **1111** benchmark queries. Each row is one evaluation instance.

Main fields:

- `query_id`: stable query identifier, for example `BQ_000001`
- `final_query`: the user-facing paper search query
- `answer_arxiv_ids`: list of relevant arXiv IDs
- `final_answer_count`: number of relevant answers
- `source_dir`: provenance of the final answer set

The released dataset keeps queries with **5 to 200** final answers.

### `datasets/query_metadata.jsonl`

The query metadata pool used before final answer filtering. It contains **13097** generated and deduplicated query records.

Main fields:

- `query_id`
- `final_query`
- `topic_seed`
- `domain`
- `category`
- `constraint_kind`
- `constraint_value`
- `rationale`
- `risk_flags`
- generation metadata such as `llm_model`, token counts, and source batch info

The four query categories are:

- `method_capability`
- `setting_anchor`
- `claim_comparison`
- `scope_control`

Use this file when you need query descriptions, metadata, or category/domain analysis. Use `ScholarQuest.jsonl` for benchmark evaluation.

## Benchmark Construction Pipeline

The construction code lives in `BenchmarkConstruction/`.

The high-level pipeline is:

1. **Seed generation**
   - Code: `BenchmarkConstruction/SeedGeneration/`
   - Builds topic seeds from ACM/arXiv taxonomy resources.

2. **Query generation**
   - Code: `BenchmarkConstruction/QueryGeneration/`
   - Expands topic seeds into ScholarQuest-style paper retrieval queries.
   - Query generation uses four controlled categories and stores metadata for each query.

3. **Answer finding**
   - Code: `BenchmarkConstruction/AnswerFinding/` and `BenchmarkConstruction/AnswerFinding_v2/`
   - Uses paper search tools and the Lewen API to collect candidate relevant papers for each query.

4. **Answer filtering**
   - Code: `BenchmarkConstruction/AnswerFilters/`
   - Applies stricter cross-model filtering to produce cleaner answer sets.

5. **Benchmark selection and analysis**
   - Code: `BenchmarkConstruction/benchmark/`
   - Stores benchmark snapshots and scripts for query-level analysis.

The reusable package-style pipeline code is under:

```text
BenchmarkConstruction/src/paperbench/
```

Useful entry points:

```bash
cd BenchmarkConstruction
python scripts/run_pipeline.py
python QueryGeneration/generate_pasa_queries.py
python QueryGeneration/validate_pasa_queries.py
python AnswerFinding/run_pasa_answer_finding.py
python AnswerFilters/run_strict_cross_filter.py
```

Some scripts require API keys or local service endpoints. See `.env.example` files and the module-level README files for details.

## Lewen API

`Lewen-API/` is the academic paper search backend used by the construction pipeline. It provides:

- semantic paper search,
- title and metadata lookup,
- paper detail retrieval,
- citation and reference queries.

The API covers roughly 3M arXiv papers in the local deployment used for construction.

Example endpoints from `Lewen-API/README.md`:

```bash
curl "http://<host>:4000/paper/search?query=transformer+attention&limit=5"
curl "http://<host>:4000/paper/2309.06180?fields=*"
curl "http://<host>:4000/paper/1706.03762/citations?limit=10"
```

See `Lewen-API/README.md` and `Lewen-API/docs/` for API details.

## Recommended Reading Order

For a quick understanding of the repository:

1. `datasets/ScholarQuest.jsonl`
2. `datasets/query_metadata.jsonl`
3. `BenchmarkConstruction/QueryGeneration/README_pasa_generation.md`
4. `BenchmarkConstruction/AnswerFilters/README.md`
5. `Lewen-API/README.md`

## Notes

- This repository is focused on benchmark construction and released data, not model inference.
- Generated caches, large raw corpora, and intermediate output directories are intentionally omitted or minimized.
- The JSONL files are line-delimited JSON: each non-empty line is one record.
