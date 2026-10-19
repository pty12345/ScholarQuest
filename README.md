# ScholarQuest

ScholarQuest is a taxonomy-guided benchmark for agentic academic paper search in open literature environments. It evaluates academic paper search as a multi-turn retrieval task: given a research query and a literature database, a system iteratively retrieves or explores papers, accumulates a candidate pool, and returns a ranked list evaluated against a relevant-paper set.

The benchmark covers four research intents:

- **Method-oriented:** find papers that use, extend, or analyze a specified method.
- **Setting-anchored:** find papers under a specified task, dataset, or experimental setting.
- **Comparison-based:** find papers that compare methods, assumptions, results, or technical claims.
- **Scope-controlled:** find papers within explicit scope boundaries while avoiding loosely relevant results.

This README follows the benchmark construction and evaluation protocol described in the paper.

## Evaluation Sets

ScholarQuest uses two complementary evaluation sets.

| Evaluation set | Queries | Intent distribution | Use in the paper |
| --- | ---: | --- | --- |
| **ScholarQuest** | 1,111 | 302 method-oriented, 322 setting-anchored, 317 scope-controlled, 170 comparison-based | System-level evaluation of complete literature-search systems |
| **ScholarQuest-100** | 100 | 25 queries for each of the four intents | Model-level evaluation under the shared multi-turn harness |

ScholarQuest is the complete output of the query-generation and automated answer-discovery pipeline. ScholarQuest-100 is a high-fidelity subset produced from ScholarQuest through stratified selection and the additional human-in-the-loop curation process described below.

## Construction of ScholarQuest

### 1. Taxonomy-guided query construction

The construction pipeline starts from 1,682 ACM Computing Classification System leaf nodes. Qwen3-Max maps each topic to one or more arXiv subject categories, and only topics assigned to arXiv computer-science categories are retained, leaving 1,631 eligible CS taxonomy nodes.

The pipeline then:

1. draws 20,000 topic-seed instances with replacement using arXiv category-frequency weights;
2. deterministically shuffles the draws and selects 10,000 instances for query generation;
3. successfully processes 9,972 instances;
4. generates four queries per successful instance, one for each research intent, producing 39,888 raw query records;
5. structurally validates the records and deduplicates identical `final_query` strings, leaving 13,097 unique queries.

### 2. Answer discovery

For each query, the pipeline generates 10 rewritten search queries and retrieves the top 10 results for each rewrite from Google Search, arXiv, and Semantic Scholar. The rewrites include method-focused, task-focused, broader-topic, narrower-topic, and terminology-based formulations.

Retrieved arXiv identifiers are normalized to base IDs by removing version suffixes, and records found through different sources or paths are deduplicated. High-confidence candidates are expanded through references and citing papers, with selective second-hop expansion from high-confidence neighbors and pruning of noisy branches.

### 3. Relevance adjudication and release filtering

A recall-oriented prefilter first removes clear mismatches. The remaining candidates are evaluated by multiple LLM-based relevance judges using the query, paper title, abstract, and metadata. The relevance rubric is:

- **0:** mismatch;
- **1:** partial match;
- **2:** strict match.

Papers with aggregated relevance score 2 are retained as strict-match answers. Borderline and high-disagreement cases receive additional adjudication. Queries with 5–200 final strict-match answers are retained for the released benchmark, yielding 1,111 queries.

The final 1,111-query release is associated with 528 sampled seed instances, 373 unique ACM leaf nodes, 355 distinct topic labels, and 36 reported arXiv CS categories.

## Construction of ScholarQuest-100

ScholarQuest-100 starts from the automatically constructed ScholarQuest answer sets. It contains 100 queries selected through stratified selection, with 25 queries for each research intent, and receives additional iterative review by Ph.D. researchers.

The quality-control objective has two complementary parts:

- remove papers that fail the query's method, setting, scope, comparison, or exclusion constraints;
- discover relevant papers missing from the current answer set.

### 1. Query boundaries and evidence standard

For each selected query, reviewers specify the necessary topic, method, setting, comparison direction, and exclusion conditions, as applicable. Ambiguous wording may be refined while preserving the underlying research intent. A substantive change to an inclusion or exclusion condition defines a revised query version and requires reassessment of the affected candidates.

Paper eligibility is assessed within the same frozen literature environment throughout a review cycle. Relevance follows the benchmark's title–abstract–metadata evidence standard: a paper is a strict match only when the available evidence supports every necessary query constraint.

### 2. Discovery and reassessment rounds

Each complete iteration combines:

1. **Relevance review:** check retained answers and borderline candidates against the current query definition, remove unsupported inclusions, and resolve conflicting judgments.
2. **Complementary discovery:** search for omissions using alternative query formulations, terminology variants, and citation/reference expansion from verified relevant papers.
3. **Consistent reassessment:** evaluate newly discovered candidates under the same evidence standard as existing candidates and revisit affected judgments whenever the query constraints change.

A round is complete only after all planned discovery actions and all resulting candidate assessments have been finished.

### 3. Operational convergence

Convergence is assessed separately for every query and answer set. A query reaches operational convergence only after a complete discovery-and-reassessment round satisfies all three conditions:

1. **Stable query interpretation:** the necessary constraints and relevance boundary no longer change, and no unresolved ambiguity could alter inclusion decisions.
2. **Stable relevance judgments:** the retained answer set and previously judged candidates no longer change; inclusion-affecting disagreements are adjudicated, and unresolved cases are not promoted to strict matches.
3. **No newly verified strict matches:** completed supplementary searches and citation checks yield no additional normalized, deduplicated strict matches, and no planned discovery action or newly identified expansion lead remains pending.

Constraint-level reassessment targets false inclusions and therefore supports precision. Complementary discovery targets omissions and therefore supports recall. The paper uses ScholarQuest-100 for model-level comparisons under the shared harness; its frozen queries and graded judgments support fine-grained evaluation of ranking quality and constraint satisfaction.

## Distinction Between the Two Quality-Control Procedures

The paper reports two separate procedures that should not be conflated:

- For the full 1,111-query ScholarQuest benchmark, a 450-pair stratified human audit evaluates the reliability of automatically assigned relevance labels. Three Ph.D. experts rescore 150 pairs from each LLM score group, and majority vote determines the human score.
- For ScholarQuest-100, query-level iterative curation reassesses constraint satisfaction and searches for missing answers until the operational convergence conditions above are met.

The first procedure assesses label reliability in the full benchmark. The second targets false inclusions and omissions in the curated 100-query evaluation set.

## Data Files

### `datasets/ScholarQuest.jsonl`

The 1,111-query benchmark used for system-level evaluation. Each line contains one query and its strict-match answer set.

Main fields:

- `query_id`
- `final_query`
- `answer_arxiv_ids`
- `final_answer_count`
- `source_dir`

### `datasets/ScholarQuest-100.jsonl`

The 100-query, intent-balanced subset used for model-level evaluation. Each line contains the query, intent and constraint metadata, and its answer set.

Main fields:

- `query_id`
- `final_query`
- `category`
- `constraint_kind`
- `constraint_value`
- `topic_seed`
- `domain`
- `answer_arxiv_ids`

### `datasets/query_metadata.jsonl`

Metadata for the 13,097 exact-string-deduplicated query records produced before final answer-set filtering.

## Repository Layout

```text
.
├── datasets/
│   ├── ScholarQuest.jsonl
│   ├── ScholarQuest-100.jsonl
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

The paper refers to the shared literature environment as **ScholarBase**. The retrieval-backend implementation in this repository is provided under `Lewen-API/`.

## Evaluation Harness in the Paper

The standardized multi-turn harness exposes two atomic tools over ScholarBase:

- `Search`: retrieves candidate papers using model-generated queries;
- `Expand`: retrieves the references and citing papers of a specified paper in the current pool.

At each round, the model selects tool calls and arguments from the research query and current paper-pool state. The harness performs relevance scoring, updates the paper pool, produces the final ranking, and logs per-round actions and paper-pool changes for consistent evaluation of retrieval quality and search efficiency.
