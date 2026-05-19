下面是一份可直接落地的 **Markdown 协议文档草案**。你可以把它保存为 `query_pool_pipeline.md`，作为你们 benchmark 构建的固定流程说明。

````markdown
# Paper Search Agent Benchmark: Query Pool Construction Pipeline

## 1. Goal

This document defines the standard pipeline for constructing the **query pool** of the paper search agent benchmark.  
The goal of this pipeline is to ensure that query construction is:

- scalable
- reproducible
- quality-controlled
- distribution-aware
- suitable for benchmark release, annotation, and downstream agent/RL evaluation

Rather than treating queries as a loose collection of search prompts, we treat each query as a **benchmark unit** with explicit metadata, quality control, and lifecycle management.

---

## 2. Design Principles

The query pool construction pipeline follows the principles below:

### 2.1 Realism
Queries should reflect realistic scholarly search needs rather than artificial or overly polished LLM-generated instructions.

### 2.2 Evaluability
Each query should correspond to a reasonably assessable relevant-paper set.  
Queries that are too vague, too subjective, or not paper-search-oriented should be filtered out.

### 2.3 Agentic Value
Since the benchmark targets **paper search agents** rather than static retrievers, the query pool should include queries that require:

- reformulation
- iterative search
- expansion
- filtering
- multi-hop retrieval
- stopping decisions

### 2.4 Diversity
The query pool should cover diverse:

- intent types
- difficulty levels
- domains
- query styles
- temporal sensitivities

### 2.5 Versionability
All queries should be traceable across versions, including their source, normalization history, filtering decisions, and metadata updates.

---

## 3. Overview of the Pipeline

The query pool construction pipeline consists of the following stages:

1. **Stage 0: Query Specification**
2. **Stage 1: Raw Query Collection**
3. **Stage 2: Query Normalization**
4. **Stage 3: Query Quality Filtering**
5. **Stage 4: Semantic Deduplication and Clustering**
6. **Stage 5: Difficulty and Intent Enrichment**
7. **Stage 6: Distribution Balancing and Sampling**
8. **Stage 7: Human Adjudication**
9. **Stage 8: Versioning and Release**

Each stage has clearly defined inputs, outputs, and decision rules.

---

## 4. Stage 0: Query Specification

### 4.1 Objective
Define what counts as an eligible benchmark query before any collection begins.

### 4.2 Eligibility Criteria
A valid query should satisfy all of the following:

#### (1) Clear scholarly retrieval intent
The query should express a meaningful academic paper search need.

**Valid examples**
- papers on process rewards for long-horizon LLM agents
- recent work on memory-augmented agents with long-term memory
- papers comparing PPO and GRPO for LLM reinforcement learning

**Invalid examples**
- tell me about RL
- what is retrieval
- good papers please

#### (2) Evaluability
The query should be associated with a reasonably stable relevant-paper set or graded relevance set.

Queries that are too broad or too subjective should be excluded.

**Examples of poor evaluability**
- best papers in AI
- most important work ever in RL

#### (3) Benchmark suitability
The query should be appropriate for a paper search benchmark, not a factoid QA benchmark or a general explanation request.

**Should be excluded**
- who proposed PPO
- when was BERT published
- explain RAG to me

#### (4) Realistic user expression
The query should resemble how a real researcher might search, including concise or slightly underspecified forms.

#### (5) Publicly assessable
The query should not depend on private context unavailable to annotators.

**Should be excluded**
- papers like the method my advisor mentioned last week

### 4.3 Output
A written **query eligibility guideline** used by all later stages.

---

## 5. Stage 1: Raw Query Collection

### 5.1 Objective
Collect a high-recall pool of raw query candidates from multiple sources.

### 5.2 Input Sources

#### Source A: Real scholarly search needs
Collect real paper search queries from authentic research workflows, such as:

- literature review
- rebuttal preparation
- related work writing
- method comparison
- benchmark hunting
- follow-up paper search

Possible provenance tags:
- `real_user`
- `lab_internal`
- `paper_writing`
- `rebuttal_related`

#### Source B: Topic-to-query generation
Given a topic cluster or paper cluster, use a generation model to produce plausible user search queries.

For each topic cluster, generate multiple forms:
- exact-topic query
- constrained query
- comparison query
- survey-seeking query
- follow-up query

This source is useful for improving coverage but must be carefully filtered later.

#### Source C: Difficulty-enhanced rewrites
Starting from simple but valid queries, systematically generate harder variants by adding:

- temporal constraints
- method constraints
- task constraints
- comparison intent
- exclusion clauses
- follow-up intent
- multi-hop structure

Example:
- base: `papers on retrieval-augmented generation`
- enhanced: `papers after 2023 on retrieval-augmented generation for scientific literature search`

#### Source D: Blind-spot targeted supplementation
Identify underrepresented query categories and generate or collect additional queries specifically for those gaps.

Possible blind spots:
- ambiguous queries
- citation-chasing queries
- recent-sensitive queries
- multi-hop queries
- broad-coverage survey queries

### 5.3 Output
A raw candidate set `raw_queries.jsonl` containing all collected query candidates with source provenance.

---

## 6. Stage 2: Query Normalization

### 6.1 Objective
Convert raw queries into a consistent structured representation for downstream filtering, deduplication, and balancing.

### 6.2 Required Fields
Each raw query should be transformed into a normalized record with at least:

- `query_id`
- `raw_query`
- `surface_form`
- `canonical_form`
- `source_type`
- `source_detail`
- `language`
- `topic_guess`
- `initial_intent_guess`
- `notes`

### 6.3 Normalization Operations

#### (1) Surface cleaning
Remove non-semantic conversational wrappers such as:
- can you help me find
- please
- I want some papers about

Keep the real retrieval intent intact.

#### (2) Constraint extraction
Parse explicit or implicit constraints into metadata fields.

Examples:
- `recent papers on X` -> `temporal_constraint = recent`
- `papers after 2023 on X` -> `year_min = 2023`

#### (3) Canonicalization
Maintain both:
- `surface_form`: original or realistic user-facing expression
- `canonical_form`: normalized semantic form for internal processing

This allows natural benchmark release while improving internal deduplication quality.

### 6.4 Output
A structured query file `normalized_queries.jsonl`.

---

## 7. Stage 3: Query Quality Filtering

### 7.1 Objective
Filter out invalid or low-value queries and assign preliminary quality scores to viable candidates.

### 7.2 Hard Filtering Rules
Queries should be rejected immediately if they satisfy any of the following:

#### (1) Not evaluable
Examples:
- best ML papers ever
- useful papers for me

#### (2) Too unclear
Examples:
- RL and search
- memory and agents maybe

#### (3) Not a paper-search query
Examples:
- explain PPO to me
- what is retrieval augmented generation

#### (4) Factoid-style QA
Examples:
- who proposed PPO
- when was BERT published

#### (5) Depends on private context
Examples:
- papers like the method my advisor suggested

### 7.3 Soft Quality Scoring
Each surviving query is scored along several axes, each on a 1-5 scale:

- `clarity`
- `realism`
- `evaluability`
- `difficulty_potential`
- `agentic_value`

#### Recommended meaning of each score
- **clarity**: Is the search need understandable?
- **realism**: Does this resemble real researcher behavior?
- **evaluability**: Can relevant papers be meaningfully determined?
- **difficulty_potential**: Can this query meaningfully separate stronger systems from weaker ones?
- **agentic_value**: Does the query benefit from multi-step agent behavior rather than one-shot retrieval?

### 7.4 Recommended Aggregate Score
An optional aggregate quality score may be computed:

\[
Q = \alpha \cdot \text{clarity} + \beta \cdot \text{realism} + \gamma \cdot \text{evaluability} + \delta \cdot \text{agentic\_value}
\]

where the weights can be adjusted based on benchmark goals.

### 7.5 Output
A filtered candidate set `filtered_queries.jsonl` with rejection reasons and quality scores.

---

## 8. Stage 4: Semantic Deduplication and Clustering

### 8.1 Objective
Remove near-duplicate queries and detect over-concentrated semantic regions in the pool.

### 8.2 Near-Duplicate Removal
Queries should be considered duplicates if they are semantically equivalent or nearly equivalent, even if they differ lexically.

Examples:
- papers on process rewards for LLM agents
- process reward papers for language model agents
- work on LLM agents with process rewards

Only one representative query should be retained unless the wording difference materially changes the intended information need.

### 8.3 Clustering
After near-duplicate removal, queries should be grouped into semantic clusters.

The purpose of clustering is:
- to inspect topic concentration
- to prevent overrepresentation of head topics
- to support later balancing and split construction

### 8.4 Deduplication Principle
The goal is not merely lexical uniqueness, but **benchmark diversity**.  
Two queries may both be valid but still redundant if they test almost the same retrieval behavior.

### 8.5 Output
A deduplicated and clustered file `clustered_queries.jsonl` with fields such as:

- `cluster_id`
- `duplicate_group_id`
- `is_cluster_representative`

---

## 9. Stage 5: Difficulty and Intent Enrichment

### 9.1 Objective
Turn a filtered query list into benchmark-ready units by assigning structured metadata and difficulty labels.

### 9.2 Required Metadata
Each query should be annotated with:

- `intent_type`
- `difficulty`
- `constraint_count`
- `temporal_sensitivity`
- `ambiguity_level`
- `expected_hops`
- `expected_tool_pattern`
- `domain`

### 9.3 Intent Taxonomy
Possible intent categories include:

- `exact_topic`
- `constrained_topic`
- `comparative`
- `survey_seeking`
- `citation_chasing`
- `author_following`
- `recent_trend`
- `ambiguous`
- `multi_hop`

A query may have multiple intent tags if needed.

### 9.4 Difficulty Labeling
Difficulty should not be assigned purely by intuition.  
Use operational definitions:

#### Easy
- topic is explicit
- few or no constraints
- a single search is likely to retrieve most core relevant papers

#### Medium
- has multiple constraints
- likely benefits from reformulation or at least one expansion step

#### Hard
- many constraints and/or broad semantic spread
- relevant papers are distributed
- likely requires multiple search-expand-filter cycles and stopping decisions

### 9.5 Systematic Difficulty Enhancement
For important but overly simple queries, create harder variants using controlled transformations:

- add year constraint
- add method constraint
- add task/domain constraint
- add comparison target
- add exclusion clause
- convert to recent-trend query
- convert to follow-up search
- convert to multi-hop exploration

All generated hard variants should re-enter the normalization and filtering stages before acceptance.

### 9.6 Output
An enriched query set `enriched_queries.jsonl`.

---

## 10. Stage 6: Distribution Balancing and Sampling

### 10.1 Objective
Construct a balanced benchmark-oriented query pool rather than an accidental collection dominated by head categories.

### 10.2 Controlled Axes
The final query pool should be balanced across at least the following dimensions:

#### (1) Intent distribution
Example target:
- exact-topic: 20%
- constrained-topic: 25%
- comparative/survey: 20%
- citation/follow-up: 15%
- ambiguous: 10%
- recent-sensitive: 10%

#### (2) Difficulty distribution
Example target:
- easy: 25%
- medium: 45%
- hard: 30%

#### (3) Domain distribution
Avoid overconcentration in a single field such as LLM agents.

Suggested broad domains:
- NLP
- CV
- multimodal
- recommender systems
- time series
- RL
- IR / literature retrieval
- agents / memory / tool use

#### (4) Temporal sensitivity distribution
Ensure some queries explicitly depend on:
- recent work
- before/after year ranges
- seminal + recent combined needs

#### (5) Query style distribution
Preserve multiple styles:
- keyword-like
- natural question
- task-style request
- lightly underspecified exploratory query

### 10.3 Sampling Rule
Sampling should not be random over the whole pool.  
Instead, it should be **stratified** according to the desired benchmark composition.

### 10.4 Output
A balanced candidate pool `balanced_queries.jsonl`.

---

## 11. Stage 7: Human Adjudication

### 11.1 Objective
Conduct final human review to ensure benchmark-level quality before freezing the query pool.

### 11.2 Review Structure

#### Round 1: Individual review
Each query is reviewed for:
- naturalness
- evaluability
- redundancy
- labeling correctness
- benchmark value

Possible decisions:
- `accept`
- `revise`
- `reject`

#### Round 2: Conflict review
Queries should receive second-pass review if they show:
- difficulty disagreement
- uncertain intent labels
- questionable realism
- strong template-like style
- possible duplication
- unclear evaluability

### 11.3 Final Freeze
Accepted queries are frozen into the versioned pool for downstream annotation and benchmark release.

### 11.4 Output
A finalized pool `query_pool_vX.Y.jsonl`.

---

## 12. Stage 8: Versioning and Release

### 12.1 Objective
Maintain a transparent history of query pool evolution.

### 12.2 Versioning Rules
Every query pool release should include:

- release version
- release date
- number of added queries
- number of removed queries
- number of revised queries
- distribution summary
- change log

### 12.3 Query-Level Tracking
Each query should maintain:
- provenance
- normalization history
- filtering outcome
- metadata revisions
- adjudication record
- release inclusion status

### 12.4 Recommended Milestones
Examples:
- `query_pool_v0.1_raw`
- `query_pool_v0.2_filtered`
- `query_pool_v0.3_balanced`
- `benchmark_test_v1.0`

---

## 13. Standard Query Record Schema

A benchmark-ready query record may contain the following fields:

```json
{
  "query_id": "QP_000123",
  "raw_query": "can you help me find recent papers on process rewards for llm agents?",
  "surface_form": "recent papers on process rewards for LLM agents",
  "canonical_form": "papers on process rewards for LLM agents",
  "source_type": "real_user",
  "source_detail": "paper_writing",
  "language": "en",
  "domain": "agents_rl",
  "intent_type": ["constrained_topic", "recent_trend"],
  "difficulty": "medium",
  "constraint_count": 2,
  "temporal_sensitivity": "recent_sensitive",
  "ambiguity_level": "low",
  "expected_hops": 2,
  "expected_tool_pattern": ["search", "expand", "rerank"],
  "clarity": 5,
  "realism": 5,
  "evaluability": 4,
  "difficulty_potential": 4,
  "agentic_value": 4,
  "cluster_id": "CL_023",
  "duplicate_group_id": null,
  "is_cluster_representative": true,
  "adjudication_status": "accept",
  "release_version": "v0.3"
}
````

---

## 14. Recommended File Organization

```text
query_pool/
  guidelines/
    query_eligibility.md
    quality_scoring.md
    intent_taxonomy.md
    difficulty_guidelines.md

  raw/
    raw_queries.jsonl

  normalized/
    normalized_queries.jsonl

  filtered/
    filtered_queries.jsonl

  clustered/
    clustered_queries.jsonl

  enriched/
    enriched_queries.jsonl

  balanced/
    balanced_queries.jsonl

  adjudicated/
    query_pool_v0.1.jsonl
    query_pool_v0.2.jsonl

  logs/
    filtering_log.jsonl
    dedup_log.jsonl
    adjudication_log.jsonl
    release_notes.md
```

---

## 15. Minimal Operational Workflow

A minimal but effective first implementation can follow this process:

1. Collect 300-500 raw query candidates from multiple sources
2. Normalize all queries into a structured representation
3. Apply hard filtering and soft scoring
4. Perform semantic deduplication and clustering
5. Assign intent/difficulty/domain metadata
6. Balance the pool using stratified sampling
7. Conduct human adjudication
8. Freeze and version the accepted pool

This workflow is sufficient for building a solid v0.1 query pool.

---

## 16. Recommended First Release Target

For an initial benchmark release, a practical target is:

* **raw candidates**: 300-500
* **post-filter candidates**: 180-250
* **final accepted query pool**: 150-200

Suggested composition:

* easy: 25%
* medium: 45%
* hard: 30%

The test set should prioritize:

* realism
* diversity
* agentic value
* evaluability

---

## 17. Summary

This pipeline defines a complete and fixed workflow for constructing the query pool of a paper search agent benchmark.

Its main purpose is to ensure that the benchmark is not merely larger, but also:

* more realistic
* more difficult
* more diverse
* more reproducible
* more aligned with agentic paper search

By treating each query as a benchmark unit with structured metadata, controlled filtering, balancing, and versioning, this pipeline supports robust benchmark creation and future iterative expansion.

```

如果你愿意，我下一步可以直接继续帮你写第二份配套文档：**query quality scoring guideline**，把 `clarity / realism / evaluability / agentic_value` 这些分数怎么打进一步细化成可执行标准。
```
