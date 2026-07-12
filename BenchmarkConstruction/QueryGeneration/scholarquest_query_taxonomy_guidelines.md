# PASA-style Query Taxonomy for Topic-Seed-Based Query Generation

## Goal

This document defines a compact taxonomy for generating **PASA-style paper retrieval queries** from topic seeds. The target is **pure paper search / paper retrieval**, not open-ended research analysis, literature review writing, or advisory-style question answering.

Generated queries should remain centered on the user's intent to **find papers**. In particular, the query should usually combine a research topic with one or more **executable retrieval constraints**.

---

## What to Generate

Given a topic seed, generate queries that fall into one of the following four categories.

### 1. Method / Architecture / Capability Constrained Queries

**Definition**  
The query specifies a research topic, and further requires returned papers to satisfy a particular **method, model architecture, training paradigm, or capability property**.

**Template intuition**
- Find papers on **[topic]** that use **[method / architecture]**
- Show papers about **[topic]** with **[capability / modeling property]**
- List papers that study **[topic]** under **[training paradigm / technical approach]**

**Representative examples**
- `Give me papers which show that using a smaller dataset in large language model pre-training can result in better models than using bigger datasets.`
- `List all papers that use autoregressive transformer to generate videos.`
- `Provide me with all papers that discuss reinforcement learning training for Large Language Model agent tasks.`

**Generation guidance**
- Prefer explicit, searchable technical constraints.
- The constraint should be about **how the method works** or **what capability it has**, not about answer style.
- Good constraints include model family, training strategy, architectural property, or optimization paradigm.

---

### 2. Dataset / Scenario / Evaluation-Setting Anchored Queries

**Definition**  
The query anchors the retrieval target to a specific **dataset, task scenario, application setting, or evaluation environment**.

**Template intuition**
- Find papers on **[topic]** evaluated on **[dataset / benchmark]**
- Show papers about **[topic]** in **[application scenario / task setting]**
- Retrieve papers studying **[topic]** under **[environment / use case]**

**Representative examples**
- `Papers that propose methods based on large language models and evaluate their performance through experiments on the HotPotQA dataset.`
- `Show me some popular papers on generating textual adversarial examples for machine translation.`
- `Provide me with all papers that discuss reinforcement learning training for Large Language Model agent tasks.`

**Generation guidance**
- Use datasets, tasks, domains, or evaluation settings that materially narrow the search space.
- Anchors should help define **where / on what / in which setting** the method is studied.
- Prefer concrete anchors such as benchmark names, task names, or clear application scenarios.

---

### 3. Comparison / Claim / Conclusion-Oriented Queries

**Definition**  
The query asks for papers supporting a particular **claim, comparison, negative result, conclusion, or stance**.

**Template intuition**
- Find papers showing that **[claim]**
- Retrieve papers comparing **[A]** and **[B]** on **[topic]**
- Show papers arguing that **[method]** performs better / worse than **[baseline]**

**Representative examples**
- `Provide papers demonstrating that the self-correction of LLMs does not enhance their performance.`
- `Provide papers claiming that reinforcement learning can negatively impact the performance of supervised fine-tuned LLMs.`
- `Can LLMs detect LLM-generated text in a zero-shot manner? Do they perform better than supervised fine-tuned small classification models? Provide related papers.`

**Generation guidance**
- The query should be claim-aware rather than merely topic-aware.
- Good queries here contain explicit comparative structure, directional claims, or support/refute intent.
- Avoid vague opinions; the claim should be something papers can plausibly support with evidence.

---

### 4. Scope-Control Queries (Exhaustive / Include-Exclude)

**Definition**  
The query explicitly controls retrieval scope, such as asking for **all** matching papers, or requiring the system to **include / exclude** specific result types.

**Template intuition**
- List **all** papers on **[topic + constraint]**
- Give me **all** models / papers that satisfy **[condition]**
- Find papers on **[topic]**, but exclude **[unwanted category]**

**Representative examples**
- `List all papers that use autoregressive transformer to generate videos.`
- `Give me all visual-LLM models that are MoE architecture.`
- `I am looking for research papers on the construction of multimodal foundation models that support both visual and audio inputs. These models should be pre-trained on large-scale datasets, including visual, audio, and audio-visual data. Please exclude survey papers.`

**Generation guidance**
- Use scope operators like `all`, `list all`, `exclude`, `except`, or equivalent phrasing.
- Scope-control should still serve paper retrieval rather than asking for synthesis.
- Exclusion constraints should be concrete and operational, e.g. exclude surveys, tutorials, or non-LLM methods.

---

## Core Principle

A PASA-style query should usually be modeled as:

**topic + retrieval constraint(s)**

The retrieval constraints should come mainly from one or more of the following dimensions:
- method / capability
- dataset / scenario / evaluation setting
- claim / comparison / conclusion
- scope control (exhaustive retrieval or include-exclude filtering)

---

## What to Avoid

When generating PASA-style queries, avoid drifting into **SPAR-style research assistant behavior**.

Do **not** generate queries that primarily ask for:
- broad literature analysis
- open-ended research advice
- long-form synthesis or survey writing
- roadmap design
- future trend discussion
- multi-step analytical reports

### Bad examples
- `What are the main challenges and future directions of multimodal LLMs?`
- `Analyze recent progress in LLM-based agents and summarize the field.`
- `What should researchers focus on next for retrieval-augmented generation?`

These are not pure paper retrieval queries.

---

## Generation Rules for LLMs

When generating queries from topic seeds:

1. **Keep the output retrieval-oriented.**  
   The user should clearly be asking to find papers, not to receive a research essay.

2. **Attach at least one executable retrieval constraint whenever possible.**  
   Avoid overly broad topic-only queries unless intentionally generating a simple baseline case.

3. **Prefer explicit and searchable constraints.**  
   Constraints should be interpretable by a paper retrieval system.

4. **Keep the wording natural.**  
   Queries may be imperative or question-like, but should remain concise and realistic.

5. **Do not overload a single query with too many constraints.**  
   One or two strong constraints are usually better than many weak ones.

6. **Do not require extensive downstream synthesis.**  
   The query should primarily evaluate retrieval, filtering, and ranking ability.

---

## Suggested Output Format for Query Generation

For each topic seed, you may generate multiple candidate queries. Each query should be labeled with one primary category.

Example schema:

```json
{
  "topic_seed": "large language model agents",
  "queries": [
    {
      "category": "method_capability",
      "query": "Find papers on large language model agents trained with reinforcement learning."
    },
    {
      "category": "setting_anchor",
      "query": "Show papers on large language model agents evaluated on multi-hop QA benchmarks."
    },
    {
      "category": "claim_comparison",
      "query": "Find papers showing that reinforcement learning improves the tool-use ability of large language model agents over supervised fine-tuning."
    },
    {
      "category": "scope_control",
      "query": "List all papers on large language model agents that use external search tools, excluding survey papers."
    }
  ]
}
```

---

## Final Summary

PASA-style queries are **paper retrieval queries** centered on a research topic and strengthened by **retrieval-executable constraints**. The four main categories are:

1. **Method / Architecture / Capability Constrained**
2. **Dataset / Scenario / Evaluation-Setting Anchored**
3. **Comparison / Claim / Conclusion-Oriented**
4. **Scope-Control (Exhaustive / Include-Exclude)**

Use this taxonomy to generate topic-seed-based queries that remain faithful to the PASA setting and do not drift into open-ended research analysis.
