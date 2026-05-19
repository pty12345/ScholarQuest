# PASA Query Generation

This directory contains a standalone PASA-style query generation workflow built from the stratified topic seeds in `QueryGeneration/topic_seeds_stratified.jsonl`.

## Files

- `pasa_types.py`: Pydantic schemas for seeds, records, traces, prompts, and config.
- `pasa_prompts.yaml`: PASA-specific prompts for plan generation, query generation, validation, and rewrite.
- `pasa_generation_config.yaml`: Runtime paths and LLM settings.
- `generate_pasa_queries.py`: Main generator script.
- `validate_pasa_queries.py`: Offline validation script.

## Output Files

Default outputs are written under `QueryGeneration/output/`:

- `pasa_query_records.jsonl`: Final accepted query records.
- `pasa_generation_trace.jsonl`: Full generation trace for each attempted `(seed, category)` pair.
- `pasa_generation_report.json`: Summary report for the generation run.
- `pasa_validation_report.json`: Summary report from offline validation.
- `pasa_llm_cache.jsonl`: Request cache for structured LLM responses.

## Query Style Principles

- Keep the topic and PASA category stable.
- Use natural English paper-retrieval wording.
- Avoid collapsing all queries into the same opening such as `Find papers ...`.
- Allow diverse human-like forms such as requests, question-like forms, and list-oriented forms.
- Prefer one-sentence queries with one strong retrieval constraint.

## Record Design

Each deduplicated topic seed is expanded into exactly four categories:

1. `method_capability`
2. `setting_anchor`
3. `claim_comparison`
4. `scope_control`

The final result file keeps the accepted query record compact, while the trace file stores:

- normalized seed metadata
- plan prompt and structured plan response
- query prompt and draft response
- validation prompt and validation response
- rewrite prompt and rewrite response when needed
- post-rewrite validation when a rewrite happens
- final status and error message

## Run

The default config uses DashScope Batch Chat for Qwen-compatible models. Set the
API key and optional model override before running:

```bash
export OPENAI_API_KEY="$DASHSCOPE_API_KEY"
export OPENAI_MODEL="qwen-max"
```

`OPENAI_BATCH_BASE_URL` can override the default batch endpoint
`https://batch.dashscope.aliyuncs.com/compatible-mode/v1`. To return to regular
OpenAI-compatible realtime calls, set `llm.mode: realtime` in
`pasa_generation_config.yaml`.

Generate queries:

```bash
python QueryGeneration/generate_pasa_queries.py --start 0 --limit 100
```

Validate generated records:

```bash
python QueryGeneration/validate_pasa_queries.py
```

Export queries for human review:

```bash
python QueryGeneration/output/export_final_queries.py --format grouped
```

## Notes

- The generator deduplicates seeds by `(stratum, acm_id, normalized_leaf_label)`.
- Resume mode is enabled by default. Existing `(seed_id, category)` pairs in the output file are skipped automatically.
- The generator only writes accepted records to the final result JSONL. Rejected or errored attempts are still preserved in the trace JSONL.
- The validation report includes top opening counts so templatic query openings are easy to spot.
