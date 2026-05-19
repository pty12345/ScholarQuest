# AnswerFilters

Strict cross-model answer filtering for `AnswerFinding_v2` compact outputs.

## Run

```bash
conda run -n agentRL python AnswerFilters/run_strict_cross_filter.py
```

Build Qwen-Max batch files for the first 347 full answer-detail rows without submitting them:

```bash
conda run -n agentrft python Yitong/upload/AnswerFilters/build_qwen_batch_files_first347.py
```

The default output directory is `Yitong/upload/AnswerFilters/output/qwen_batch_inputs_first347`.

The script resumes by default. Query IDs already completed with `status == "ok"` in the output files are skipped, while failed or interrupted queries can be retried on the next run.

Use `--no-resume` only when you want to reprocess all selected queries.

Use `--model-request-concurrency` or `ANSWER_FILTER_MODEL_REQUEST_CONCURRENCY` to control how many batch API requests each model sends in parallel for one query.

## Outputs

- `strict_filter_counts.jsonl`: per-query counts, model call counts, accepted/rejected counts, and elapsed time.
- `strict_filter_details.jsonl`: per-query paper judgments from qwen and deepseek, including each paper's final score/status and short reasons.
- `strict_filter_summary.json`: aggregate run summary with per-query records.
- `errors.jsonl`: query-level failures that can be retried later.
