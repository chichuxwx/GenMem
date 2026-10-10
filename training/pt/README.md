# Summary continuation

Run from the repository root. First attach the SID codes from `quantization/train_rqk.py` to your raw memory entries by `sample_id`. Each entry must contain `sid.tokens` (four SID tokens) and `experience_summary` or `experience_text`. Keep the original query, dataset, sample ID, and execution-trajectory fields for subsequent CoT construction.

## Merge experiences at each SID

Serve Qwen3-32B with an OpenAI-compatible endpoint, then run:

```bash
python training/pt/merge_memory_bank.py \
  --input private_data/raw_bank.json \
  --output private_data/merged_bank.json \
  --api-base http://127.0.0.1:8000/v1 --model Qwen3-32B \
  --workers 8 --max-tokens 2048 --checkpoint-every 20
```

For an authenticated endpoint, set `OPENAI_API_KEY` in the environment. Adjust concurrency to its capacity. The original merge prompt is defined as `MERGE_PROMPT` in the script. It asks for common strategies, useful techniques, success patterns, and failure pitfalls. Singleton SIDs keep their original experience; repeated identical texts are deduplicated. By default, a cluster contributes up to 30 distinct experiences (seed 42) within a 25,000-character prompt budget. Character limits are not token limits; adjust them to your server's context window.

The output is one entry per SID with `experience_summary`, `experience_text`, and `merged_from_count`, retaining the first source's metadata and removing stale embeddings. Successful requests are checkpointed atomically. Repeat the same command to resume. Failed, empty, reasoning-only, or output-truncated responses are retried; unresolved failures prevent a final bank from being written. To change the input or generation settings, use a new output/checkpoint path.

## Build bidirectional training examples

```bash
python training/pt/build_summary_data.py \
  --input private_data/merged_bank.json \
  --output private_data/summary.jsonl
```

Register `summary.jsonl` in `private_data/dataset_info.json`:

```json
{
  "summary": {
    "file_name": "summary.jsonl",
    "formatting": "alpaca",
    "columns": {"prompt": "instruction", "query": "input", "response": "output"}
  }
}
```

Set `dataset: summary` in `configs/summary_initial.yaml`, point `model_name_or_path` at your raw-alignment PT checkpoint, and run:

```bash
FORCE_TORCHRUN=1 llamafactory-cli train configs/summary_initial.yaml
```

If summaries arrive in batches, save the initial bank subset and build the next dataset with `--exclude-sids-from private_data/initial_subset.json`. This excludes already-trained SIDs. Register the resulting dataset, update `dataset` in `configs/summary.yaml`, and train from `artifacts/summary_initial`. If the full merged bank was ready for the first pass, use that final model for MT directly rather than training the same summaries again.
