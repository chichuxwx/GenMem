# CoT mid-training data

Use DeepSeek V4 Flash to generate the teacher fields, then assemble them into MemRetriever and MemEvolver tool-call conversations. Run the commands below from the repository root.

## Prepare teacher inputs

```bash
python LlamaFactory/mt/build_cot_generation_inputs.py \
  --raw-bank private_data/raw_bank.json \
  --merged-bank private_data/merged_bank.json \
  --memr-output private_data/memr_teacher_input.json \
  --meme-output private_data/meme_teacher_input.json
```

The default sample counts are 5,000 for MemR and 4,000 update + 1,000 empty-memory insertion examples for MemE. The input banks are JSON arrays. Raw entries supply the query, trajectory, experience, and SID; merged entries supply the same SID, its summary, and `merged_from_count`.

For a two-source MemE example, one source supplies the trajectory and the other supplies the existing experience; the merged summary becomes `evo_exp`. For a singleton, the existing `experience` is empty and the summary becomes `evo_exp`. The builder filters duplicate examples and identical trajectory/experience/target text.

Keep the input arrays, bank files, record order, and sampling seed unchanged until assembly finishes. The stitchers use them to reconstruct the SID labels. Assign each input its zero-based array index as `source_index` before submitting teacher requests.

## Generate CoT with DeepSeek V4 Flash

Use your DeepSeek V4 Flash endpoint and credentials for the following requests. Collect the requested text from the model response and save the results as JSONL. Copy source fields from the original input record in your request driver, rather than asking the teacher to reproduce them.

The release provides an OpenAI-compatible request driver. Set `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`, and `DEEPSEEK_API_KEY` for your DeepSeek V4 Flash deployment, then run:

```bash
python LlamaFactory/mt/generate_teacher_cot.py \
  --input private_data/memr_teacher_input.json \
  --output private_data/memr_teacher_output.jsonl \
  --agent memretriever --api-base "$DEEPSEEK_BASE_URL" \
  --model "$DEEPSEEK_MODEL" --workers 4
python LlamaFactory/mt/generate_teacher_cot.py \
  --input private_data/meme_teacher_input.json \
  --output private_data/meme_teacher_output.jsonl \
  --agent memevolver --api-base "$DEEPSEEK_BASE_URL" \
  --model "$DEEPSEEK_MODEL" --workers 4
```

Repeat the same command to retry failed records without regenerating successful ones. The accompanying `.meta.json` prevents resuming with changed data or generation settings. Keep both files together and run only one writer per output file. The editable prompts in `teacher_prompts.json` implement the two-stage MemR / one-stage MemE contract; they are release templates reconstructed from that contract, rather than an archived copy of the original teacher requests. The historical teacher was DeepSeek V4 Flash.

| Agent | Teacher generation | Fields retained from the input |
| --- | --- | --- |
| MemR | First generate `thinking` describing the query's memory needs; then generate `rewritten_exp` adapting the retrieved experience to the query. | `query`, `experience`, `trajectory` |
| MemE | Generate `thinking` analysing the trajectory and the kind of reusable memory it should update. | `query`, `experience`, `trajectory`, `evo_exp` |

For MemR, the first response should describe the query's relevant domain, constraints, and useful experience. In the second request, provide the query and retrieved `experience` and ask for concise, actionable support. The final answer is the rewritten experience, rather than an answer to the downstream task.

For MemE, base the reasoning on the query and trajectory. The reasoning appears before `memory_lookup` in the training conversation, so it should describe what to look up without claiming to have already received the tool response. Keep `evo_exp` as the supervision target from the input. The stitcher handles the existing-memory observation or the empty-memory response.

Store plain text in `thinking` and `rewritten_exp`; the stitchers add the `<think>` and `<answer>` tags and the SID tool call. SID labels are recovered from the banks during assembly.

A MemR output record has this structure (the strings below illustrate the schema):

```json
{
  "source_index": 0,
  "task_type": "memretriever",
  "query": "Copy the original query exactly",
  "experience": "Copy the original experience exactly",
  "trajectory": "Copy the original trajectory exactly",
  "thinking": "Teacher-generated retrieval reasoning",
  "rewritten_exp": "Teacher-generated query-specific experience"
}
```

A MemE output record has this structure:

```json
{
  "source_index": 0,
  "task_type": "memevolver",
  "query": "Copy the original query exactly",
  "experience": "Copy the original experience exactly, including an empty string",
  "trajectory": "Copy the original trajectory exactly",
  "evo_exp": "Copy the original evolution target exactly",
  "thinking": "Teacher-generated trajectory analysis"
}
```

Retry failed requests under the same `source_index`. Keep one successful output per index. The commands below assume all 5,000 records succeeded in each stream.

## Assemble and validate

```bash
python LlamaFactory/mt/stitch_memr_two_stage_cot.py \
  --cot-input private_data/memr_teacher_output.jsonl \
  --base-input private_data/memr_teacher_input.json \
  --raw-bank private_data/raw_bank.json \
  --merged-bank private_data/merged_bank.json \
  --output private_data/memr_cot_stitched.json \
  --lookup-output private_data/memr_sid_lookup.json \
  --validation-report private_data/memr_validation.json \
  --base-samples 5000 --expected-cot-rows 5000

python LlamaFactory/mt/retarget_memr_rl_prompt.py \
  --input private_data/memr_cot_stitched.json \
  --output private_data/memr_cot.json --expected-records 5000

python LlamaFactory/mt/stitch_meme_thinking_cot.py \
  --teacher-input private_data/meme_teacher_output.jsonl \
  --base-input private_data/meme_teacher_input.json \
  --raw-bank private_data/raw_bank.json \
  --merged-bank private_data/merged_bank.json \
  --output private_data/meme_cot.json \
  --lookup-output private_data/meme_sid_lookup.json \
  --validation-report private_data/meme_validation.json \
  --expected-base-rows 5000 --expected-teacher-rows 5000 \
  --expected-missing-indices '' --expected-source-mismatches ''
```

The stitchers check source indices, unchanged source fields, valid SIDs, and deterministic reconstruction against the banks. MemR also checks that the full teacher-input trajectory is absent from the final conversation. The MemE command explicitly clears the exception flags from the historical run, which produced 4,999 successful teacher records; new runs should set these flags to match their own validated outputs.

MemR records follow `query -> thinking -> memory_lookup(SID) -> experience -> rewritten experience`. MemE records follow `query + trajectory -> thinking -> memory_lookup(SID) -> initial experience -> evo_exp`.

## Register datasets and train

Add these entries to `private_data/dataset_info.json`, preserving any PT or summary entries already present:

```json
{
  "memr_cot": {
    "file_name": "memr_cot.json",
    "formatting": "sharegpt",
    "columns": {"messages": "conversations", "tools": "tools"}
  },
  "meme_cot": {
    "file_name": "meme_cot.json",
    "formatting": "sharegpt",
    "columns": {"messages": "conversations", "tools": "tools"}
  }
}
```

Set `dataset` to `memr_cot` in `LlamaFactory/configs/mt_memr.yaml` and to `meme_cot` in `LlamaFactory/configs/mt_meme.yaml`. Point both configs at your summary-continuation PT checkpoint, then run:

```bash
FORCE_TORCHRUN=1 llamafactory-cli train LlamaFactory/configs/mt_memr.yaml
FORCE_TORCHRUN=1 llamafactory-cli train LlamaFactory/configs/mt_meme.yaml
```
