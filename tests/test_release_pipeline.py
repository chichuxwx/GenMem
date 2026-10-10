import ast
import copy
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "LlamaFactory/pt"), str(ROOT / "LlamaFactory/mt")]
import build_summary_data as summary
import generate_teacher_cot as teacher
import merge_memory_bank as merge
import retarget_memr_rl_prompt as retarget
import stitch_memr_two_stage_cot as stitch_memr
import stitch_meme_thinking_cot as stitch_meme


SID = "<SID_L1_1><SID_L2_2><SID_L3_3><SID_L4_4>"


def client_with(content, finish="stop"):
    choice = SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(choices=[choice])
    )))


class SummaryTest(unittest.TestCase):
    def test_singleton_preserved_and_pairs_merged(self):
        entries = [{"experience_text": "First technique"}, {"experience_text": "Second technique"}]
        self.assertEqual(merge.merge_experiences(None, "test", SID, entries[:1]), "First technique")
        self.assertEqual(merge.merge_experiences(client_with("Shared technique"), "test", SID, entries), "Shared technique")

    def test_rejects_truncation_reasoning_and_empty(self):
        entries = [{"experience_text": "First"}, {"experience_text": "Second"}]
        for content, finish in [("incomplete", "length"), ("<think>hidden</think>", "stop"), ("", "stop")]:
            with self.subTest(content=content), self.assertRaises(RuntimeError):
                merge.merge_experiences(client_with(content, finish), "test", SID, entries, max_retries=1)

    def test_prompt_budget_and_bidirectional_labels(self):
        prompt = merge.build_merge_prompt(SID, ["x" * 3000, "y" * 3000], 1500)
        self.assertLessEqual(len(prompt), 1500)
        rows, stats = summary.convert_entries([{"sid": {"tokens": SID}, "experience_summary": "Shared technique"}])
        self.assertEqual(stats.output_records, 2)
        self.assertEqual(rows[0]["output"], SID)
        self.assertEqual(rows[1]["input"], SID)

    def test_singleton_cli_resume_and_changed_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "raw.json", root / "merged.json"
            record = {"sid": {"tokens": SID}, "experience_text": "Reusable technique", "embedding": [1]}
            source.write_text(json.dumps([record]))
            command = [sys.executable, str(ROOT / "LlamaFactory/pt/merge_memory_bank.py"), "--input", str(source), "--output", str(output)]
            for _ in range(2):
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            merged = json.loads(output.read_text())[0]
            self.assertEqual(merged["merged_from_count"], 1)
            self.assertNotIn("embedding", merged)
            record["experience_text"] = "Changed input"
            source.write_text(json.dumps([record]))
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Checkpoint input/settings changed", result.stderr)

    def test_failures_do_not_publish_partial_bank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "raw.json", root / "merged.json"
            source.write_text(json.dumps([
                {"sid": {"tokens": SID}, "experience_text": "First"},
                {"sid": {"tokens": SID}, "experience_text": "Second"},
            ]))
            args = ["merge", "--input", str(source), "--output", str(output)]
            with patch.object(sys, "argv", args), patch.object(merge, "merge_sid_job", side_effect=RuntimeError("mock failure")):
                with self.assertRaises(RuntimeError):
                    merge.main()
            self.assertFalse(output.exists())
            self.assertEqual(json.loads(Path(str(output) + ".ckpt.json").read_text())["merged"], {})


class TeacherTest(unittest.TestCase):
    def test_memr_has_two_calls_without_source_mutation(self):
        row = {"query": "Question", "experience": "Technique", "trajectory": "Execution"}
        original = copy.deepcopy(row)
        prompts = json.loads((ROOT / "LlamaFactory/mt/teacher_prompts.json").read_text())
        with patch.object(teacher, "request_field", side_effect=["Need relevant techniques", "Adapted technique"]) as call:
            result = teacher.generate_record(3, row, SimpleNamespace(agent="memretriever"), prompts)
        self.assertEqual(call.call_count, 2)
        self.assertEqual(call.call_args_list[0].args[2], {"query": row["query"]})
        self.assertEqual(result["source_index"], 3)
        self.assertEqual(row, original)
        self.assertEqual(result["rewritten_exp"], "Adapted technique")

    def test_meme_keeps_evolution_target_and_empty_observation(self):
        row = {"query": "Question", "experience": "", "trajectory": "Execution", "evo_exp": "Merged reusable skill"}
        prompts = json.loads((ROOT / "LlamaFactory/mt/teacher_prompts.json").read_text())
        with patch.object(teacher, "request_field", return_value="Trajectory analysis") as call:
            result = teacher.generate_record(0, row, SimpleNamespace(agent="memevolver"), prompts)
        self.assertEqual(call.call_count, 1)
        self.assertNotIn("experience", call.call_args.args[2])
        self.assertEqual(result["evo_exp"], row["evo_exp"])
        self.assertEqual(result["experience"], "")

    def test_resume_rejects_duplicate_index_and_changed_source(self):
        row = {"query": "Question", "experience": "Skill", "trajectory": "Execution"}
        result = dict(row, source_index=0, task_type="memretriever", thinking="Reason", rewritten_exp="Support")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out.jsonl"
            output.write_text(json.dumps(result) + "\n")
            self.assertEqual(len(teacher.load_completed(output, [row], "memretriever")), 1)
            output.write_text((json.dumps(result) + "\n") * 2)
            with self.assertRaises(ValueError):
                teacher.load_completed(output, [row], "memretriever")
            output.write_text(json.dumps(dict(result, query="Changed")) + "\n")
            with self.assertRaises(ValueError):
                teacher.load_completed(output, [row], "memretriever")


class PromptTest(unittest.TestCase):
    def test_mt_rl_prompt_matches(self):
        tree = ast.parse((ROOT / "verl/recipe/genmem/build_training_data.py").read_text())
        values = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "SYSTEM_PROMPT" for t in node.targets)]
        self.assertEqual(values, [retarget.RL_SYSTEM_PROMPT])

    def test_retarget_preserves_conversation_and_rejects_multiple_calls(self):
        call = '<tool_call>' + json.dumps({"name": "memory_lookup", "arguments": {"sid_list": [SID]}}) + '</tool_call>'
        row = {"sid": SID, "conversations": [
            {"from": "system", "value": "Old prompt"},
            {"from": "human", "value": "Question"},
            {"from": "gpt", "value": "<think>Reason</think>" + call},
            {"from": "observation", "value": SID + " Skill"},
            {"from": "gpt", "value": "<answer>Support</answer>"},
        ]}
        result = retarget.retarget([row])[0]
        self.assertEqual(result["conversations"][1:], row["conversations"][1:])
        self.assertEqual(row["conversations"][0]["value"], "Old prompt")
        row["conversations"][2]["value"] += call
        with self.assertRaises(ValueError):
            retarget.retarget([row])


class AssemblyTest(unittest.TestCase):
    def test_memr_assembly_then_prompt_alignment(self):
        source = {"query": "How to check a result?", "experience": "Use an independent check.", "trajectory": "Original execution trace reserved for source matching."}
        teacher_row = dict(source, source_index=0, task_type="memretriever", thinking="Need verification techniques.", rewritten_exp="Check the result independently.")
        lookup = [{"source_index": 0, "sid": SID, "dataset": "synthetic", "sample_id": "example"}]
        records, stats = stitch_memr.build_sft_records([teacher_row], [source], lookup, 1, 0)
        aligned = retarget.retarget(records)[0]
        self.assertEqual(stats["trajectory_leaks"], 0)
        self.assertEqual(aligned["conversations"][0]["value"], retarget.RL_SYSTEM_PROMPT)
        self.assertIn("Check the result independently.", aligned["conversations"][-1]["value"])

    def test_meme_assembly_for_insert_and_revise(self):
        for initial in ("", "Previously learned technique."):
            with self.subTest(initial=initial):
                source = {"query": "How to verify a result?", "experience": initial, "trajectory": "Tried a calculation and checked it independently.", "evo_exp": "Combine calculation with an independent verification method."}
                replay = dict(source, sid=SID, dataset="synthetic", trajectory_source_id="a", existing_memory_source_id="b" if initial else "", merged_from_count=2 if initial else 1, update_type="merge_existing" if initial else "insert_new")
                rows, stats = stitch_meme.build_sft_records({0: dict(source, thinking="The trajectory suggests a reusable verification strategy.")}, [source], [replay])
                self.assertEqual(stats["output_records"], 1)
                self.assertEqual(rows[0]["conversations"][-1]["value"], "<answer>\n" + source["evo_exp"] + "\n</answer>")
                self.assertIn(SID, rows[0]["conversations"][2]["value"])


class LayoutTest(unittest.TestCase):
    def test_training_configs_resolve_from_repository_root(self):
        for path in (ROOT / "LlamaFactory/configs").glob("*.yaml"):
            match = re.search(r"^deepspeed:\s*(\S+)", path.read_text(), re.M)
            self.assertIsNotNone(match, str(path))
            self.assertTrue((ROOT / match.group(1)).is_file(), str(path))
        self.assertIn("-e ./LlamaFactory", (ROOT / "requirements-pt.txt").read_text())
        self.assertIn("-e ./verl", (ROOT / "requirements-rl.txt").read_text())

    def test_rl_launcher_default_paths_after_move(self):
        env = dict(os.environ)
        env.pop("VERL_ROOT", None)
        env.pop("TOOL_CONFIG", None)
        for name in ("MODEL_PATH", "TRAIN_DATA_PATH", "TEST_DATA_PATH", "MEMORY_BANK_PATH", "SID_CONFIG_PATH", "DEERFLOW_BASE_URL", "DEERFLOW_MODEL", "DEERFLOW_EMAIL", "DEERFLOW_PASSWORD"):
            env[name] = "test-only-placeholder"
        result = subprocess.run(
            ["bash", "-c", 'source "$1"; printf "%s\\n" "$REPO_DIR" "$PROJECT_DIR" "$RL_DIR" "$TOOL_CONFIG"', "test", str(ROOT / "verl/recipe/genmem/config/local_config_v33.sh")],
            env=env, capture_output=True, text=True, cwd="/tmp",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        resolved = [Path(line).resolve() for line in result.stdout.splitlines()]
        self.assertEqual(resolved, [ROOT, ROOT / "verl", ROOT / "verl/recipe/genmem", ROOT / "verl/recipe/genmem/config/mem_tool_config_48_16_8_8.yaml"])


if __name__ == "__main__":
    unittest.main()
