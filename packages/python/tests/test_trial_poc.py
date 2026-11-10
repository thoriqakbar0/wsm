"""PoC controller boundaries. Cloud spending is not exercised by these tests."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import subprocess
import venv

from wsm.errors import WeightlessError
from wsm.trial import LocalTrialRunner, Trial
from wsm.trial_grading import grade_in_process
from wsm.trial_poc import Budget, inspect_weights, load_config, resume_student, run_agent, watch
from test_editing import _stack, write_tiny_model


class BudgetTests(unittest.TestCase):
    def test_reserves_before_dispatch_and_retains_failed_budget(self):
        prices = {"max_usd": 10, "modal_hourly_usd": 2, "modal_build_usd": 1, "modal_egress_per_gib": 0.04,
                  "tinker_prefill_per_million": 1, "tinker_sample_per_million": 2,
                  "tinker_train_per_million": 3, "tinker_storage_usd": 0.01}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "budget.jsonl"
            budget = Budget(prices, path)
            budget.reserve("modal_sandbox", {"seconds": 3600})
            budget.reserve("tinker_sample", {"prefill_tokens": 1_000_000, "sample_tokens": 2_000_000})
            self.assertEqual(budget.reserved, 7)
            with self.assertRaises(WeightlessError):
                budget.reserve("tinker_train", {"train_tokens": 2_000_000})
            self.assertEqual(budget.reserved, 7)
            self.assertEqual(len(path.read_text().splitlines()), 2)
            with self.assertRaises(FileExistsError):
                Budget(prices, path)


class ControllerTests(unittest.TestCase):
    def test_configured_grader_keeps_its_virtualenv_interpreter(self):
        from wsm.modal_trial import REVISION
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime"
            venv.EnvBuilder(with_pip=False, symlinks=True).create(runtime)
            interpreter = runtime / "bin/python"
            model = root / "model"
            snapshot = model / "snapshots" / REVISION
            snapshot.mkdir(parents=True)
            (snapshot / "model.safetensors").write_bytes(b"preflight fixture")
            public, private = root / "public.json", root / "private.json"
            public.write_text('[{"prompt":"a","target":"b"}]')
            private.write_text('[{"prompt":"c","target":"d"}]')
            config = {"model_source": str(model), "public_cases": str(public), "heldout_cases": str(private),
                "grader_python": str(interpreter), "instruction": "test", "scoring": "negative_loss",
                "sandbox_seconds": 60, "command_seconds": 30, "grader_seconds": 30,
                "policy": {"provider":"responses_proxy", "base_url":"http://127.0.0.1:8317/v1",
                    "model_env":"EDITOR", "api_key_env":"EDITOR_KEY", "max_tokens":64,"max_turns":1,"request_seconds":30},
                "student": {"provider":"spindle", "base_url":"https://student.invalid", "model":"student",
                    "rank":8,"loss_fn":"cross_entropy","learning_rate":0.00001},
                "budget": {"max_usd":10,"modal_hourly_usd":5,"modal_build_usd":0.5,"modal_egress_per_gib":0.04,
                    "spindle_hourly_usd":8,"spindle_seconds":1200}}
            path = root / "config.json"
            path.write_text(json.dumps(config))
            selected = load_config(path)["grader_python"]
            prefix = subprocess.check_output([selected, "-c", "import sys;print(sys.prefix)"], text=True).strip()
            self.assertEqual(Path(prefix), runtime)

    def test_unknown_prior_student_attempt_blocks_recovery_into_another_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            trial = Trial.create(source / "trial", runner=None, initial_checkpoint="/fixed",
                                 cases=[{"prompt":"a","target":"b"}])
            trial.finish()
            (source / "student-training-intent.json").write_text('{"state":"outcome_unknown"}')
            with self.assertRaises(ValueError):
                resume_student({"policy":{"provider":"responses_proxy"}}, source, root / "new-output")
            self.assertFalse((root / "new-output").exists())

    def test_large_tool_output_remains_recorded_without_overflowing_next_agent_prompt(self):
        class Runner:
            def set_history(self, events):
                pass

            def execute(self, source, checkpoint, cases):
                return {"checkpoint": "/next", "stdout": "x" * 10000 + "END_MARKER"}

        class Policy:
            config = {"max_turns": 2}
            prompts = []

            def sample(self, prompt):
                self.prompts.append(prompt)
                action = ({"action": "python", "source": "def edit(model): pass"}
                          if len(self.prompts) == 1 else {"action": "finish"})
                return {"text": json.dumps(action)}

        with tempfile.TemporaryDirectory() as temporary:
            trial = Trial.create(Path(temporary) / "trial", runner=Runner(), initial_checkpoint="/initial",
                                 cases=[{"prompt": "a", "target": "b"}])
            policy = Policy()
            run_agent(trial, policy, "test bounded feedback")
            self.assertNotIn("END_MARKER", policy.prompts[-1])
            self.assertIn("truncated", policy.prompts[-1])
            self.assertIn("END_MARKER", json.dumps(trial.history()))

    def test_preflight_rejects_private_cases_inside_image_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            public = root / "public.json"
            public.write_text('[{"prompt":"a","target":"b"}]')
            hidden = model / "heldout.json"
            hidden.write_text('[{"prompt":"c","target":"d"}]')
            config = {"model_source": str(model), "public_cases": str(public),
                      "heldout_cases": str(hidden), "grader_python": sys.executable,
                      "instruction": "test", "scoring": "negative_loss", "policy": {}, "budget": {},
                      "sandbox_seconds": 60, "command_seconds": 30, "grader_seconds": 30}
            path = root / "config.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "outside every model"):
                load_config(path)

    def test_terminal_reader_escapes_control_sequences_and_does_not_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            text = json.dumps({"kind": "frozen", "data": "\x1b[2J"}) + "\n"
            path.write_text(text)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                watch(temporary)
            self.assertNotIn("\x1b", output.getvalue())
            self.assertEqual(path.read_text(), text)

    @unittest.skipUnless(_stack(), "Torch and Transformers unavailable")
    def test_scripted_controller_freezes_and_grades_without_agent_config_or_hidden_cases(self):
        from safetensors.torch import load_file, save_file
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_tiny_model(root / "source")
            # SmolLM2 source storage is BF16; WSM publications are always F32.
            from transformers import AutoModelForCausalLM
            import torch
            original = AutoModelForCausalLM.from_pretrained(source, local_files_only=True)
            original.to(torch.bfloat16).save_pretrained(source)
            del original
            self.assertEqual(inspect_weights(source / "model.safetensors",
                             tensor="transformer.ln_f.weight", indices=[0])["values"], [1.0])
            runner = LocalTrialRunner(root / "worker", python=sys.executable)
            public = [{"prompt": "a", "target": "b"}]
            private = root / "private.json"
            private.write_text(json.dumps([{"prompt": "c", "target": "d"}]))
            unchanged = grade_in_process(source, source / "model.safetensors", private,
                                         scoring="loss_improvement", python=sys.executable, timeout=60)
            self.assertEqual(unchanged["reward"], 0.0)
            trial = Trial.create(root / "trial", runner=runner, initial_checkpoint=str(source),
                                 cases=public, max_actions=2)

            class ScriptedPolicy:
                config = {"max_turns": 3}
                actions = iter([{"action": "history"}, {"action": "python", "source":
                    "def edit(model):\n    model.weights['transformer.ln_f.weight'].mul_(0.9)\n"},
                    {"action": "finish"}])

                def sample(self, prompt):
                    if "heldout_cases" in prompt or str(private) in prompt:
                        raise AssertionError("Private grader leaked into agent prompt")
                    return {"text": json.dumps(next(self.actions)), "tokens": [1],
                            "logprobs": [-1.0], "prompt_tokens": [2], "policy": "scripted"}

            trajectory = run_agent(trial, ScriptedPolicy(), "Reduce loss on the supplied public cases.")
            self.assertEqual(len(trajectory), 3)
            final = Path(trial.observation()["checkpoint"])
            # A hostile agent config must never reach the trusted grading process.
            (final / "config.json").write_text('{"auto_map":{"AutoModelForCausalLM":"evil.Model"}}')
            exported = root / "final.safetensors"
            exported.write_bytes((final / "model.safetensors").read_bytes())
            inspected = inspect_weights(exported, tensor="transformer.ln_f.weight", indices=[0])
            self.assertAlmostEqual(inspected["values"][0], 0.9, places=6)
            grade = grade_in_process(source, exported, private, scoring="negative_loss",
                                     python=sys.executable, timeout=60)
            self.assertEqual(grade["reward"], -grade["final"]["loss"])
            self.assertEqual(grade["changed_tensors"], ["transformer.ln_f.weight"])
            history_before_grade = (trial.directory / "events.jsonl").read_bytes()
            trial.record_grade(grade)
            self.assertEqual(history_before_grade, (trial.directory / "events.jsonl").read_bytes())
            self.assertEqual(trial.observation()["state"], "frozen")
            self.assertEqual((final / "config.json").read_text(),
                             '{"auto_map":{"AutoModelForCausalLM":"evil.Model"}}')
            # Invalid tensor names fail in the trusted grader before model use.
            tensors = load_file(exported)
            tensors["unexpected"] = tensors.pop(next(iter(tensors)))
            malformed = root / "malformed.safetensors"
            save_file(tensors, malformed)
            with self.assertRaises(WeightlessError):
                grade_in_process(source, malformed, private, scoring="negative_loss",
                                 python=sys.executable, timeout=60)


class RecoverySafetyTests(unittest.TestCase):
    def test_explicit_pretraining_retry_rejects_any_training_dispatch(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, prior = root / "source", root / "prior" / "run"
            source.mkdir()
            (prior / "training").mkdir(parents=True)
            (source / "student-training-intent.json").write_text(json.dumps(
                {"state": "failed", "output": str(prior)}))
            (prior / "training/receipt.json").write_text(json.dumps(
                {"state": "failed_before_training"}))
            (prior / "budget.jsonl").write_text(json.dumps({"operation": "spindle_train"}) + "\n")
            (prior.parent / "cleanup-confirmed.json").write_text(json.dumps(
                {"confirmed": True, "remaining_tasks": 0}))
            with patch("wsm.trial_poc.Trial") as trial:
                trial.return_value.observation.return_value = {"state": "frozen"}
                with self.assertRaisesRegex(ValueError, "uncertain"):
                    resume_student({"policy": {"provider": "responses_proxy"}}, source,
                                   root / "retry", retry_pretraining=True)
            self.assertFalse((root / "retry").exists())
