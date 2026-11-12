"""Benchmark contracts at configuration, installed command, and model boundaries."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from wsm.benchmark import digest, load_protocol, preflight, read, regrade, reservation, run, tree_digest
from wsm.benchmark_report import report, summarize
from test_editing import _stack, write_tiny_model


def fixture(root, source, actions=None):
    def save(name, value):
        path = root / name
        path.write_text(json.dumps(value))
        return {"path": str(path), "sha256": digest(path)}
    public = save("public.json", [{"prompt": "a", "target": "b"}])
    private = save("private.json", [{"prompt": "a b", "target": "c"}, {"prompt": "a", "target": "c b"}])
    regression = save("regression.json", [{"prompt": "c", "target": "d"}])
    script = root / "edit.py"
    script.write_text("""def edit(model):
    model.weights['transformer.ln_f.weight'].zero_()
    model.weights['transformer.ln_f.bias'].fill_(1)
    model.weights['lm_head.weight'].zero_()
    model.weights['lm_head.weight'][0].fill_(100)
""")
    spoof = ("def edit(model):\n    print('untrusted-output-sentinel')\n"
             "    model.evaluate = lambda cases: {'loss': -999, 'cases': []}\n")
    tape = save("actions.json", actions or [{"action": "python", "source": spoof}, {"action": "finish"}])
    lock = root / "runtime.txt"
    lock.write_text("synthetic fixture; runtime versions also frozen in manifest\n")
    config = {"format": "wsm.benchmark.v1", "name": "synthetic", "repetitions": 1,
              "model": {"path": str(source), "id": "tiny-gpt2", "revision": "synthetic-v1", "sha256": tree_digest(source)},
              "provenance": {"code_revision": "0" * 40, "dependency_lock": {"path": str(lock), "sha256": digest(lock)}},
              "runtime": {"backend": "local_trusted", "grader_python": sys.executable, "actions": tape},
              "limits": {"turns": 3, "evaluations": 1, "output_tokens": 4096, "wall_seconds": 120,
                         "command_seconds": 60, "grader_seconds": 60},
              "budget": {"max_usd": 0, "prior_reserved_usd": 0, "price_checked_on": None,
                         "rates": {"build_usd": 0, "sandbox_hourly_usd": 0, "egress_per_gib": 0, "teacher_request_usd": 0}},
              "statistics": {"seed": 7, "bootstrap_samples": 100},
              "review": {"data_rights": False, "semantic_overlap": False, "power": False,
                         "owner": None, "private_artifact_store": None},
              "regression": regression,
              "tasks": [{"id": "task", "family": "completion", "instruction": "Improve public completion loss",
                         "public": public, "private": private, "script": {"path": str(script), "sha256": digest(script)}}]}
    path = root / "config.json"
    path.write_text(json.dumps(config))
    return path


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / "source"
        source.mkdir()
        (source / "model.safetensors").write_bytes(b"preflight never loads model tensors")
        self.path = fixture(self.root, source)

    def update(self, change):
        config = read(self.path)
        change(config)
        self.path.write_text(json.dumps(config))

    def test_preflight_is_offline_and_freezes_all_arms(self):
        result = preflight(self.path)
        self.assertEqual(result["external_calls"], 0)
        self.assertEqual(result["scheduled_trials"], 3)
        self.assertFalse(result["confirmatory"])
        self.assertEqual([s.arm for s in load_protocol(self.path).schedule], ["unchanged", "scripted", "agent"])

    def test_stochastic_scripts_repeat_the_agent_schedule(self):
        self.update(lambda c: c.update(repetitions=3))
        self.update(lambda c: c["tasks"][0].update(scripted_stochastic=True))
        schedule = load_protocol(self.path).schedule
        self.assertEqual([(s.arm, s.repetition) for s in schedule],
                         [("unchanged", 0), ("scripted", 0), ("scripted", 1), ("scripted", 2),
                          ("agent", 0), ("agent", 1), ("agent", 2)])

    def test_rejects_mutated_inputs_and_normalized_cross_partition_duplicates(self):
        (self.root / "public.json").write_text('[]')
        with self.assertRaisesRegex(ValueError, "digest"):
            preflight(self.path)
        (self.root / "public.json").write_text(json.dumps([{"prompt": " a b ", "target": " c "}]))
        self.update(lambda c: c["tasks"][0]["public"].update(sha256=digest(self.root / "public.json")))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            preflight(self.path)

    def test_rejects_invalid_limits_nonfinite_prices_and_unknown_fields(self):
        original = self.path.read_text()
        for mutate in (lambda c: c["limits"].update(turns=True),
                       lambda c: c["budget"]["rates"].update(build_usd=float("nan")),
                       lambda c: c.update(secret="must not accept arbitrary metadata"),
                       lambda c: c["statistics"].update(bootstrap_samples=0)):
            with self.subTest(mutate=mutate):
                self.path.write_text(original)
                self.update(mutate)
                with self.assertRaises(ValueError):
                    preflight(self.path)

    def test_local_execution_needs_explicit_trust_and_output_is_exclusive(self):
        with self.assertRaisesRegex(ValueError, "local-trusted"):
            run(self.path, self.root / "run")
        output = self.root / "existing"
        output.mkdir()
        (output / "user-data").write_text("keep")
        with self.assertRaises(FileExistsError):
            run(self.path, output, local_trusted=True)
        self.assertEqual((output / "user-data").read_text(), "keep")

    def test_reservation_covers_every_possible_published_checkpoint(self):
        config = read(self.path)
        config["runtime"]["backend"] = "modal"
        config["limits"]["turns"] = 4
        config["budget"]["rates"] = {"build_usd": 1, "sandbox_hourly_usd": 30,
                                      "egress_per_gib": 2, "teacher_request_usd": 0.25}
        # Each export reserves 600 MiB, an 8 MiB diagnostic allowance, and one sentinel byte.
        self.assertEqual(reservation(config, "agent"), 7.75 + 8 / 1024**3)
        self.assertEqual(reservation(config, "scripted"), 3.1875 + 2 / 1024**3)
        self.assertEqual(reservation(config, "unchanged"), 0)

    def test_missing_editor_credentials_are_rejected_before_campaign_creation(self):
        from wsm.modal_trial import MODEL, REVISION
        config = read(self.path)
        policy = self.root / "policy.json"
        policy.write_text(json.dumps({"provider": "responses_proxy", "base_url": "https://example.invalid/v1",
                                     "model_env": "WSM_BENCHMARK_TEST_MODEL", "api_key_env": "WSM_BENCHMARK_TEST_KEY",
                                     "max_tokens": 4096, "max_turns": 3, "request_seconds": 30}))
        config["model"].update(id=MODEL, revision=REVISION)
        config["runtime"] = {"backend": "modal", "grader_python": sys.executable,
                             "policy": {"path": str(policy), "sha256": digest(policy)}}
        config["budget"]["price_checked_on"] = "2026-11-11"
        config["budget"]["rates"] = {"build_usd": 1, "sandbox_hourly_usd": 1,
                                      "egress_per_gib": 1, "teacher_request_usd": 1}
        self.path.write_text(json.dumps(config))
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "environment"):
                preflight(self.path)
            output = self.root / "missing-credentials"
            with self.assertRaisesRegex(ValueError, "environment"):
                run(self.path, output)
        self.assertFalse(output.exists())


class StatisticsTests(unittest.TestCase):
    def test_equal_family_and_task_weights_with_unequal_repetitions(self):
        # Equal families: ((2 + 4) / 2 + 9) / 2 = 6. One task has many more repeats.
        groups = {"a": {"t1": [(2, 1, 0)] * 10, "t2": [(4, 3, 2)]}, "b": {"t3": [(9, 8, 4)]}}
        result = summarize(groups, seed=3, samples=200)
        self.assertEqual(result["improvement_unchanged"]["mean"], 6)
        self.assertEqual(result["improvement_scripted"]["mean"], 5)
        self.assertEqual(result["regression"]["mean"], 2.5)
        self.assertEqual(result["counts"]["agent_repetitions"], 12)
        self.assertEqual(result, summarize(groups, seed=3, samples=200))

    def test_paired_bootstrap_has_exact_interval_for_constant_effect(self):
        result = summarize({"a": {"x": [(3, 2, -1)] * 5, "y": [(3, 2, -1)]}}, seed=4, samples=100)
        self.assertEqual(result["improvement_unchanged"]["ci95"], [3, 3])
        self.assertEqual(result["regression"]["ci95"], [-1, -1])
        self.assertIsNone(summarize({}, seed=1, samples=100))

    def test_bootstrap_resamples_tasks_instead_of_treating_repeats_as_tasks(self):
        result = summarize({"a": {"x": [(0, 0, 0)] * 20, "y": [(10, 10, 10)] * 20}}, seed=7, samples=1000)
        self.assertEqual(result["improvement_unchanged"]["ci95"], [0, 10])


class ModelCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _stack() is None:
            raise unittest.SkipTest("model dependencies unavailable")
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.source = write_tiny_model(cls.root / "source")
        cls.config = fixture(cls.root, cls.source)
        cls.output = cls.root / "run"
        run(cls.config, cls.output, local_trusted=True)
        cls.result = report(cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_independent_grading_pristine_resets_and_token_counts(self):
        self.assertTrue(self.result["complete"], self.result)
        unchanged, scripted, agent = self.result["rows"]
        self.assertGreater(scripted["nll"], unchanged["nll"] + 100)
        self.assertEqual(agent["nll"], unchanged["nll"])
        self.assertEqual(agent["regression_nll"], unchanged["regression_nll"])
        self.assertEqual(agent["target_tokens"], 3)
        self.assertEqual(agent["case_count"], 2)
        self.assertNotAlmostEqual(agent["token_weighted_nll"], agent["nll"], places=8)
        self.assertEqual(self.result["decision"], "exploratory")
        self.assertIsNone(self.result["success_rate"])
        self.assertEqual(tree_digest(self.source), read(self.config)["model"]["sha256"])

    def copy_campaign(self, name):
        target = self.root / name
        shutil.copytree(self.output, target)
        return target

    def test_offline_report_cli_is_identical_without_provider_modules_or_input_files(self):
        script = """import importlib.abc, json, sys
class NoProviders(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('modal', 'tinker', 'torch', 'transformers', 'httpx'):
            raise AssertionError('offline report imported ' + fullname)
sys.meta_path.insert(0, NoProviders())
from wsm.benchmark_report import report
print(json.dumps(report(sys.argv[1]), sort_keys=True))
"""
        copied = self.copy_campaign("relocated")
        original_source = self.source.with_name("source-hidden")
        self.source.rename(original_source)
        original_private = self.root / "private.json"
        original_private.rename(self.root / "private-hidden.json")
        try:
            completed = subprocess.run([sys.executable, "-c", script, str(copied)], capture_output=True, text=True)
        finally:
            original_source.rename(self.source)
            (self.root / "private-hidden.json").rename(original_private)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout), self.result)
        report_text = json.dumps(self.result)
        self.assertNotIn('"prompt"', report_text)
        self.assertNotIn('"target"', report_text)
        self.assertNotIn('target_nll', report_text)
        self.assertNotIn('untrusted-output-sentinel', report_text)
        history = (self.output / "trials/task.agent.0000/trial/events.jsonl").read_text()
        self.assertIn('untrusted-output-sentinel', history)
        self.assertIn('-999', history)
        self.assertEqual(report(self.output), self.result)

    def test_tampered_receipt_and_missing_checkpoint_are_refused(self):
        for name, target in (("tampered", "grade"), ("missing", "artifact")):
            with self.subTest(target=target):
                copied = self.copy_campaign(name)
                if target == "grade":
                    receipt = copied / "restricted/task.agent.0000/grade.json"
                    receipt.write_text(receipt.read_text() + " ")
                else:
                    outcome = read(copied / "trials/task.agent.0000/frozen.json")
                    outcome["weights"] = str(copied / "absent.safetensors")
                    (copied / "trials/task.agent.0000/frozen.json").write_text(json.dumps(outcome))
                with self.assertRaises((ValueError, FileNotFoundError)):
                    report(copied)

    def test_interrupted_trial_is_counted_without_scores(self):
        copied = self.copy_campaign("partial")
        (copied / "trials/task.agent.0000/outcome.json").unlink()
        result = report(copied)
        self.assertFalse(result["complete"])
        self.assertEqual(result["counts"]["outcome_unknown"], 1)
        self.assertEqual(result["scheduled_agent_trials"], 1)
        self.assertIsNone(result["rows"][-1]["nll"])
        self.assertIsNone(result["summary"])

    def test_frozen_artifact_can_be_regraded_without_rerunning_edits(self):
        before = digest(self.output / "trials/task.agent.0000/trial/events.jsonl")
        self.assertTrue(regrade(self.output, "task.agent.0000")["matches"])
        self.assertEqual(digest(self.output / "trials/task.agent.0000/trial/events.jsonl"), before)

    def test_budget_exhaustion_blocks_isolated_editing_before_provider_import(self):
        from wsm.modal_trial import MODEL, REVISION
        inputs = self.root / "blocked-inputs"
        inputs.mkdir()
        config_path = fixture(inputs, self.source)
        config = read(config_path)
        policy = inputs / "policy.json"
        policy.write_text(json.dumps({"provider": "responses_proxy", "base_url": "https://example.invalid/v1",
                                     "model_env": "WSM_BENCHMARK_TEST_MODEL", "api_key_env": "WSM_BENCHMARK_TEST_KEY",
                                     "max_tokens": 4096, "max_turns": 3, "request_seconds": 30}))
        config["model"].update(id=MODEL, revision=REVISION)
        config["runtime"] = {"backend": "modal", "grader_python": sys.executable,
                             "policy": {"path": str(policy), "sha256": digest(policy)}}
        config["budget"]["price_checked_on"] = "2026-11-11"
        config["budget"]["rates"] = {"build_usd": 1, "sandbox_hourly_usd": 1,
                                      "egress_per_gib": 1, "teacher_request_usd": 1}
        config_path.write_text(json.dumps(config))
        output = self.root / "blocked"
        with patch.dict(os.environ, {"WSM_BENCHMARK_TEST_MODEL": "synthetic-test-policy",
                                     "WSM_BENCHMARK_TEST_KEY": "synthetic-no-dispatch"}):
            run(config_path, output)
        result = report(output)
        self.assertEqual(result["counts"]["completed"], 1)
        self.assertEqual(result["counts"]["budget_blocked"], 2)
        self.assertEqual(result["cost"]["reserved_usd"], 0)
        self.assertNotIn("modal", sys.modules)

    def test_nonfinite_artifact_retains_machine_readable_grader_error(self):
        from safetensors.torch import load_file, save_file
        from wsm.errors import WeightlessError
        from wsm.trial_grading import grade_in_process
        tensors = load_file(self.source / "model.safetensors")
        tensors[next(iter(tensors))].view(-1)[0] = float("nan")
        malformed = self.root / "nonfinite.safetensors"
        save_file(tensors, malformed)
        with self.assertRaises(WeightlessError) as raised:
            grade_in_process(self.source, malformed, self.root / "private.json", scoring="loss_improvement",
                             python=sys.executable, timeout=60, detailed=True)
        self.assertEqual(raised.exception.code, "nonfinite")
        malformed.write_bytes(b"not a safetensors header")
        with self.assertRaises(WeightlessError) as raised:
            grade_in_process(self.source, malformed, self.root / "private.json", scoring="loss_improvement",
                             python=sys.executable, timeout=60, detailed=True)
        self.assertEqual(raised.exception.code, "invalid_checkpoint")

    def test_grader_runtime_failure_is_not_an_invalid_artifact(self):
        from wsm.errors import WeightlessError
        from wsm.trial_grading import grade_in_process
        cases = self.root / "beyond-context.json"
        cases.write_text(json.dumps([{"prompt": "a " * 40, "target": "b"}]))
        with self.assertRaises(WeightlessError) as raised:
            grade_in_process(self.source, self.source / "model.safetensors", cases, scoring="loss_improvement",
                             python=sys.executable, timeout=60, detailed=True)
        self.assertEqual(raised.exception.code, "final_grading_failed")


class FailureCampaignTests(unittest.TestCase):
    def test_lifecycle_rejection_does_not_request_another_policy_action(self):
        from wsm.benchmark_runtime import _ActionLimits
        from wsm.errors import WeightlessError
        from wsm.trial import Trial
        from wsm.trial_poc import run_agent

        class Policy:
            config = {"max_turns": 3}
            requests = 0

            def sample(self, prompt):
                self.requests += 1
                return {"text": '{"action":"evaluate"}'}

        class Runner:
            def set_history(self, events):
                pass

        with tempfile.TemporaryDirectory() as temporary:
            trial = Trial.create(Path(temporary) / "trial", runner=_ActionLimits(Runner(), 0),
                                 initial_checkpoint="/source", cases=[{"prompt": "a", "target": "b"}])
            policy = Policy()
            with self.assertRaises(WeightlessError):
                run_agent(trial, policy, "Evaluate public cases")
            self.assertEqual(policy.requests, 1)
            self.assertEqual(trial.observation()["state"], "failed")

    def test_stochastic_script_comparisons_use_the_matching_repetition(self):
        if _stack() is None:
            self.skipTest("model dependencies unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_tiny_model(root / "source")
            config_path = fixture(root, source)
            counter = root / "counter.txt"
            counter.write_text("0")
            script = root / "edit.py"
            script.write_text("def edit(model):\n    from pathlib import Path\n    p = Path(" + repr(str(counter))
                              + ")\n    n = int(p.read_text()) + 1\n    p.write_text(str(n))\n"
                                "    model.weights['transformer.ln_f.weight'].mul_(n + 1)\n")
            config = read(config_path)
            config["repetitions"] = 2
            config["tasks"][0].update(scripted_stochastic=True)
            config["tasks"][0]["script"]["sha256"] = digest(script)
            config_path.write_text(json.dumps(config))
            output = root / "run"
            run(config_path, output, local_trusted=True)
            result = report(output)
            self.assertTrue(result["complete"], result["counts"])
            scripts = [r for r in result["rows"] if r["arm"] == "scripted"]
            agents = [r for r in result["rows"] if r["arm"] == "agent"]
            self.assertEqual(counter.read_text(), "2")
            self.assertNotEqual(scripts[0]["nll"], scripts[1]["nll"])
            for script_row, agent in zip(scripts, agents, strict=True):
                self.assertEqual(agent["improvement_scripted"], script_row["nll"] - agent["nll"])

    def test_known_worker_failure_and_timeout_have_distinct_terminal_outcomes(self):
        if _stack() is None:
            self.skipTest("model dependencies unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_tiny_model(root / "source")
            for name, code, expected in (("failure", "raise RuntimeError('synthetic error')", "failed"),
                                         ("timeout", "import time; time.sleep(5)", "timed_out")):
                with self.subTest(name=name):
                    inputs = root / name
                    inputs.mkdir()
                    config_path = fixture(inputs, source)
                    (inputs / "edit.py").write_text("def edit(model):\n    " + code + "\n")
                    config = read(config_path)
                    config["tasks"][0]["script"]["sha256"] = digest(inputs / "edit.py")
                    config["runtime"]["actions"] = {"path": str(inputs / "actions.json"), "sha256": digest(inputs / "actions.json")}
                    config["limits"]["wall_seconds"] = 30
                    config["limits"]["command_seconds"] = 1 if name == "timeout" else 30
                    config_path.write_text(json.dumps(config))
                    output = root / (name + "-run")
                    run(config_path, output, local_trusted=True)
                    result = report(output)
                    self.assertEqual(result["rows"][1]["status"], expected)
                    self.assertIsNone(result["rows"][1]["nll"])
