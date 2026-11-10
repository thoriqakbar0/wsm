"""Trial persistence and inspection through real file and subprocess boundaries."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

from wsm import WeightModel
from wsm.errors import WeightlessError
from wsm.trial import LocalTrialRunner, Trial
from test_editing import _stack, write_tiny_model


class TrialRecordTests(unittest.TestCase):
    def test_private_grade_is_not_in_agent_history_and_freeze_blocks_actions(self):
        with tempfile.TemporaryDirectory() as temporary:
            trial = Trial.create(Path(temporary) / "trial", runner=None,
                                 initial_checkpoint="/fixed-source", cases=[{"prompt": "a", "target": "b"}])
            trial.record_message("assistant", "I will inspect the weights", tokens=[1, 2])
            frozen = trial.finish()
            trial.record_grade({"reward": 0.123, "private_case": "held-out-secret"})
            self.assertEqual(frozen["checkpoint"], "/fixed-source")
            self.assertEqual(trial.finish(), frozen)
            self.assertNotIn("held-out-secret", json.dumps(trial.observation()))
            self.assertEqual(json.loads((trial.directory / "grade.json").read_text())["result"]["reward"], 0.123)
            for action in (lambda: trial.execute("def edit(model): pass"), trial.evaluate,
                           lambda: trial.record_message("assistant", "more")):
                with self.assertRaises(WeightlessError):
                    action()

    def test_unfinished_intent_is_not_retried_after_reopening(self):
        with tempfile.TemporaryDirectory() as temporary:
            trial = Trial.create(Path(temporary) / "trial", runner=None,
                                 initial_checkpoint="/fixed-source", cases=[{"prompt": "a", "target": "b"}])
            with (trial.directory / "events.jsonl").open("a") as stream:
                stream.write(json.dumps({"sequence": 1, "time": 0, "kind": "action",
                                         "data": {"action": 1, "operation": "execute"}}) + "\n")
            reopened = Trial(trial.directory, None)
            self.assertEqual(reopened.observation()["state"], "outcome_unknown")
            with self.assertRaises(WeightlessError):
                reopened.execute("def edit(model): pass")
            with self.assertRaises(WeightlessError):
                reopened.finish()

    def test_overlapping_controller_cannot_write_or_grade_before_freeze(self):
        with tempfile.TemporaryDirectory() as temporary:
            trial = Trial.create(Path(temporary) / "trial", runner=None,
                                 initial_checkpoint="/fixed-source", cases=[{"prompt": "a", "target": "b"}])
            with self.assertRaises(WeightlessError):
                trial.record_grade({"reward": 1})
            with trial._lock():
                with self.assertRaises(WeightlessError) as raised:
                    Trial(trial.directory, None).record_message("assistant", "overlap")
            self.assertEqual(raised.exception.code, "trial_busy")
            self.assertEqual(len(trial.history()), 1)


class TrialModelTests(unittest.TestCase):
    def setUp(self):
        if _stack() is None:
            self.skipTest("torch, transformers, and tokenizers unavailable")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = write_tiny_model(self.root / "source")
        self.cases = [{"prompt": "a b", "target": "c"}]
        self.runner = LocalTrialRunner(self.root / "worker", python=sys.executable)
        self.trial = Trial.create(self.root / "trial", runner=self.runner,
                                  initial_checkpoint=str(self.source), cases=self.cases, max_actions=3)

    def test_worse_edit_persists_explicit_undo_restores_and_next_trial_starts_fresh(self):
        baseline = self.trial.evaluate()
        source = """def edit(model):
    import os
    import torch
    print('python output')
    os.write(1, b'file descriptor output\\n')
    model.weights['transformer.ln_f.weight'].zero_()
    model.weights['transformer.ln_f.bias'].fill_(1)
    model.weights['lm_head.weight'].zero_()
    model.weights['lm_head.weight'][0].fill_(100)
    print(model.evaluate([{'prompt': 'a b', 'target': 'c'}]))
"""
        result = self.trial.execute(source, message="Try an intentionally destructive edit")
        edited = WeightModel.load(result["checkpoint"])
        self.assertTrue(bool((edited.weights["transformer.ln_f.weight"] == 0).all()))
        self.assertEqual(self.trial.observation()["checkpoint"], result["checkpoint"])
        self.assertIn("python output", result["python_stdout"])
        self.assertIn("file descriptor output", result["stdout"])
        self.assertGreaterEqual(len(result["evaluations"]), 3)
        self.assertIn("transformer.ln_f.weight", result["changes"])
        self.assertGreater(result["receipt"]["candidate"]["loss"], baseline["loss"] + 100)
        undo = "def edit(model):\n    from wsm import WeightModel\n    import torch\n    original = WeightModel.load(" + repr(str(self.source)) + ")\n    for name in model.weights:\n        model.weights[name].copy_(original.weights[name])\n"
        restored = self.trial.execute(undo, message="Undo the experiment")
        self.assertAlmostEqual(restored["receipt"]["candidate"]["loss"], baseline["loss"], places=6)
        self.assertEqual((self.trial.directory / "actions/0002/edit.py").read_text(), source)
        self.assertIn("Undo the experiment", json.dumps(self.trial.history()))
        with self.assertRaises(WeightlessError) as raised:
            self.trial.evaluate()
        self.assertEqual(raised.exception.code, "trial_budget")
        self.assertEqual(self.trial.finish()["checkpoint"], restored["checkpoint"])
        fresh = Trial.create(self.root / "next", runner=self.runner,
                             initial_checkpoint=str(self.source), cases=self.cases)
        self.assertAlmostEqual(fresh.evaluate()["loss"], baseline["loss"], places=6)

    def test_exception_ends_trial_without_silently_retrying(self):
        source = "def edit(model):\n    print('output before failure')\n    model.weights['transformer.ln_f.weight'].zero_()\n    raise RuntimeError('deliberate failure')\n"
        with self.assertRaises(RuntimeError):
            self.trial.execute(source)
        view = self.trial.observation()
        self.assertEqual(view["state"], "failed")
        self.assertEqual(view["checkpoint"], str(self.source))
        self.assertEqual(view["actions"], 1)
        self.assertIn("output before failure", view["history"][-1]["data"]["observation"]["worker"]["stdout.txt"])
        with self.assertRaises(WeightlessError):
            self.trial.execute(source)


if __name__ == "__main__":
    unittest.main()
