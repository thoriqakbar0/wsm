import copy
import hashlib
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from weightless import (MetricPolicy, NumericalEvaluation, SearchRun, WeightlessError,
                        evaluation_result, evaluator_config, replay)
from weightless.binary64 import create_checkpoint, evaluate, parse_checkpoint, predict
from weightless.core import _sealed

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = json.loads((ROOT / "fixtures/addition.json").read_text())
POLICY = MetricPolicy(("development.loss", "minimize"), ("train.loss", "minimize"))


def initial():
    return create_checkpoint(FIXTURE["graph"], FIXTURE["tensors"])


def new_run(evaluator=None, budget=8, checkpoint=None, policy=POLICY):
    return SearchRun(checkpoint or initial(), evaluator=evaluator or NumericalEvaluation(FIXTURE["task"]),
                     policy=policy, budget=budget)


def proposal(run, edits=None):
    return {"base": run.inspect()["checkpoint"]["id"], "author": "scripted-teacher",
            "edits": edits if edits is not None else FIXTURE["attempts"][1]["edits"]}


def edit(value, tensor="output.weight", index=0):
    return {"tensor": tensor, "index": index, "value": value}


def reseal(value):
    return _sealed({k: v for k, v in value.items() if k != "id"})


class RecordingEvaluation:
    def __init__(self, response=None):
        self.calls = []
        self.config = evaluator_config("test", "development", {"opaque": ["no numerical examples"]})
        self.response = response

    def __call__(self, request, checkpoint):
        self.calls.append(copy.deepcopy((request, checkpoint)))
        if self.response:
            return self.response(request, checkpoint)
        return evaluation_result(request, status="completed", metrics={"quality": -request["attempt"]},
                                 reward={"reward": 0.0}, native_status="graded")


class ContractTests(unittest.TestCase):
    def assert_code(self, code, function, *args):
        with self.assertRaises(WeightlessError) as result:
            function(*args)
        self.assertEqual(result.exception.code, code)

    def test_signed_addition_golden_and_independent_sum(self):
        golden = json.loads((ROOT / "fixtures/golden.json").read_text())
        self.assertEqual(initial()["id"], golden["id"])
        self.assertEqual("sha256:" + hashlib.sha256(bytes.fromhex(golden["identityHex"])).hexdigest(), golden["id"])
        run = new_run()
        for attempt in FIXTURE["attempts"]:
            decision = run.submit(proposal(run, attempt["edits"]))
            self.assertEqual((decision["accepted"], decision["reason"]), (attempt["accepted"], attempt["reason"]))
        final = replay(run.finish())
        for x in range(-12, 13):
            for y in range(-12, 13):
                self.assertEqual(predict(final, [x, y]), [x + y])
        self.assertEqual(evaluate(final, FIXTURE["task"]["train"])["loss"], 0)

    def test_row_major_arithmetic_and_independent_metrics(self):
        graph = {"inputSize": 2, "nodes": [{"id": "out", "op": "affine", "input": "input", "weight": "w", "bias": "b"}], "output": "out"}
        checkpoint = create_checkpoint(graph, {"w": {"shape": [2, 2], "values": [2, 3, -1, 4]}, "b": {"shape": [2], "values": [1, -2]}})
        self.assertEqual(predict(checkpoint, [2, -1]), [2, -8])
        self.assertEqual(evaluate(checkpoint, [{"input": [2, -1], "target": [1, -4]}]),
                         {"loss": 0.5, "rmse": math.sqrt(8.5), "exact": 0, "count": 1})
        graph["inputSize"] = 3
        ordered = create_checkpoint(graph, {"w": {"shape": [1, 3], "values": [1e16, -1e16, 1]}, "b": {"shape": [1], "values": [0]}})
        self.assertEqual(predict(ordered, [1, 1, 1]), [1])

    def test_direct_numerical_evaluation_checks_actual_request_binding(self):
        evaluator = NumericalEvaluation(FIXTURE["task"])
        checkpoint = initial()
        run = new_run(evaluator)
        request = {"format": "weightless.request.v1", "run": run.inspect()["run"],
                   "attempt": 0, "checkpoint": checkpoint["id"],
                   "config": evaluator.config["id"], "purpose": "development"}
        self.assertEqual(evaluator(request, checkpoint)["status"], "completed")
        other = NumericalEvaluation(dict(FIXTURE["task"], name="other-problem"))
        changed = copy.deepcopy(checkpoint)
        changed["tensors"]["output.bias"]["values"][0] = 1
        changed = create_checkpoint(changed["graph"], changed["tensors"])
        for field, value in (("checkpoint", changed["id"]),
                             ("config", other.config["id"]), ("purpose", "final")):
            with self.subTest(field=field):
                self.assert_code("invalid_result", evaluator, dict(request, **{field: value}), checkpoint)

    def test_nested_config_and_evidence_fit_frozen_envelope(self):
        nested = None
        for _ in range(30):
            nested = {"nested": nested}
        evaluator = RecordingEvaluation()
        evaluator.config = evaluator_config("deep", "development", {"opaque": nested})
        with self.subTest(boundary="configuration"):
            self.assert_code("invalid_config", new_run, evaluator, 1,
                             None, MetricPolicy(("quality", "minimize")))
        nested = None
        for _ in range(27):
            nested = {"nested": nested}
        def response(request, checkpoint):
            return evaluation_result(request, status="completed", metrics={"quality": -request["attempt"]},
                                     reward={"reward": 0.75}, diagnostics={"opaque": nested})
        run = new_run(RecordingEvaluation(response), policy=MetricPolicy(("quality", "minimize")))
        decision = run.submit(proposal(run))
        with self.subTest(boundary="evaluation"):
            result = run.inspect()["events"][0]["evaluation"]
            self.assertEqual(result["status"], "invalid_response")
            self.assertEqual(result["native"]["reward"], {"reward": 0.75})
            self.assertFalse(decision["accepted"])
            self.assertEqual(replay(run.finish()), initial())

    def test_atomic_validation_stale_proposals_and_budget(self):
        evaluator = RecordingEvaluation()
        run = new_run(evaluator, budget=1, policy=MetricPolicy(("quality", "minimize")))
        before = run.inspect()
        bad = [[edit(1), edit(2)], [edit(True)], [edit(1, index=True)], [edit(1, index=-1)],
               [edit(math.inf)], [edit(1, tensor="missing")], [], [dict(edit(1), extra=0)]]
        for edits in bad:
            self.assert_code("invalid_proposal", run.submit, proposal(run, edits))
            self.assertEqual(run.inspect(), before)
            self.assertFalse(evaluator.calls)
        stale = proposal(run)
        run.submit(proposal(run))
        self.assertEqual(len(evaluator.calls), 2)
        self.assert_code("budget_exhausted", run.submit, proposal(run))
        other = new_run()
        old = proposal(other)
        other.submit(proposal(other))
        self.assert_code("stale_base", other.propose, old)
        self.assertEqual(other.inspect()["used"], 1)

    def test_checkpoint_shapes_imports_limits_and_duplicates(self):
        for mutate in (lambda c: c["tensors"]["output.weight"].update(shape=[2, 1]),
                       lambda c: c["graph"].update(inputSize=True),
                       lambda c: c["graph"]["nodes"][0].update(input="output"),
                       lambda c: c["graph"]["nodes"][0].update(id="input"),
                       lambda c: c["graph"]["nodes"][0].update(op="sigmoid"),
                       lambda c: c["tensors"].update(unused={"shape": [1], "values": [0]}),
                       lambda c: c["graph"].update(extra=1),
                       lambda c: c["tensors"]["output.bias"].update(values=[float("nan")]),
                       lambda c: c["graph"].update(inputSize=1025),
                       lambda c: c["graph"].update(nodes=c["graph"]["nodes"] * 44),
                       lambda c: c["tensors"]["hidden.weight"].update(shape=[100, 1001], values=[0] * 100100)):
            value = initial()
            mutate(value)
            self.assert_code("invalid_checkpoint", parse_checkpoint, value)
        self.assert_code("invalid_checkpoint", parse_checkpoint, '{"format":0,"format":1}')
        value = initial()
        value["graph"]["inputSize"] = 2.0
        self.assertEqual(parse_checkpoint(value)["id"], initial()["id"])

    def test_copies_candidate_ownership_and_closure(self):
        task, checkpoint = copy.deepcopy(FIXTURE["task"]), initial()
        evaluator = NumericalEvaluation(task)
        run = new_run(evaluator, checkpoint=checkpoint)
        other = new_run()
        task["train"][0]["target"][0] = 999
        evaluator.config["settings"].clear()
        checkpoint["tensors"].clear()
        observation = run.inspect()
        observation["config"].clear()
        observation["checkpoint"]["tensors"].clear()
        candidate = run.propose(proposal(run))
        self.assert_code("foreign_candidate", other.evaluate, candidate)
        for field, replacement in (("attempt", True), ("run", other.inspect()["run"])):
            changed = copy.deepcopy(candidate)
            changed[field] = replacement
            self.assert_code("foreign_candidate", run.evaluate, changed)
        changed = copy.deepcopy(candidate)
        changed["checkpoint"]["tensors"]["output.weight"]["values"][0] = 4
        self.assert_code("foreign_candidate", run.evaluate, changed)
        result = run.evaluate(candidate)
        changed = copy.deepcopy(result)
        changed["request"]["attempt"] = True
        self.assert_code("invalid_result", run.select, candidate, changed)
        self.assertTrue(run.select(candidate, result)["accepted"])
        self.assert_code("foreign_candidate", run.evaluate, run.inspect()["incumbent"])
        frozen = run.finish()
        self.assertEqual(replay(frozen), frozen["checkpoint"])
        frozen["events"].clear()
        self.assertEqual(len(run.finish()["events"]), 1)
        self.assert_code("closed", run.propose, proposal(run))
        self.assert_code("closed", run.evaluate, candidate)
        self.assert_code("closed", run.select, candidate, result)
        self.assertEqual(other.inspect()["checkpoint"], initial())

    def test_baseline_is_lazy_once_and_failed_evidence_freezes(self):
        def broken(request, checkpoint):
            raise OSError("docker unavailable")
        evaluator = RecordingEvaluation(broken)
        run = new_run(evaluator, policy=MetricPolicy(("quality", "minimize")))
        self.assertFalse(evaluator.calls)
        baseline = run.evaluate(run.inspect()["incumbent"])
        self.assertEqual(baseline["status"], "infrastructure_error")
        self.assert_code("invalid_state", run.evaluate, run.inspect()["incumbent"])
        decision = run.submit(proposal(run))
        self.assertEqual(decision["reason"], "baseline_failed")
        self.assertEqual(len(evaluator.calls), 2)
        frozen = run.finish()
        self.assertEqual(frozen["baseline"]["error"]["message"], "docker unavailable")
        self.assertEqual(replay(frozen), initial())

    def test_completed_zero_native_reward_is_usable(self):
        evaluator = RecordingEvaluation()
        run = new_run(evaluator, policy=MetricPolicy(("quality", "minimize")))
        self.assertTrue(run.submit(proposal(run))["accepted"])
        event = run.finish()["events"][0]
        self.assertEqual(event["evaluation"]["status"], "completed")
        self.assertEqual(event["evaluation"]["native"]["reward"], {"reward": 0.0})
        self.assertEqual(len(evaluator.calls), 2)

    def test_result_bindings_config_drift_and_malformed_metrics(self):
        def corrupt(field, replacement):
            def response(request, checkpoint):
                if request["attempt"]:
                    request[field] = replacement
                return evaluation_result(request, status="completed", metrics={"quality": 0})
            return response
        for field, replacement in (("run", "sha256:" + "0" * 64), ("attempt", 0),
                                   ("checkpoint", "sha256:" + "0" * 64),
                                   ("config", "sha256:" + "0" * 64), ("purpose", "final")):
            run = new_run(RecordingEvaluation(corrupt(field, replacement)), policy=MetricPolicy(("quality", "minimize")))
            self.assertFalse(run.submit(proposal(run))["accepted"])
            self.assertEqual(run.finish()["events"][0]["evaluation"]["status"], "invalid_response")
        for metric in (True, None, "0", float("nan")):
            def bad(request, checkpoint):
                if request["attempt"]:
                    result = evaluation_result(request, status="completed", metrics={"quality": 0})
                    result["native"]["metrics"]["quality"] = metric
                    return reseal(result)
                return evaluation_result(request, status="completed", metrics={"quality": 1})
            run = new_run(RecordingEvaluation(bad), policy=MetricPolicy(("quality", "minimize")))
            self.assertEqual(run.submit(proposal(run))["reason"], "evaluation_failed")
        evaluator = RecordingEvaluation()
        run = new_run(evaluator, policy=MetricPolicy(("quality", "minimize")))
        evaluator.config = evaluator_config("test", "development", {"changed": True})
        self.assertEqual(run.submit(proposal(run))["reason"], "baseline_failed")
        self.assertFalse(evaluator.calls)

    def test_pending_stages_freeze_and_no_hidden_retries(self):
        run = new_run()
        candidate = run.propose(proposal(run))
        self.assert_code("pending", run.propose, proposal(run))
        result = run.evaluate(candidate)
        self.assert_code("invalid_state", run.evaluate, candidate)
        frozen = run.finish()
        self.assertIsNone(frozen["events"][0]["decision"])
        self.assertEqual(replay(frozen), initial())
        untouched = new_run()
        untouched.propose(proposal(untouched))
        self.assertEqual(replay(untouched.finish()), initial())
        self.assertEqual(replay(new_run().finish()), initial())

    def test_numerical_failure_cannot_win(self):
        run = new_run(budget=1)
        decision = run.submit(proposal(run, [edit(1e308)]))
        self.assertEqual(decision["reason"], "evaluation_failed")
        result = run.finish()["events"][0]["evaluation"]
        self.assertEqual(result["status"], "task_failure")
        self.assertEqual(result["error"]["type"], "numerical_failure")
        self.assertEqual(replay(run.finish()), initial())

    def test_training_tiebreak_and_development_priority(self):
        graph = {"inputSize": 1, "nodes": [{"id": "o", "op": "affine", "input": "input", "weight": "w", "bias": "b"}], "output": "o"}
        checkpoint = create_checkpoint(graph, {"w": {"shape": [1, 1], "values": [0]}, "b": {"shape": [1], "values": [0]}})
        task = {"format": "weightless.task.v1", "name": "tie", "train": [{"input": [1], "target": [1]}],
                "development": [{"input": [0], "target": [0]}], "budget": 3}
        run = new_run(NumericalEvaluation(task), checkpoint=checkpoint)
        self.assertTrue(run.submit(proposal(run, [edit(1, "w")]))["accepted"])
        self.assertFalse(run.submit(proposal(run, [edit(2, "w")]))["accepted"])
        conflict = dict(task, development=[{"input": [0], "target": [-1]}])
        run = new_run(NumericalEvaluation(conflict), checkpoint=checkpoint)
        self.assertFalse(run.submit(proposal(run, [edit(1, "b")]))["accepted"])
        self.assert_code("invalid_task", NumericalEvaluation, dict(task, heldout=[]))
        self.assert_code("invalid_task", NumericalEvaluation, dict(task, train=[]))
        self.assert_code("invalid_task", NumericalEvaluation, dict(task, budget=True))

    def test_replay_checks_history_without_evaluating(self):
        run = new_run()
        run.submit(proposal(run))
        frozen = run.finish()
        with patch.object(NumericalEvaluation, "__call__", side_effect=AssertionError("must not evaluate")):
            self.assertEqual(replay(json.dumps(frozen)), frozen["checkpoint"])
        mutations = [lambda a: a["events"][0]["decision"].update(accepted=False),
                     lambda a: a["events"][0]["decision"].update(before=False),
                     lambda a: a["events"][0]["evaluation"]["native"]["metrics"]["train"].update(count=True),
                     lambda a: a["events"][0]["evaluation"]["request"].update(attempt=0),
                     lambda a: a["events"][0]["proposal"]["edits"][0].update(value=2),
                     lambda a: a["provenance"].update(version="2"),
                     lambda a: a["policy"]["metrics"][0].update(direction="maximize"),
                     lambda a: a.update(used=True),
                     lambda a: a.update(extra=0),
                     lambda a: a["checkpoint"]["tensors"]["output.bias"].update(values=[1])]
        for mutate in mutations:
            artifact = copy.deepcopy(frozen)
            mutate(artifact)
            self.assert_code("invalid_artifact", replay, reseal(artifact))
        reordered = new_run()
        for attempt in FIXTURE["attempts"]:
            reordered.submit(proposal(reordered, attempt["edits"]))
        artifact = reordered.finish()
        artifact["events"].reverse()
        self.assert_code("invalid_artifact", replay, reseal(artifact))

    def test_reentrant_evaluator_cannot_mutate_run(self):
        def response(request, checkpoint):
            self.assert_code("invalid_state", run.finish)
            self.assert_code("invalid_state", run.propose, proposal(run))
            checkpoint["tensors"].clear()
            return evaluation_result(request, status="completed", metrics={"quality": 1})
        evaluator = RecordingEvaluation(response)
        run = new_run(evaluator, policy=MetricPolicy(("quality", "minimize")))
        run.submit(proposal(run))
        self.assertEqual(len(evaluator.calls), 2)
        self.assertEqual(replay(run.finish()), initial())

    def test_history_space_is_reserved_before_dispatch(self):
        checkpoint = create_checkpoint({"inputSize": 1000, "nodes": [
            {"id": "o", "op": "affine", "input": "input", "weight": "w", "bias": "b"}], "output": "o"},
            {"w": {"shape": [1, 1000], "values": [0] * 1000}, "b": {"shape": [1], "values": [0]}})
        evaluator = RecordingEvaluation()
        with patch("weightless.core.MAX_BYTES", 550000):
            run = new_run(evaluator, checkpoint=checkpoint, policy=MetricPolicy(("quality", "minimize")))
            edits = [edit(1, "w", index) for index in range(1000)]
            self.assert_code("invalid_proposal", run.propose, proposal(run, edits))
            self.assertEqual(run.inspect()["used"], 0)
            self.assertFalse(evaluator.calls)
            self.assertEqual(replay(run.finish()), checkpoint)

    def test_oversized_result_is_retained_as_invalid_response(self):
        def response(request, checkpoint):
            result = evaluation_result(request, status="completed", metrics={"quality": -request["attempt"]})
            if request["attempt"]:
                result["native"]["diagnostics"]["output"] = "x" * 300000
            return reseal(result)
        run = new_run(RecordingEvaluation(response), policy=MetricPolicy(("quality", "minimize")))
        self.assertEqual(run.submit(proposal(run))["reason"], "evaluation_failed")
        self.assertEqual(run.finish()["events"][0]["evaluation"]["status"], "invalid_response")
        self.assertEqual(replay(run.finish()), initial())


if __name__ == "__main__":
    unittest.main()
