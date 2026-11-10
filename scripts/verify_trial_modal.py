"""Verify Modal trial mechanics with scripted Python, without choosing a training task."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages/python/src"))

from wsm.modal_trial import INITIAL_CHECKPOINT, ModalTrialRunner, trial_image
from wsm.trial import Trial
from wsm.trial_poc import Budget


def verify(source, output):
    import modal

    output.mkdir(mode=0o700)
    # Conservative allowances for this fixed, sequential, two-sandbox check.
    budget = Budget({"max_usd": 2, "modal_hourly_usd": 5, "modal_build_usd": 0.5,
                     "modal_egress_per_gib": 0.04, "tinker_prefill_per_million": 1,
                     "tinker_sample_per_million": 1, "tinker_train_per_million": 1,
                     "tinker_storage_usd": 0.01}, output / "budget.jsonl")
    cases = json.loads((ROOT / "examples/coding/cases.json").read_text())
    app = modal.App("wsm-trial-mechanics-proof", include_source=False)
    report = {"scope": "Scripted Modal mechanics only. No hosted agent or training update.", "passed": False}
    runners = []
    try:
        with app.run():
            budget.reserve("modal_build", {})
            image = trial_image(source, app)

            def fresh(name):
                budget.reserve("modal_sandbox", {"seconds": 300})
                sandbox = modal.Sandbox.create("sleep", "infinity", app=app, image=image,
                                               gpu="T4", cpu=(2, 2), memory=(8192, 8192),
                                               timeout=300, block_network=True, workdir="/trial",
                                               include_oidc_identity_token=False)
                runner = ModalTrialRunner(sandbox, command_seconds=90, artifacts=output / (name + "-weights"),
                                          reserve=budget.reserve)
                runners.append(runner)
                return Trial.create(output / name, runner=runner, initial_checkpoint=INITIAL_CHECKPOINT,
                                    cases=cases, max_actions=5)

            trial = fresh("first")
            baseline = trial.evaluate()
            bad = trial.execute("def edit(model):\n    model.weights['model.norm.weight'].zero_()\n",
                                message="Scripted negative control")
            assert bad["receipt"]["candidate"]["loss"] > baseline["loss"] + 1
            persisted = trial.evaluate()
            assert persisted["loss"] == bad["receipt"]["candidate"]["loss"]
            trial.runner.set_history(trial.history())
            undo = trial.execute(
                "import json\nfrom pathlib import Path\nfrom wsm import WeightModel\n"
                "def edit(model):\n"
                "    history = json.loads(Path('/trial/history.json').read_text())\n"
                "    print('history_events', len(history))\n"
                f"    original = WeightModel.load({INITIAL_CHECKPOINT!r}, device='cuda')\n"
                "    model.weights['model.norm.weight'].copy_(original.weights['model.norm.weight'])\n",
                message="Scripted agent-directed undo")
            assert "history_events" in undo["python_stdout"]
            assert abs(undo["receipt"]["candidate"]["loss"] - baseline["loss"]) < 1e-6
            assert undo["changes"]["model.norm.weight"]["after"] == bad["changes"]["model.norm.weight"]["before"]
            frozen = trial.finish()
            trial.runner.export_weights(frozen["checkpoint"], output / "final.safetensors")
            trial.runner.close()
            reset = fresh("second")
            reset_evaluation = reset.evaluate()
            assert reset_evaluation["loss"] == baseline["loss"]
            reset.finish()
            reset.runner.close()
            report.update(passed=True, baseline=baseline, worse=persisted,
                          undo=undo["receipt"]["candidate"], fresh_reset=reset_evaluation,
                          retained_checkpoints=[bad["artifact"], undo["artifact"]],
                          history_read=True, sandbox_termination_confirmed=True,
                          reserved_usd=budget.reserved)
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)[:2000]}
        raise
    finally:
        for runner in runners:
            runner.close()
        (output / "proof.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.model_source, args.output.resolve()), indent=2))
