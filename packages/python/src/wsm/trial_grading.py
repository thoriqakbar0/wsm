"""Independent final grading with trusted architecture and untrusted tensor data."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

from .binary64 import WeightlessError


def grade_weights(source, weights, cases, *, scoring, device="cpu"):
    """Load no agent code or configuration. Use the original model's file schema."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    from .editing import WeightModel, _cases

    if scoring not in ("negative_loss", "loss_improvement"):
        raise WeightlessError("invalid_scoring", "Select negative_loss or loss_improvement explicitly")
    cases = _cases(cases)
    if not cases:
        raise WeightlessError("invalid_cases", "Final grading requires nonempty held-out cases")
    source, weights = Path(source), Path(weights)
    trusted_weights = source / "model.safetensors"
    with safe_open(trusted_weights, framework="pt", device="cpu") as original:
        with safe_open(weights, framework="pt", device="cpu") as candidate:
            if set(original.keys()) != set(candidate.keys()):
                raise WeightlessError("invalid_checkpoint", "Exported weights differ from the trusted tensor names")
            for name in original.keys():
                left, right = original.get_slice(name), candidate.get_slice(name)
                allowed_dtypes = {left.get_dtype()}
                if left.get_dtype() in {"F64", "F32", "F16", "BF16"}:
                    allowed_dtypes.add("F32")
                if (left.get_shape() != right.get_shape()
                        or right.get_dtype() not in allowed_dtypes):
                    raise WeightlessError("invalid_checkpoint", "Exported tensor layout differs from the trusted model")
    model = WeightModel.load(source, device=device)
    baseline = model.evaluate(cases)
    tensors = load_file(weights, device="cpu")
    if any(not bool(torch.isfinite(tensor).all()) for tensor in tensors.values()):
        raise WeightlessError("nonfinite", "Final checkpoint contains nonfinite weights")
    initial = model.model.state_dict()
    changed = [name for name, tensor in tensors.items()
               if not torch.equal(initial[name].detach().cpu(), tensor.to(dtype=initial[name].dtype))]
    # HF safetensors omits tied aliases. Exact source-file keys were checked above.
    model.model.load_state_dict(tensors, strict=False)
    final = model.evaluate(cases)
    reward = -final["loss"] if scoring == "negative_loss" else baseline["loss"] - final["loss"]
    if not math.isfinite(reward):
        raise WeightlessError("invalid_reward", "Final grading did not produce a finite reward")
    with weights.open("rb") as handle:
        identity = "sha256:" + hashlib.file_digest(handle, "sha256").hexdigest()
    return {"format": "wsm.trial-grade.v1", "checkpoint": identity, "scoring": scoring,
            "reward": reward, "baseline": baseline, "final": final, "changed_tensors": changed}


def grade_in_process(source, weights, private_cases, *, scoring, python, timeout):
    """The caller must terminate the agent sandbox before invoking this grader."""
    request = {"source": str(Path(source).resolve()), "weights": str(Path(weights).resolve()),
               "cases": json.loads(Path(private_cases).read_text()), "scoring": scoring}
    package = str(Path(__file__).resolve().parents[1])
    import os
    env = {key: value for key, value in os.environ.items()
           if key in ("PATH", "LANG", "LC_ALL", "SYSTEMROOT")}
    env["PYTHONPATH"] = package
    completed = subprocess.run([str(python), "-m", "wsm.trial_grading"],
                               input=json.dumps(request, allow_nan=False), capture_output=True,
                               text=True, env=env, cwd="/", timeout=timeout)
    if completed.returncode:
        raise WeightlessError("final_grading_failed", completed.stderr[-4000:])
    return json.loads(completed.stdout)


def main():
    import sys
    argparse.ArgumentParser(description=__doc__).parse_args()
    request = json.load(sys.stdin)
    print(json.dumps(grade_weights(**request), allow_nan=False))


if __name__ == "__main__":
    main()
