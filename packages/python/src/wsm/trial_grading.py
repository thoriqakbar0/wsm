"""Independent final grading with trusted architecture and untrusted tensor data."""

import argparse
import contextlib
import hashlib
import json
import math
from pathlib import Path
import subprocess

from .errors import WeightlessError


def grade_weights(source, weights, cases, *, scoring, device="cpu", detailed=False):
    """Load no agent code or configuration. Use the original model's file schema."""
    import torch
    from safetensors import SafetensorError, safe_open
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
        try:
            candidate_file = safe_open(weights, framework="pt", device="cpu")
        except SafetensorError as error:
            raise WeightlessError("invalid_checkpoint", "Exported tensor file is malformed") from error
        with candidate_file as candidate:
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
    try:
        tensors = load_file(weights, device="cpu")
    except SafetensorError as error:
        raise WeightlessError("invalid_checkpoint", "Exported tensor file is malformed") from error
    if any(not bool(torch.isfinite(tensor).all()) for tensor in tensors.values()):
        raise WeightlessError("nonfinite", "Final checkpoint contains nonfinite weights")
    initial = model.model.state_dict()
    changed = [name for name, tensor in tensors.items()
               if not torch.equal(initial[name].detach().cpu(), tensor.to(dtype=initial[name].dtype))]
    # HF safetensors omits tied aliases. Exact source-file keys were checked above.
    model.model.load_state_dict(tensors, strict=False)
    final = model.evaluate(cases)
    if detailed:
        for evaluation in (baseline, final):
            for case, row in zip(cases, evaluation["cases"], strict=True):
                row["target_tokens"] = len(model.tokenizer.encode(case["target"], add_special_tokens=False))
    reward = -final["loss"] if scoring == "negative_loss" else baseline["loss"] - final["loss"]
    if not math.isfinite(reward):
        raise WeightlessError("invalid_reward", "Final grading did not produce a finite reward")
    with weights.open("rb") as handle:
        identity = "sha256:" + hashlib.file_digest(handle, "sha256").hexdigest()
    return {"format": "wsm.trial-grade.v1", "checkpoint": identity, "scoring": scoring,
            "reward": reward, "baseline": baseline, "final": final, "changed_tensors": changed}


def grader_environment():
    package = str(Path(__file__).resolve().parents[1])
    import os
    env = {key: value for key, value in os.environ.items()
           if key in ("PATH", "LANG", "LC_ALL", "SYSTEMROOT", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    env["PYTHONPATH"] = package
    return env


def grade_in_process(source, weights, private_cases, *, scoring, python, timeout, detailed=False):
    """The caller must terminate the agent sandbox before invoking this grader."""
    request = {"source": str(Path(source).resolve()), "weights": str(Path(weights).resolve()),
               "cases": json.loads(Path(private_cases).read_text()), "scoring": scoring, "detailed": detailed}
    completed = subprocess.run([str(python), "-m", "wsm.trial_grading"],
                               input=json.dumps(request, allow_nan=False), capture_output=True,
                               text=True, env=grader_environment(), cwd="/", timeout=timeout)
    if completed.returncode:
        if detailed:
            try:
                code = json.loads(completed.stdout)["error"]["code"]
            except (ValueError, KeyError, TypeError):
                code = "final_grading_failed"
            if code not in {"invalid_checkpoint", "nonfinite", "invalid_cases"}:
                code = "final_grading_failed"
            error = WeightlessError(code, "Independent grading failed; inspect restricted evidence")
            error.diagnostics = completed.stderr[-4000:]
            raise error
        raise WeightlessError("final_grading_failed", completed.stderr[-4000:])
    return json.loads(completed.stdout)


def main():
    import sys
    argparse.ArgumentParser(description=__doc__).parse_args()
    request = json.load(sys.stdin)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = grade_weights(**request)
        print(json.dumps(result, allow_nan=False))
    except Exception as error:
        if not request.get("detailed"):
            raise
        print(str(error), file=sys.stderr)
        print(json.dumps({"error": {"code": getattr(error, "code", "final_grading_failed")}}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
