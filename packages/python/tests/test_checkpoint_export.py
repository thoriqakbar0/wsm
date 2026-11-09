"""Selected checkpoints reload from the exported bytes, not from the original source."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from wsm import binary64
from wsm.checkpoint_export import export_checkpoint
from wsm.session import Session
from wsm.target_worker import run


class ControlledSelectionRunner:
    """Real numerical execution with forced selection to isolate export behavior."""
    config = {"mode": "in-process", "image": "local"}

    def __init__(self):
        self.initial = None

    def execute(self, action, target, **kwargs):
        started = kwargs.pop("on_dispatch", None)
        if started:
            started()
        request = {"action": action, "target": target, "patches": kwargs.get("patches") or [],
                   "edits": kwargs.get("edits") or [], "cases": kwargs.get("cases"),
                   "query": kwargs.get("query"), "generate": bool(kwargs.get("generate"))}
        if kwargs.get("forward_limit") is not None:
            request["forward_limit"] = kwargs["forward_limit"]
        result = run(request)
        if action == "describe" and self.initial is None:
            self.initial = result["checkpoint"]
        if action == "evaluate" and self.initial is not None and result["checkpoint"] != self.initial:
            result["metrics"]["loss"] = -1.0
        return result


def _selected_bundle(directory, target, *, tensor, index, value, cases):
    runner = ControlledSelectionRunner()
    declaration = {"hypothesis": "A scalar replacement is selected and exported.",
                   "arms": ["editor"], "editable": [tensor],
                   "budgets": {"attempts": 2, "evaluations": 6, "inspections": 2,
                               "teacher_requests": 0, "final_evaluations": 0},
                   "max_edits": 2, "max_inspect": 4, "wall_seconds": 120}
    session = Session.create(Path(directory) / "session.db", runner=runner, target=target,
                             declaration=declaration, development=cases)
    try:
        session.inspect({"checkpoint": session.observe("editor")["base"], "tensor": tensor, "indices": [index]},
                        key="read", arm="editor")
        view = session.observe("editor")
        proposal = {key: view[key] for key in ("session", "arm", "declaration", "observation", "base")}
        proposal["edits"] = [{"tensor": tensor, "index": index, "value": value}]
        submitted = session.submit(proposal, key="edit")
        if not submitted["selection"]["accepted"]:
            raise AssertionError("edit was not selected")
        frozen = session.freeze(key="freeze")
        if not frozen["complete"]:
            raise AssertionError("freeze incomplete")
        return session.export(), frozen["arms"]["editor"]["checkpoint"], runner.initial
    finally:
        session.close()


class BinaryExportTests(unittest.TestCase):
    def test_binary_checkpoint_matches_selected_identity(self):
        with open(os.path.join(os.path.dirname(__file__), "../../../fixtures/addition.json")) as handle:
            fixture = json.load(handle)
        checkpoint = binary64.create_checkpoint(fixture["graph"], fixture["tensors"])
        target = {"kind": "binary64", "checkpoint": checkpoint}
        described = run({"target": target, "action": "describe", "patches": [], "edits": [],
                         "cases": None, "query": None, "generate": False})
        name = next(iter(described["parameters"]))
        current = run({"target": target, "action": "inspect", "patches": [], "edits": [],
                       "cases": None, "query": {"tensor": name, "indices": [0]}, "generate": False})["values"][0]["value"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, selected, initial = _selected_bundle(
                root, target, tensor=name, index=0, value=current + 1,
                cases=[{"id": "one", "input": [1, 2], "target": [3]}])
            destination = root / "selected"
            receipt = export_checkpoint(bundle, target=target, arm="editor",
                                        destination=destination, max_bytes=1_000_000)
            reloaded = json.loads((destination / "checkpoint.json").read_text())
        self.assertEqual(receipt["format"], "wsm.checkpoint-export.v1")
        self.assertEqual(reloaded["id"], selected)
        self.assertNotEqual(reloaded["id"], initial)


class ModelExportTests(unittest.TestCase):
    def test_changed_model_reloads_after_the_original_is_gone(self):
        try:
            from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
            from tokenizers import Tokenizer
            from tokenizers.models import WordLevel
            from tokenizers.pre_tokenizers import Whitespace
        except ImportError:
            self.skipTest("torch, transformers, and tokenizers unavailable")

        revision = "a" * 40
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "hub" / "snapshots" / revision
            snapshot.mkdir(parents=True)
            backend = Tokenizer(WordLevel({"<unk>": 0, "a": 1, "b": 2, "c": 3}, unk_token="<unk>"))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", eos_token="a")
            config = GPT2Config(vocab_size=4, n_positions=32, n_embd=16, n_layer=1, n_head=2, n_inner=32,
                                bos_token_id=None, eos_token_id=None, pad_token_id=None)
            GPT2LMHeadModel(config).save_pretrained(snapshot, safe_serialization=True)
            tokenizer.save_pretrained(snapshot)
            from wsm.sandbox import stage_model
            from wsm.target_worker import ModelTarget
            staged = stage_model(root / "hub", root / "staged", model="local/tiny", revision=revision)
            model = ModelTarget(staged)
            name = next(iter(model.parameters))
            value = model.value(name, 0) + 0.25
            case = {"id": "one", "prompt": "hi", "target": "a"}
            bundle, selected, initial = _selected_bundle(root, staged, tensor=name, index=0, value=value, cases=[case])
            destination = root / "selected"
            export_checkpoint(bundle, target=staged, arm="editor", destination=destination, max_bytes=50_000_000)
            reloaded_here = ModelTarget({**staged, "source": str(destination)})
            metrics, rows = reloaded_here.evaluate([case], False, None, 4)
            self.assertEqual(reloaded_here.identity(), selected)
            self.assertNotEqual(selected, initial)
            original_source = staged["source"]
            shutil.rmtree(original_source)
            script = r'''
import json, sys
from pathlib import Path
from wsm.target_worker import ModelTarget
request = json.loads(sys.stdin.read())
assert not Path(request["original"]).exists()
model = ModelTarget(request["target"])
metrics, rows = model.evaluate(request["cases"], False, None, 4)
print(json.dumps({"checkpoint": model.identity(), "metrics": metrics, "rows": rows}))
'''
            child = subprocess.run([sys.executable, "-c", script], input=json.dumps({
                "original": original_source, "target": {**staged, "source": str(destination)}, "cases": [case]}),
                text=True, capture_output=True, timeout=120)
            self.assertEqual(child.returncode, 0, child.stderr)
            reloaded = json.loads(child.stdout)
            self.assertEqual(reloaded["checkpoint"], selected)
            self.assertEqual(reloaded["metrics"], metrics)
            self.assertEqual(reloaded["rows"], rows)


@unittest.skipUnless(Path("/root/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct").is_dir(),
                     "cached SmolLM snapshot is not present")
class SmolLMExportTests(unittest.TestCase):
    def test_cached_smollm_edit_reloads_without_the_original_mount(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch and transformers unavailable")
        from wsm.sandbox import stage_smollm
        from wsm.target_worker import ModelTarget

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged = stage_smollm("/root/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct", root / "staged")
            model = ModelTarget(staged)
            value = model.value("model.norm.weight", 0) + 0.125
            case = {"id": "one", "prompt": "Say hello.", "target": "Hello"}
            bundle, selected, initial = _selected_bundle(
                root, staged, tensor="model.norm.weight", index=0, value=value, cases=[case])
            destination = root / "selected"
            export_checkpoint(bundle, target=staged, arm="editor", destination=destination, max_bytes=2_000_000_000)
            original_source = staged["source"]
            shutil.rmtree(original_source)
            script = r'''
import json, sys
from pathlib import Path
from wsm.target_worker import ModelTarget
request = json.loads(sys.stdin.read())
assert not Path(request["original"]).exists()
model = ModelTarget(request["target"])
assert model.aliases["lm_head.weight"] == "model.embed_tokens.weight"
metrics, rows = model.evaluate(request["cases"], True, {"max_new_tokens": 1}, 4)
print(json.dumps({"checkpoint": model.identity(), "metrics": metrics, "rows": rows,
                  "tied": True, "original_absent": True}))
'''
            parent = ModelTarget({**staged, "source": str(destination)})
            expected_metrics, expected_rows = parent.evaluate([case], True, {"max_new_tokens": 1}, 4)
            child = subprocess.run([sys.executable, "-c", script], input=json.dumps({
                "original": original_source, "target": {**staged, "source": str(destination)}, "cases": [case]}),
                text=True, capture_output=True, timeout=180)
            self.assertEqual(child.returncode, 0, child.stderr)
            reloaded = json.loads(child.stdout)
            self.assertEqual(reloaded["checkpoint"], selected)
            self.assertNotEqual(selected, initial)
            self.assertEqual(reloaded["metrics"], expected_metrics)
            self.assertEqual(reloaded["rows"], expected_rows)


if __name__ == "__main__":
    unittest.main()


class ExportBoundaryTests(unittest.TestCase):
    def test_source_containment_and_size_reject_before_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "source" / "snapshots" / ("a" * 40)
            snapshot.mkdir(parents=True)
            (snapshot / "model.safetensors").write_bytes(b"too large")
            target = {"kind": "huggingface", "source": str(root / "source")}
            for destination, limit in ((snapshot / "nested", 100), (root / "export", 1)):
                with self.subTest(destination=destination), \
                     patch("wsm.checkpoint_export._selected", return_value=("checkpoint", [])), \
                     patch("wsm.checkpoint_export.snapshot_directory", return_value=str(snapshot)), \
                     patch("wsm.checkpoint_export.ModelTarget") as model:
                    with self.assertRaises(binary64.WeightlessError):
                        export_checkpoint({}, target=target, arm="editor", destination=destination, max_bytes=limit)
                    model.assert_not_called()
                    self.assertFalse(destination.exists())
            self.assertEqual((snapshot / "model.safetensors").read_bytes(), b"too large")
