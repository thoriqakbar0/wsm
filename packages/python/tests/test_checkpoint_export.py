import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from wsm.binary64 import WeightlessError, create_checkpoint, parse_checkpoint
from wsm.checkpoint_export import export_checkpoint
from wsm.session import Session, _hash

ROOT = Path(__file__).resolve().parents[3]


def drive_module():
    path = ROOT / "scripts" / "drive_session.py"
    spec = importlib.util.spec_from_file_location("drive_session_for_export", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def graph():
    return {"inputSize": 1, "nodes": [{"id": "out", "op": "affine", "input": "input", "weight": "w", "bias": "b"}], "output": "out"}


def target(weight=0, bias=0):
    checkpoint = create_checkpoint(graph(), {"w": {"shape": [1, 1], "values": [weight]}, "b": {"shape": [1], "values": [bias]}})
    return {"kind": "binary64", "checkpoint": checkpoint}


def declaration():
    return {"hypothesis": "A scalar replacement solves the finite task", "arms": ["solver"], "editable": ["w", "b"],
            "budgets": {"attempts": 2, "evaluations": 4, "inspections": 0, "teacher_requests": 0,
                        "final_evaluations": 0, "forward_calls": 16},
            "max_edits": 2, "max_inspect": 1, "wall_seconds": 60, "claim": "exploratory", "selection": "loss"}


def reseal(value):
    body = {key: item for key, item in value.items() if key != "id"}
    return {**body, "id": _hash(body)}


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.runner = drive_module().InProcessBinaryRunner()
        self.session = None

    def tearDown(self):
        if self.session is not None:
            self.session.close()
        self.tmp.cleanup()

    def test_import_does_not_load_torch(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "packages" / "python" / "src")
        code = ("import sys, wsm; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules; "
                "assert 'wsm.checkpoint_export' not in sys.modules; "
                "from wsm import export_checkpoint; assert 'torch' not in sys.modules; assert callable(export_checkpoint)")
        subprocess.run([sys.executable, "-c", code], check=True, env=env)

    def session_with(self, model, edits, cases):
        self.session = Session.create(self.root / "session.db", runner=self.runner, target=model,
                                      declaration=declaration(), development=cases)
        view = self.session.observe("solver")
        proposal = {key: view[key] for key in ("session", "arm", "declaration", "observation", "base")}
        proposal["edits"] = edits
        self.session.submit(proposal, key="edit")
        self.session.freeze(key="freeze")
        return self.session.export()

    def test_binary_export_reloads_selected_values(self):
        model = target()
        cases = [{"id": "row", "input": [2], "target": [7]}]
        bundle = self.session_with(model, [{"tensor": "w", "index": 0, "value": 2}, {"tensor": "b", "index": 0, "value": 3}], cases)
        expected = create_checkpoint(graph(), {"w": {"shape": [1, 1], "values": [2]}, "b": {"shape": [1], "values": [3]}})["id"]
        self.assertEqual(bundle["frozen"]["arms"]["solver"]["checkpoint"], expected)
        self.assertNotEqual(expected, bundle["initial"])
        destination = self.root / "checkpoint"
        receipt = export_checkpoint(bundle, target=model, arm="solver", destination=destination, max_bytes=100000)
        loaded = parse_checkpoint((destination / "checkpoint.json").read_text())
        self.assertEqual(loaded["id"], expected)
        self.assertEqual(loaded["tensors"]["w"]["values"], [2.0])
        self.assertEqual(loaded["tensors"]["b"]["values"], [3.0])
        self.assertEqual(receipt["selected"], expected)
        self.assertEqual(receipt["initial"], bundle["initial"])
        self.assertEqual(receipt["evidence"], bundle["id"])
        self.assertEqual(receipt["frozen"], bundle["frozen"]["id"])
        self.assertEqual(receipt["runner"], bundle["runner"])
        self.assertEqual(receipt["runtime"]["python"], sys.version.split()[0])
        self.assertEqual(receipt["target"], {"kind": "binary64"})
        self.assertNotIn(str(self.root), json.dumps(receipt))
        for record in receipt["files"]:
            self.assertFalse(record["path"].startswith("/"))
            data = (destination / record["path"]).read_bytes()
            self.assertEqual(record["bytes"], len(data))
            self.assertEqual(record["sha256"], "sha256:" + hashlib.sha256(data).hexdigest())
        self.assertFalse((destination / "receipt.json").read_text().find(str(self.root)) >= 0)
        with self.assertRaises(WeightlessError) as error:
            export_checkpoint(bundle, target=model, arm="solver", destination=destination, max_bytes=100000)
        self.assertEqual(error.exception.code, "export_exists")

    def test_negative_zero_exports_canonical_zero(self):
        model = target(bias=1)
        bundle = self.session_with(model, [{"tensor": "b", "index": 0, "value": -0.0}],
                                    [{"id": "row", "input": [4], "target": [0]}])
        expected = create_checkpoint(graph(), {"w": {"shape": [1, 1], "values": [0]}, "b": {"shape": [1], "values": [0.0]}})["id"]
        destination = self.root / "zero"
        export_checkpoint(bundle, target=model, arm="solver", destination=destination, max_bytes=100000)
        text = (destination / "checkpoint.json").read_text()
        self.assertNotIn("-0", text)
        loaded = parse_checkpoint(text)
        self.assertEqual(loaded["id"], expected)
        self.assertEqual(loaded["tensors"]["b"]["values"][0], 0.0)

    def test_rejects_bad_evidence_and_cleans_a_failed_publish(self):
        model = target()
        bundle = self.session_with(model, [{"tensor": "w", "index": 0, "value": 2}, {"tensor": "b", "index": 0, "value": 3}],
                                    [{"id": "row", "input": [2], "target": [7]}])
        bare = self.root / "missing-parent" / "out"
        with self.assertRaises(WeightlessError):
            export_checkpoint(bundle, target=model, arm="missing", destination=self.root / "arm", max_bytes=100000)
        self.assertFalse((self.root / "arm").exists())
        changed = json.loads(json.dumps(model))
        changed["checkpoint"]["tensors"]["b"]["values"] = [1]
        changed["checkpoint"] = create_checkpoint(changed["checkpoint"]["graph"], {"w": changed["checkpoint"]["tensors"]["w"],
                                                                                    "b": {"shape": [1], "values": [1]}})
        with self.assertRaises(WeightlessError) as mismatch:
            export_checkpoint(bundle, target=changed, arm="solver", destination=self.root / "wrong", max_bytes=100000)
        self.assertEqual(mismatch.exception.code, "identity_mismatch")
        self.assertFalse(any(path.name.startswith(".wsm-export-") for path in self.root.iterdir()))
        tampered = json.loads(json.dumps(bundle))
        fake = "sha256:" + "ab" * 32
        for op in tampered["operations"]:
            if op.get("selection") and op["selection"].get("accepted"):
                op["result"]["checkpoint"] = fake
                op["selection"]["after"] = fake
                key = op["key"]
        for event in tampered["journal"]:
            if event["type"] == "result" and event["key"] == key:
                event["result"]["checkpoint"] = fake
            if event["type"] == "selection" and event["key"] == key:
                event["decision"]["after"] = fake
        tampered["frozen"]["arms"]["solver"]["checkpoint"] = fake
        tampered["frozen"] = reseal(tampered["frozen"])
        tampered = reseal(tampered)
        with self.assertRaises(WeightlessError) as tamper:
            export_checkpoint(tampered, target=model, arm="solver", destination=self.root / "tamper", max_bytes=100000)
        self.assertEqual(tamper.exception.code, "identity_mismatch")
        self.assertFalse((self.root / "tamper").exists())
        occupied = self.root / "occupied"
        occupied.mkdir()
        marker = occupied / "keep"
        marker.write_text("keep")
        with self.assertRaises(WeightlessError) as exists:
            export_checkpoint(bundle, target=model, arm="solver", destination=occupied, max_bytes=100000)
        self.assertEqual(exists.exception.code, "export_exists")
        self.assertEqual(marker.read_text(), "keep")
        with self.assertRaises(WeightlessError) as small:
            export_checkpoint(bundle, target=model, arm="solver", destination=self.root / "small", max_bytes=32)
        self.assertEqual(small.exception.code, "export_too_large")
        self.assertFalse((self.root / "small").exists())
        self.assertFalse(any(path.name.startswith(".wsm-export-") for path in self.root.iterdir()))
        export_checkpoint(bundle, target=model, arm="solver", destination=self.root / "small", max_bytes=100000)
        self.assertTrue((self.root / "small" / "checkpoint.json").is_file())
        with self.assertRaises(WeightlessError):
            export_checkpoint(bundle, target=model, arm="solver", destination=bare, max_bytes=100000)
        self.session.close()
        self.session = Session.create(self.root / "plain.db", runner=self.runner, target=model,
                                      declaration=declaration(), development=[{"id": "row", "input": [1], "target": [1]}])
        incomplete = self.session.export()
        self.session.freeze(key="freeze")
        incomplete = self.session.export()
        with self.assertRaises(WeightlessError) as frozen:
            export_checkpoint(incomplete, target=model, arm="solver", destination=self.root / "incomplete", max_bytes=100000)
        self.assertEqual(frozen.exception.code, "incomplete_freeze")
        self.assertFalse((self.root / "incomplete").exists())

    def test_publication_preserves_a_concurrently_created_empty_directory(self):
        import wsm.checkpoint_export as exporter
        model = target()
        bundle = self.session_with(model, [{"tensor": "w", "index": 0, "value": 2}],
                                   [{"id": "row", "input": [2], "target": [4]}])
        destination = self.root / "concurrent"
        inode = []
        synchronize = exporter._fsync_tree
        def create_destination(staged):
            synchronize(staged)
            destination.mkdir()
            inode.append(destination.stat().st_ino)
        with patch.object(exporter, "_fsync_tree", side_effect=create_destination):
            with self.assertRaises(WeightlessError) as error:
                export_checkpoint(bundle, target=model, arm="solver", destination=destination, max_bytes=100000)
        self.assertEqual(error.exception.code, "export_exists")
        self.assertEqual(destination.stat().st_ino, inode[0])
        self.assertEqual(list(destination.iterdir()), [])
        self.assertFalse(any(path.name.startswith(".wsm-export-") for path in self.root.iterdir()))

    def test_rejects_falsely_complete_and_malformed_evidence(self):
        model = target()
        self.session = Session.create(self.root / "session.db", runner=self.runner, target=model,
                                      declaration=declaration(), development=[{"id": "row", "input": [1], "target": [1]}])
        self.session.freeze(key="freeze")
        bundle = self.session.export()
        bundle["frozen"] = reseal({**bundle["frozen"], "complete": True})
        bundle = reseal(bundle)
        for evidence in (bundle, reseal({})):
            with self.subTest(evidence=evidence):
                with self.assertRaises(WeightlessError) as error:
                    export_checkpoint(evidence, target=model, arm="solver", destination=self.root / "invalid", max_bytes=100000)
                self.assertEqual(error.exception.code, "invalid_artifact")
                self.assertFalse((self.root / "invalid").exists())

    def test_deep_weight_header_has_a_structured_error(self):
        from wsm.checkpoint_export import _stored_bytes
        header = b'[' * 10000 + b'0' + b']' * 10000
        path = self.root / "model.safetensors"
        path.write_bytes(len(header).to_bytes(8, "little") + header)
        with self.assertRaises(WeightlessError) as error:
            _stored_bytes(path)
        self.assertEqual(error.exception.code, "invalid_target")


def _smollm_ready():
    source = os.environ.get("WSM_SMOLLM_SOURCE")
    if not source:
        return False
    try:
        import torch
        import transformers
    except ImportError:
        return False
    snapshot = Path(source) / "snapshots" / "12fd25f77366fa6b3b4b768ec3050bf629380bac"
    return snapshot.is_dir() and torch.__version__ and transformers.__version__


@unittest.skipUnless(_smollm_ready(), "set WSM_SMOLLM_SOURCE to a staged SmolLM snapshot and install torch")
class SmolLMExportTests(unittest.TestCase):
    def test_edited_snapshot_preserves_metadata_and_identity_on_reload(self):
        import gc
        from wsm.target_worker import ModelTarget, run
        source = str(Path(os.environ["WSM_SMOLLM_SOURCE"]).resolve())
        model = {"kind": "smollm", "source": source, "revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac",
                 "model": "HuggingFaceTB/SmolLM2-135M-Instruct", "dtype": "float32"}
        loaded = ModelTarget(model)
        original = loaded.value("model.norm.weight", 0)
        edited = original - 0.125
        del loaded
        gc.collect()
        before = _fingerprint(Path(source))

        class RealModelRunner:
            """Trusted in-process numerical execution for the optional model check."""

            config = {"mode": "inprocess-smollm-test", "isolation": "none"}

            def execute(self, action, target, *, patches=None, edits=None, cases=None, query=None,
                        generate=False, purpose="development", timeout=None, on_dispatch=None, forward_limit=None):
                request = {"action": action, "target": target, "patches": [] if patches is None else patches,
                           "edits": [] if edits is None else edits, "cases": cases, "query": query,
                           "generate": generate, "purpose": purpose}
                if forward_limit is not None:
                    request["forward_limit"] = forward_limit
                if on_dispatch is not None:
                    on_dispatch()
                return run(request)

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        session = Session.create(root / "session.db", runner=RealModelRunner(), target=model, declaration={
            "hypothesis": "A norm edit is a finite candidate", "arms": ["solver"], "editable": ["model.norm.weight"],
            "budgets": {"attempts": 1, "evaluations": 4, "inspections": 0, "teacher_requests": 0,
                        "final_evaluations": 0, "forward_calls": 128},
            "max_edits": 1, "max_inspect": 1, "wall_seconds": 600, "claim": "exploratory", "selection": "loss"},
            development=[{"id": "duck", "prompt": "What sound does a duck make?", "target": "quack quack turnip"},
                         {"id": "cow", "prompt": "What sound does a cow make?", "target": "moo moo comet"}])
        try:
            view = session.observe("solver")
            proposal = {key: view[key] for key in ("session", "arm", "declaration", "observation", "base")}
            proposal["edits"] = [{"tensor": "model.norm.weight", "index": 0, "value": edited}]
            submitted = session.submit(proposal, key="edit")
            self.assertTrue(submitted["selection"]["accepted"])
            session.freeze(key="freeze")
            bundle = session.export()
        finally:
            session.close()
        self.assertNotEqual(bundle["frozen"]["arms"]["solver"]["checkpoint"], bundle["initial"])
        destination = root / "exported"
        receipt = export_checkpoint(bundle, target=model, arm="solver", destination=destination, max_bytes=700_000_000)
        self.assertNotIn(source, json.dumps(receipt))
        self.assertEqual(_fingerprint(Path(source)), before)
        snapshot = destination / "snapshots" / model["revision"]
        original_snapshot = Path(source) / "snapshots" / model["revision"]
        for path in original_snapshot.iterdir():
            if path.suffix in {".json", ".txt", ".model", ".jinja"}:
                self.assertEqual((snapshot / path.name).read_bytes(), path.read_bytes(), path.name)
        self.assertTrue((snapshot / "model.safetensors").is_file())
        env = os.environ.copy()
        env.update(PYTHONPATH=str(ROOT / "packages" / "python" / "src"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   HF_HOME=str(root / "empty-hub"), HUGGINGFACE_HUB_CACHE=str(root / "empty-hub"), TRANSFORMERS_CACHE=str(root / "empty-hub"))
        for name in list(env):
            if "HF_HUB" in name and name not in {"HF_HUB_OFFLINE", "HUGGINGFACE_HUB_CACHE"}:
                env.pop(name)
        code = ("import json,sys\nfrom wsm.target_worker import ModelTarget\n"
                "target=json.loads(sys.stdin.read())\nmodel=ModelTarget(target)\n"
                "json.dump({'identity':model.identity(),'alias':model.aliases.get('lm_head.weight'),'value':model.value('model.norm.weight',0)},sys.stdout)\n")
        exported = {"kind": "smollm", "source": str(destination), "revision": model["revision"], "model": model["model"], "dtype": "float32"}
        completed = subprocess.run([sys.executable, "-c", code], input=json.dumps(exported).encode(), capture_output=True, env=env, check=True)
        fresh = json.loads(completed.stdout)
        patch = next(item["value"] for item in bundle["frozen"]["arms"]["solver"]["patches"]
                     if item["tensor"] == "model.norm.weight" and item["index"] == 0)
        self.assertEqual(fresh["identity"], bundle["frozen"]["arms"]["solver"]["checkpoint"])
        self.assertEqual(fresh["alias"], "model.embed_tokens.weight")
        self.assertNotEqual(fresh["value"], original)
        self.assertEqual(fresh["value"], patch)


def _fingerprint(root):
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


if __name__ == "__main__":
    unittest.main()
