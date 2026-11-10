"""Trusted Python execution publishes only a verified checkpoint."""
import hashlib
import json
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_editing import _files, _stack, write_tiny_model
from wsm import WeightModel, execute_program
from wsm.errors import WeightlessError


CASES = [{"prompt": "a b", "target": "c"}, {"prompt": "a", "target": "b"}]


def _write(path, text):
    path.write_text(text)
    return path


class BoundaryTests(unittest.TestCase):
    def test_existing_output_file_and_directory_are_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            program = _write(root / "edit.py", "def edit(model):\n    pass\n")
            output = root / "output"
            output.write_text("safe")
            with self.assertRaises(WeightlessError) as caught:
                execute_program(program, model_path=root, output=output, cases=CASES)
            self.assertEqual(caught.exception.code, "output_exists")
            self.assertEqual(output.read_text(), "safe")
            directory = root / "directory"
            directory.mkdir()
            marker = directory / "keep.txt"
            marker.write_text("keep")
            before = _files(directory)
            with self.assertRaises(WeightlessError) as again:
                execute_program(program, model_path=root, output=directory, cases=CASES)
            self.assertEqual(again.exception.code, "output_exists")
            self.assertEqual(_files(directory), before)
            self.assertFalse((directory / "run.json").exists())
            self.assertFalse((directory / "checkpoint").exists())

    def test_invalid_cases_and_missing_program_create_no_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            program = _write(root / "edit.py", "def edit(model):\n    pass\n")
            output = root / "output"
            with self.assertRaises(WeightlessError) as caught:
                execute_program(program, model_path=root, output=output, cases=[{"prompt": 1, "target": "b"}])
            self.assertEqual(caught.exception.code, "invalid_cases")
            self.assertFalse(output.exists())
            missing = root / "missing.py"
            with self.assertRaises(WeightlessError):
                execute_program(missing, model_path=root, output=output, cases=CASES)
            self.assertFalse(output.exists())

    def test_syntax_error_records_failure_without_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            program = _write(root / "edit.py", "def edit(model)\n    pass\n")
            output = root / "output"
            with self.assertRaises(SyntaxError):
                execute_program(program, model_path=model, output=output)
            journal = json.loads((output / "run.json").read_text())
            self.assertEqual(journal["state"], "failure")
            self.assertEqual(journal["phases"], ["ready", "running", "failure"])
            self.assertEqual(journal["error"]["type"], "SyntaxError")
            self.assertIsNone(journal["baseline"])
            self.assertIsNone(journal["checkpoint"])
            self.assertNotIn("published", journal["phases"])
            self.assertFalse((output / "checkpoint").exists())
            self.assertNotIn("NaN", (output / "run.json").read_text())
            self.assertNotIn("Infinity", (output / "run.json").read_text())

    def test_unreadable_snapshot_records_failure_without_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            program = _write(root / "edit.py", "def edit(model):\n    raise AssertionError('edit ran')\n")
            output = root / "output"
            with self.assertRaises(WeightlessError):
                execute_program(program, model_path=model, output=output, cases=CASES)
            journal = json.loads((output / "run.json").read_text())
            self.assertEqual(journal["state"], "failure")
            self.assertNotIn("published", journal["phases"])
            self.assertFalse((output / "checkpoint").exists())
            self.assertFalse(list(output.glob(".wsm-stage-*")))
            self.assertTrue(journal["source_sha256"].startswith("sha256:"))
            self.assertEqual(journal["source_sha256"], "sha256:" + hashlib.sha256(program.read_bytes()).hexdigest())


class ProgramTests(unittest.TestCase):
    def _run_cli(self, source, output):
        program = _write(self.root / "edit.py", source)
        cases = _write(self.root / "cases.json", json.dumps(CASES))
        return subprocess.run(
            [sys.executable, "-m", "wsm", "execute", str(program),
             "--model", str(self.root / "source"), "--cases", str(cases), "--output", output],
            cwd=self.root, capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")})

    def test_relative_output_stays_at_requested_location_after_chdir(self):
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        result = self._run_cli(
            "import os\ndef edit(model):\n    os.chdir(" + repr(str(elsewhere)) + ")\n"
            "    model.weights['transformer.ln_f.weight'].mul_(0.95)\n", "output")
        self.assertEqual(result.returncode, 0, result.stderr)
        output = self.root / "output"
        self.assertEqual(json.loads((output / "run.json").read_text())["state"], "published")
        WeightModel.load(output / "checkpoint")
        self.assertFalse((elsewhere / "output").exists())

    def test_system_exit_zero_is_a_failed_cli_run(self):
        result = self._run_cli("import sys\ndef edit(model):\n    sys.exit(0)\n", "output")
        self.assertNotEqual(result.returncode, 0)
        output = self.root / "output"
        self.assertEqual(json.loads((output / "run.json").read_text())["state"], "failure")
        self.assertEqual(json.loads(result.stderr.splitlines()[-1])["error"]["type"], "SystemExit")
        self.assertFalse((output / "checkpoint").exists())

    def test_executes_the_captured_source_when_file_changes_during_loading(self):
        from unittest.mock import patch
        program = _write(self.root / "edit.py", textwrap_edit())
        digest = "sha256:" + hashlib.sha256(program.read_bytes()).hexdigest()
        load = WeightModel.load

        def changed_source(*args, **kwargs):
            program.write_text('raise RuntimeError("changed after admission")')
            return load(*args, **kwargs)

        with patch.object(WeightModel, 'load', side_effect=changed_source):
            result = execute_program(program, model_path=self.source, output=self.root / 'captured', cases=CASES)
        self.assertEqual(result['state'], 'published')
        self.assertEqual(result['source_sha256'], digest)

    def setUp(self):
        self.tmp = None
        if _stack() is None:
            self.skipTest("torch, transformers, and tokenizers unavailable")
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = write_tiny_model(self.root / "source")

    def tearDown(self):
        if self.tmp is not None:
            self.tmp.cleanup()

    def test_published_run_preserves_baseline_and_writes_the_edited_tensor(self):
        import torch
        program = _write(self.root / "edit.py", textwrap_edit())
        cases = CASES
        baseline = WeightModel.load(self.source).evaluate(cases)
        source_files = _files(self.source)
        output = self.root / "run"
        result = execute_program(program, model_path=self.source, output=output, device="cpu", cases=cases)
        journal = json.loads((output / "run.json").read_text())
        self.assertEqual(result["state"], "published")
        self.assertEqual(result["phases"], ["ready", "running", "staging", "verified", "published"])
        self.assertEqual(journal, result)
        self.assertEqual(result["baseline"], baseline)
        self.assertEqual(result["source_sha256"], "sha256:" + hashlib.sha256(program.read_bytes()).hexdigest())
        self.assertNotIn("accepted", result)
        self.assertNotIn("selection", result)
        self.assertEqual(_files(self.source), source_files)
        self.assertIn("out-marker", result["stdout"]["text"])
        self.assertIn("err-marker", result["stderr"]["text"])
        changed = WeightModel.load(output / "checkpoint")
        fresh = WeightModel.load(self.source)
        expected = fresh.weights["transformer.ln_f.weight"].clone()
        expected[:16].mul_(0.95)
        self.assertTrue(torch.equal(changed.weights["transformer.ln_f.weight"], expected))
        self.assertTrue(torch.equal(
            changed.weights["transformer.wpe.weight"], fresh.weights["transformer.wpe.weight"]))
        self.assertEqual(result["candidate"], changed.evaluate(cases))
        self.assertNotEqual(result["baseline"]["loss"], result["candidate"]["loss"])
        self.assertEqual(
            changed.weights["lm_head.weight"].data_ptr(),
            changed.weights["transformer.wte.weight"].data_ptr())
        self.assertTrue((output / "checkpoint" / "model.safetensors").is_file())
        self.assertFalse(list(output.glob(".wsm-stage-*")))
        json.dumps(result, allow_nan=False)
        checkpoint_files = _files(output / "checkpoint")
        with self.assertRaises(WeightlessError) as caught:
            execute_program(program, model_path=self.source, output=output, cases=cases)
        self.assertEqual(caught.exception.code, "output_exists")
        self.assertEqual(_files(output / "checkpoint"), checkpoint_files)
        self.assertEqual(json.loads((output / "run.json").read_text())["state"], "published")

    def test_import_error_does_not_publish(self):
        program = _write(self.root / "edit.py", "import wsm_test_missing_module_zz\n\ndef edit(model):\n    pass\n")
        output = self.root / "run"
        with self.assertRaises(ModuleNotFoundError):
            execute_program(program, model_path=self.source, output=output, cases=CASES)
        journal = json.loads((output / "run.json").read_text())
        self.assertEqual(journal["state"], "failure")
        self.assertEqual(journal["error"]["type"], "ModuleNotFoundError")
        self.assertIsNotNone(journal["baseline"])
        self.assertIsNone(journal["candidate"])
        self.assertNotIn("published", journal["phases"])
        self.assertFalse((output / "checkpoint").exists())

    def test_edit_exception_keeps_baseline_stdout_and_no_checkpoint(self):
        program = _write(
            self.root / "edit.py",
            "def edit(model):\n    print('before-fail')\n    raise RuntimeError('nope')\n")
        output = self.root / "run"
        with self.assertRaises(RuntimeError):
            execute_program(program, model_path=self.source, output=output, cases=CASES)
        journal = json.loads((output / "run.json").read_text())
        self.assertEqual(journal["state"], "failure")
        self.assertEqual(journal["phases"], ["ready", "running", "failure"])
        losses = [row["target_nll"] for row in journal["baseline"]["cases"]]
        self.assertEqual(journal["baseline"]["loss"], sum(losses) / len(losses))
        self.assertIsNone(journal["candidate"])
        self.assertIn("before-fail", journal["stdout"]["text"])
        self.assertEqual(journal["error"]["message"], "nope")
        self.assertFalse((output / "checkpoint").exists())
        self.assertFalse(list(output.glob(".wsm-stage-*")))

    def test_interrupt_during_edit_does_not_publish(self):
        program = _write(self.root / "edit.py", "def edit(model):\n    raise KeyboardInterrupt()\n")
        output = self.root / "run"
        with self.assertRaises(KeyboardInterrupt):
            execute_program(program, model_path=self.source, output=output, cases=CASES)
        journal = json.loads((output / "run.json").read_text())
        self.assertEqual(journal["state"], "interrupted")
        self.assertEqual(journal["phases"], ["ready", "running", "interrupted"])
        self.assertIsNotNone(journal["baseline"])
        self.assertIsNone(journal["candidate"])
        self.assertFalse((output / "checkpoint").exists())
        self.assertFalse(list(output.glob(".wsm-stage-*")))

    def test_nonfinite_edit_does_not_publish(self):
        program = _write(
            self.root / "edit.py",
            "def edit(model):\n"
            "    model.weights['transformer.ln_f.weight'].fill_(float('nan'))\n")
        output = self.root / "run"
        with self.assertRaises(WeightlessError) as caught:
            execute_program(program, model_path=self.source, output=output, cases=CASES)
        self.assertEqual(caught.exception.code, "nonfinite")
        journal = json.loads((output / "run.json").read_text())
        self.assertEqual(journal["state"], "failure")
        self.assertIsNotNone(journal["baseline"])
        self.assertIsNone(journal["candidate"])
        self.assertFalse((output / "checkpoint").exists())
        json.dumps(journal, allow_nan=False)


def textwrap_edit():
    return (
        "def edit(model):\n"
        "    print('out-marker')\n"
        "    import sys\n"
        "    print('err-marker', file=sys.stderr)\n"
        "    model.weights['transformer.ln_f.weight'][:16].mul_(0.95)\n"
    )


if __name__ == "__main__":
    unittest.main()
