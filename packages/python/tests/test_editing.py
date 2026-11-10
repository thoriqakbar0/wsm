"""Live tensor edits, teacher-forced loss, and verified checkpoint publication."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

from wsm import WeightModel, WeightlessError

ROOT = Path(__file__).resolve().parents[1]


def _stack():
    try:
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
    except ImportError:
        return None
    return {
        "torch": torch,
        "config": GPT2Config,
        "model": GPT2LMHeadModel,
        "tokenizer": PreTrainedTokenizerFast,
        "backend": Tokenizer,
        "word_level": WordLevel,
        "whitespace": Whitespace,
    }


def write_tiny_model(directory):
    stack = _stack()
    if stack is None:
        raise unittest.SkipTest("torch, transformers, and tokenizers unavailable")
    directory.mkdir(parents=True, exist_ok=True)
    backend = stack["backend"](stack["word_level"](
        {"<unk>": 0, "a": 1, "b": 2, "c": 3, "d": 4}, unk_token="<unk>"))
    backend.pre_tokenizer = stack["whitespace"]()
    tokenizer = stack["tokenizer"](
        tokenizer_object=backend, unk_token="<unk>", eos_token="d", pad_token="d")
    config = stack["config"](
        vocab_size=5, n_positions=32, n_embd=16, n_layer=1, n_head=2, n_inner=32,
        bos_token_id=4, eos_token_id=4, pad_token_id=4, tie_word_embeddings=True)
    stack["model"](config).save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return directory


def _files(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _manual_nll(model, tokenizer, prompt, target):
    import torch
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    target_ids = list(tokenizer.encode(target, add_special_tokens=False))
    ids = torch.tensor([prompt_ids + target_ids], dtype=torch.long, device=model.device)
    with torch.inference_mode():
        logits = model.model(input_ids=ids).logits
    start = len(prompt_ids) - 1
    chosen = logits[0, start:start + len(target_ids), :]
    labels = torch.tensor(target_ids, dtype=torch.long, device=model.device)
    return float(torch.nn.functional.cross_entropy(chosen, labels).item())


class ImportTests(unittest.TestCase):
    def test_package_import_does_not_load_torch(self):
        script = textwrap.dedent("""
            import importlib.abc
            import sys

            class Block(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path, target=None):
                    if fullname == "torch" or fullname.startswith("torch."):
                        raise ModuleNotFoundError(fullname)
                    if fullname == "transformers" or fullname.startswith("transformers."):
                        raise ModuleNotFoundError(fullname)
                    return None

            sys.meta_path.insert(0, Block())
            import wsm
            from wsm import SearchRun, Session, WeightModel, execute_program
            assert WeightModel.__name__ == "WeightModel"
            assert callable(execute_program)
            assert "torch" not in sys.modules
            assert "transformers" not in sys.modules
            print("ok")
        """)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        completed = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("ok", completed.stdout)


class EditingTests(unittest.TestCase):
    def test_interrupt_after_rename_does_not_report_unpublished(self):
        from unittest.mock import patch
        from wsm.checkpoint_export import _publish
        model = WeightModel.load(self.source)
        destination = self.root / 'interrupt-after-rename'
        phases = []

        def publish_then_interrupt(source, target):
            _publish(source, target)
            raise KeyboardInterrupt()

        with patch('wsm.checkpoint_export._publish', publish_then_interrupt):
            with self.assertRaises(KeyboardInterrupt):
                model.save(destination, on_phase=phases.append)
        self.assertTrue(destination.is_dir())
        self.assertNotIn('interrupted', phases)
        WeightModel.load(destination)

    def setUp(self):
        self.tmp = None
        if _stack() is None:
            self.skipTest("torch, transformers, and tokenizers unavailable")
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = write_tiny_model(self.root / "source")
        self.model = WeightModel.load(self.source)

    def tearDown(self):
        if self.tmp is not None:
            self.tmp.cleanup()

    def test_inplace_slice_edits_tied_parameter_storage(self):
        import torch
        norm = "transformer.ln_f.weight"
        before = self.model.model.get_parameter(norm).detach().clone()
        self.model.weights[norm][:16].mul_(0.95)
        expected = before.clone()
        expected[:16].mul_(0.95)
        self.assertTrue(torch.equal(self.model.model.get_parameter(norm).detach(), expected))
        self.assertFalse(self.model.weights[norm].requires_grad)
        tied = self.model.weights["lm_head.weight"]
        source = self.model.weights["transformer.wte.weight"]
        self.assertEqual(tied.data_ptr(), source.data_ptr())
        tied[0, 0] = 0.25
        self.assertEqual(source[0, 0].item(), 0.25)
        self.assertIn("transformer.wte.weight", self.model.weights)
        with self.assertRaises(KeyError):
            self.model.weights["missing.weight"]

    def test_evaluate_is_teacher_forced_target_nll(self):
        import torch
        cases = [{"prompt": "a b", "target": "c"}, {"prompt": "a", "target": "b"}]
        untouched = {name: tensor.detach().clone() for name, tensor in self.model.model.named_parameters()}
        result = self.model.evaluate(cases)
        again = self.model.evaluate(cases)
        manual = [_manual_nll(self.model, self.model.tokenizer, case["prompt"], case["target"]) for case in cases]
        self.assertEqual(result, again)
        self.assertEqual([row["target_nll"] for row in result["cases"]], manual)
        self.assertEqual(result["loss"], sum(manual) / len(manual))
        json.dumps(result, allow_nan=False)
        for name, tensor in self.model.model.named_parameters():
            self.assertTrue(torch.equal(tensor.detach(), untouched[name]))
        self.assertEqual(self.model.evaluate([]), {"loss": None, "cases": []})
        self.assertFalse(self.model.model.training)
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in self.model.model.parameters()))

    def test_save_reloads_exact_tied_float32_checkpoint(self):
        import torch
        norm = "transformer.ln_f.weight"
        self.model.weights[norm][:16].mul_(0.95)
        expected = self.model.weights[norm].detach().cpu().clone()
        source_files = _files(self.source)
        destination = self.root / "checkpoint"
        phases = []
        receipt = self.model.save(destination, on_phase=phases.append)
        self.assertEqual(phases, ["ready", "running", "staging", "verified", "published"])
        self.assertEqual(receipt["state"], "published")
        self.assertEqual(_files(self.source), source_files)
        self.assertTrue((destination / "model.safetensors").is_file())
        self.assertTrue((destination / "tokenizer_config.json").is_file() or (destination / "tokenizer.json").is_file())
        reloaded = WeightModel.load(destination)
        self.assertTrue(torch.equal(reloaded.weights[norm].cpu(), expected))
        for name, tensor in self.model.model.named_parameters(remove_duplicate=False):
            self.assertTrue(torch.equal(reloaded.weights[name].cpu(), tensor.detach().cpu()))
        self.assertEqual(
            reloaded.weights["lm_head.weight"].data_ptr(),
            reloaded.weights["transformer.wte.weight"].data_ptr())
        self.assertFalse(list(self.root.glob(".wsm-stage-*")))

    def test_nonfinite_save_leaves_no_destination_or_staging(self):
        destination = self.root / "checkpoint"
        self.model.weights["transformer.ln_f.weight"].fill_(float("nan"))
        with self.assertRaises(WeightlessError) as caught:
            self.model.save(destination)
        self.assertEqual(caught.exception.code, "nonfinite")
        self.assertFalse(destination.exists())
        self.assertFalse(list(self.root.glob(".wsm-stage-*")))

    def test_existing_destination_is_not_modified(self):
        destination = self.root / "checkpoint"
        destination.mkdir()
        marker = destination / "keep.txt"
        marker.write_text("keep")
        digest = hashlib.sha256(marker.read_bytes()).hexdigest()
        with self.assertRaises(WeightlessError) as caught:
            self.model.save(destination)
        self.assertEqual(caught.exception.code, "export_exists")
        self.assertEqual(marker.read_text(), "keep")
        self.assertEqual(hashlib.sha256(marker.read_bytes()).hexdigest(), digest)
        self.assertEqual(list(destination.iterdir()), [marker])

    def test_interrupt_before_rename_removes_staging(self):
        destination = self.root / "checkpoint"
        seen = []

        def on_phase(state):
            seen.append(state)
            if state == "verified":
                raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            self.model.save(destination, on_phase=on_phase)
        self.assertIn("verified", seen)
        self.assertNotIn("published", seen)
        self.assertFalse(destination.exists())
        self.assertFalse(list(self.root.glob(".wsm-stage-*")))

    def test_callback_error_after_publication_keeps_checkpoint(self):
        import torch
        destination = self.root / "checkpoint"
        expected = self.model.weights["transformer.wte.weight"].detach().cpu().clone()

        def on_phase(state):
            if state == "published":
                self.assertTrue(destination.is_dir())
                raise RuntimeError("after publication")

        with self.assertRaises(RuntimeError):
            self.model.save(destination, on_phase=on_phase)
        reloaded = WeightModel.load(destination)
        self.assertTrue(torch.equal(reloaded.weights["transformer.wte.weight"].cpu(), expected))
        self.assertEqual(
            reloaded.weights["lm_head.weight"].data_ptr(),
            reloaded.weights["transformer.wte.weight"].data_ptr())


if __name__ == "__main__":
    unittest.main()
