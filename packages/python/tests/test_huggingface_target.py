import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from wsm import WeightlessError, stage_model, stage_smollm
from wsm.sandbox import DockerRunner
from wsm.target_worker import manifest_digest, run, state_id

REV = "0123456789abcdef0123456789abcdef01234567"
PINNED = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
SMOLLM = Path("/root/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct")


def snapshot(root, revision=REV):
    path = Path(root) / "snapshots" / revision
    path.mkdir(parents=True)
    return path


def config(path, **extra):
    body = {"model_type": "gpt2"}
    body.update(extra)
    (path / "config.json").write_text(json.dumps(body), encoding="utf-8")


def weights(path, name="model.safetensors", content=b"weights"):
    (path / name).write_bytes(content)


def request(target, action="describe", **extra):
    body = {"target": target, "action": action, "patches": [], "edits": [],
            "cases": None, "query": None, "generate": False}
    body.update(extra)
    return body


def digest_of(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class StageTests(unittest.TestCase):
    def test_stage_copies_blob_links_and_files_outside_the_smollm_suffix_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            snap = snapshot(root)
            blob = root / "blobs" / "weight"
            blob.parent.mkdir()
            blob.write_bytes(b"safetensor-bytes")
            config(snap)
            (snap / "model.safetensors").symlink_to(Path("../../blobs/weight"))
            (snap / "notes.md").write_text("beyond-allowlist", encoding="utf-8")
            nested = snap / "additional_chat_templates"
            nested.mkdir()
            (nested / "extra.jinja").write_text("extra template", encoding="utf-8")
            before = {path.relative_to(root).as_posix(): digest_of(path) for path in root.rglob("*") if path.is_file()}
            links = [path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_symlink()]
            destination = Path(tmp) / "staged"
            staged = stage_model(root, destination, model="fixture/tiny", revision=REV)
            self.assertEqual(staged, {"kind": "huggingface", "source": str(destination.resolve()),
                                      "revision": REV, "model": "fixture/tiny", "dtype": "float32"})
            copied = destination / "snapshots" / REV
            self.assertFalse((copied / "model.safetensors").is_symlink())
            self.assertEqual((copied / "model.safetensors").read_bytes(), b"safetensor-bytes")
            self.assertEqual((copied / "model.safetensors").stat().st_mode & 0o777, 0o444)
            self.assertEqual((copied / "notes.md").read_text(encoding="utf-8"), "beyond-allowlist")
            self.assertEqual((copied / "additional_chat_templates" / "extra.jinja").read_text(encoding="utf-8"),
                             "extra template")
            after = {path.relative_to(root).as_posix(): digest_of(path) for path in root.rglob("*") if path.is_file()}
            self.assertEqual(before, after)
            self.assertEqual(links, [path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_symlink()])

    def test_shared_blob_store_is_copied_but_not_mounted_until_it_is_inside_the_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models--fixture--tiny"
            snap = snapshot(root)
            config(snap)
            outside = Path(tmp) / "blobs" / "aa" / ("a" * 64)
            outside.parent.mkdir(parents=True)
            outside.write_bytes(b"dedup-weight")
            blob = root / "blobs" / "hash"
            blob.parent.mkdir()
            blob.symlink_to(outside)
            (snap / "model.safetensors").symlink_to(Path("../../blobs/hash"))
            destination = Path(tmp) / "staged"
            stage_model(root, destination, model="fixture/tiny", revision=REV)
            copied = destination / "snapshots" / REV / "model.safetensors"
            self.assertFalse(copied.is_symlink())
            self.assertEqual(copied.read_bytes(), b"dedup-weight")
            with patch("wsm.sandbox.subprocess.run") as inspect:
                inspect.return_value.stdout = "sha256:" + "a" * 64
                runner = DockerRunner("trusted")
            target = {"kind": "huggingface", "source": str(root.resolve()), "revision": REV,
                      "model": "fixture/tiny", "dtype": "float32"}
            with patch.object(runner, "_run") as run_mock:
                with self.assertRaises(WeightlessError) as error:
                    runner.execute("describe", target)
            run_mock.assert_not_called()
            self.assertEqual(error.exception.code, "invalid_target")
            mounted = {"kind": "huggingface", "source": str(destination.resolve()), "revision": REV,
                       "model": "fixture/tiny", "dtype": "float32"}
            with patch.object(runner, "_run", return_value=b'{"result":{"loaded":true}}'):
                self.assertEqual(runner.execute("describe", mounted), {"loaded": True})

    def test_stage_smollm_keeps_the_pinned_descriptor_and_suffix_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            snap = snapshot(root, PINNED)
            weights(snap, content=b"pinned-weights")
            config(snap)
            (snap / "notes.md").write_text("skip", encoding="utf-8")
            destination = Path(tmp) / "staged"
            staged = stage_smollm(root, destination)
            self.assertEqual(staged, {"kind": "smollm", "source": str(destination.resolve()), "revision": PINNED,
                                      "model": "HuggingFaceTB/SmolLM2-135M-Instruct", "dtype": "float32"})
            copied = destination / "snapshots" / PINNED
            self.assertEqual((copied / "model.safetensors").read_bytes(), b"pinned-weights")
            self.assertFalse((copied / "notes.md").exists())
            self.assertEqual((snap / "notes.md").read_text(encoding="utf-8"), "skip")

    def test_rejects_malformed_identity_escape_and_remote_code_without_leaving_a_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            snap = snapshot(root)
            config(snap)
            weights(snap)
            destination = Path(tmp) / "staged"
            destination.mkdir()
            (destination / "keep").write_text("keep", encoding="utf-8")
            with self.assertRaises(WeightlessError) as error:
                stage_model(root, destination, model="fixture/tiny", revision=REV)
            self.assertEqual(error.exception.code, "invalid_target")
            self.assertEqual((destination / "keep").read_text(encoding="utf-8"), "keep")
            alias = Path(tmp) / "source-alias"
            alias.symlink_to(root, target_is_directory=True)
            with self.assertRaises(WeightlessError):
                stage_model(root, alias / "inside", model="fixture/tiny", revision=REV)
            self.assertFalse((root / "inside").exists())

            outside = Path(tmp) / "outside"
            outside.write_bytes(b"secret")
            escaped = Path(tmp) / "escaped"
            snap = snapshot(escaped)
            config(snap)
            (snap / "model.safetensors").symlink_to(outside)
            with self.assertRaises(WeightlessError) as error:
                stage_model(escaped, Path(tmp) / "out-escape", model="fixture/tiny", revision=REV)
            self.assertEqual(error.exception.code, "invalid_target")
            self.assertFalse((Path(tmp) / "out-escape").exists())
            self.assertEqual(outside.read_bytes(), b"secret")
            chain = root / "blobs" / "chain"
            chain.parent.mkdir()
            chain.symlink_to(outside)
            (root / "snapshots" / REV / "notes.md").symlink_to("../../blobs/chain")
            with self.assertRaises(WeightlessError) as error:
                stage_model(root, Path(tmp) / "out-chain", model="fixture/tiny", revision=REV)
            self.assertEqual(error.exception.code, "invalid_target")
            self.assertFalse((Path(tmp) / "out-chain").exists())

            linked = Path(tmp) / "linked"
            real = Path(tmp) / "real"
            real.mkdir()
            config(real)
            weights(real)
            (linked / "snapshots").mkdir(parents=True)
            (linked / "snapshots" / REV).symlink_to(real)
            with self.assertRaises(WeightlessError) as error:
                stage_model(linked, Path(tmp) / "out-link", model="fixture/tiny", revision=REV)
            self.assertEqual(error.exception.code, "invalid_target")
            self.assertFalse((Path(tmp) / "out-link").exists())

            remote = Path(tmp) / "remote"
            snap = snapshot(remote)
            config(snap, auto_map={"AutoModelForCausalLM": "modeling.Custom"})
            weights(snap)
            (snap / "modeling.py").write_text("class Custom: pass\n", encoding="utf-8")
            with self.assertRaises(WeightlessError) as error:
                stage_model(remote, Path(tmp) / "out-remote", model="fixture/tiny", revision=REV)
            self.assertEqual(error.exception.code, "unsupported_target")
            self.assertFalse((Path(tmp) / "out-remote").exists())

            indexed = Path(tmp) / "indexed"
            snap = snapshot(indexed)
            config(snap)
            weights(snap)
            (snap / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"w": "../evil.safetensors"}}), encoding="utf-8")
            with self.assertRaises(WeightlessError) as error:
                stage_model(indexed, Path(tmp) / "out-index", model="fixture/tiny", revision=REV)
            self.assertEqual(error.exception.code, "invalid_target")
            self.assertFalse((Path(tmp) / "out-index").exists())

            for model, revision in (("", REV), ("../secret", REV), ("fixture/tiny", "A" * 40),
                                    ("fixture/tiny", "../" + "a" * 37), ("fixture/tiny", "ab")):
                with self.assertRaises(WeightlessError) as error:
                    stage_model(root, Path(tmp) / "unused", model=model, revision=revision)
                self.assertEqual(error.exception.code, "invalid_target")
                self.assertFalse((Path(tmp) / "unused").exists())

    @unittest.skipUnless(SMOLLM.is_dir(), "cached SmolLM snapshot is not present")
    def test_real_smollm_cache_is_dereferenced_into_a_regular_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "staged"
            staged = stage_model(SMOLLM, destination, model="HuggingFaceTB/SmolLM2-135M-Instruct", revision=PINNED)
            copied = destination / "snapshots" / PINNED
            source = SMOLLM / "snapshots" / PINNED / "config.json"
            self.assertEqual(staged["kind"], "huggingface")
            self.assertEqual(staged["revision"], PINNED)
            self.assertFalse((copied / "model.safetensors").is_symlink())
            self.assertEqual((copied / "model.safetensors").stat().st_size, (SMOLLM / "snapshots" / PINNED / "model.safetensors").stat().st_size)
            self.assertEqual((copied / "config.json").read_bytes(), source.read_bytes())
            self.assertTrue((SMOLLM / "snapshots" / PINNED / "model.safetensors").is_symlink())


class RunnerTests(unittest.TestCase):
    def runner(self):
        with patch("wsm.sandbox.subprocess.run") as inspect:
            inspect.return_value.stdout = "sha256:" + "a" * 64
            return DockerRunner("trusted")

    def tree(self, root):
        snap = snapshot(root)
        config(snap)
        weights(snap)
        return root

    def test_valid_target_is_mounted_and_host_path_is_not_sent_to_the_worker(self):
        runner = self.runner()
        with tempfile.TemporaryDirectory() as tmp:
            root = self.tree(Path(tmp) / "repo").resolve()
            captured = []

            def capture(command, payload, timeout=None, on_dispatch=None):
                captured.append((command, json.loads(payload)))
                return b'{"result":{"loaded":true}}'

            target = {"kind": "huggingface", "source": str(root), "revision": REV,
                      "model": "fixture/tiny", "dtype": "float32"}
            original = json.loads(json.dumps(target))
            with patch.object(runner, "_run", side_effect=capture):
                result = runner.execute("describe", target)
            self.assertEqual(result, {"loaded": True})
            self.assertEqual(target, original)
            command, payload = captured[0]
            mount = command[command.index("--mount") + 1]
            self.assertIn("src=" + str(root) + ",dst=/target,readonly", mount)
            self.assertEqual(payload["target"]["source"], "/target")
            self.assertEqual(payload["target"]["model"], "fixture/tiny")
            self.assertEqual(payload["target"]["revision"], REV)
            self.assertNotIn(str(root), json.dumps(payload))

    def test_malformed_generic_target_and_unpinned_smollm_do_not_mount(self):
        runner = self.runner()
        with tempfile.TemporaryDirectory() as tmp:
            root = str(self.tree(Path(tmp) / "repo").resolve())
            comma = Path(tmp) / "has,comma"
            shutil.copytree(root, comma)
            rejects = [
                {"kind": "huggingface", "source": root, "revision": "main", "model": "fixture/tiny", "dtype": "float32"},
                {"kind": "huggingface", "source": "relative", "revision": REV, "model": "fixture/tiny", "dtype": "float32"},
                {"kind": "huggingface", "source": str(comma), "revision": REV, "model": "fixture/tiny", "dtype": "float32"},
                {"kind": "smollm", "source": root, "revision": REV, "model": "HuggingFaceTB/SmolLM2-135M-Instruct", "dtype": "float32"},
                {"kind": "other", "source": root},
            ]
            for target in rejects:
                with patch.object(runner, "_run") as run_mock:
                    with self.assertRaises(WeightlessError) as error:
                        runner.execute("describe", target)
                run_mock.assert_not_called()
                self.assertEqual(error.exception.code, "invalid_target")


class WorkerBoundaryTests(unittest.TestCase):
    def test_unknown_kind_is_rejected_without_loading_a_model(self):
        with self.assertRaises(WeightlessError) as error:
            run(request({"kind": "mystery", "source": "/tmp", "revision": REV, "model": "fixture/tiny", "dtype": "float32"}))
        self.assertEqual(error.exception.code, "unsupported_target")
        self.assertNotIn("dependencies", str(error.exception))
        completed = subprocess.run([sys.executable, "-m", "wsm.target_worker"],
                                   input=json.dumps(request({"kind": "mystery"})), text=True, capture_output=True, check=True)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["error"]["code"], "unsupported_target")

    def test_worker_rejects_a_snapshot_link_that_leaves_the_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            outside = Path(tmp) / "outside"
            outside.mkdir()
            config(outside)
            weights(outside)
            root = Path(tmp) / "repo"
            (root / "snapshots").mkdir(parents=True)
            (root / "snapshots" / REV).symlink_to(outside)
            target = {"kind": "huggingface", "source": str(root.resolve()), "revision": REV,
                      "model": "fixture/tiny", "dtype": "float32"}
            with self.assertRaises(WeightlessError) as error:
                run(request(target))
            self.assertEqual(error.exception.code, "invalid_target")


class IdentityContractTests(unittest.TestCase):
    def test_smollm_digest_bytes_stay_on_the_original_domain(self):
        target = {"model": "org/model", "revision": REV, "dtype": "float32"}
        aliases = {"lm_head.weight": "transformer.wte.weight"}
        parameters = {"transformer.wte.weight": {"shape": [4, 2], "dtype": "float32", "size": 8}}
        sidecars = {"config.json": b'{"model_type":"gpt2"}', "tokenizer_config.json": b"{}"}
        manifest = manifest_digest("smollm", target, "2.4.0", "4.44.0", aliases, parameters, sidecars)
        self.assertEqual(manifest.hex(), "cbb06fdf646f6e7cf4ce9d0f0e02abf9650d457be153a5a2f4bf7cd9d9ae6eeb")
        identifier = state_id("smollm", manifest, [("transformer.wte.weight", b"\x00\x00\x80\x3f" * 8)], [])
        self.assertEqual(identifier, "sha256:bd75ad807fadf1274dc9e311a93fa8d77a716baaa13509f16e15e2be230f2bc6")
        changed = dict(target)
        changed["model"] = "org/other"
        self.assertNotEqual(manifest_digest("huggingface", target, "2.4.0", "4.44.0", aliases, parameters, sidecars), manifest)
        self.assertNotEqual(manifest_digest("smollm", changed, "2.4.0", "4.44.0", aliases, parameters, sidecars), manifest)
        self.assertNotEqual(manifest_digest("smollm", target, "2.4.1", "4.44.0", aliases, parameters, sidecars), manifest)


def _loss(model, logits_model, prompt_ids, label_ids):
    import torch
    ids = torch.tensor([prompt_ids + label_ids], dtype=torch.long)
    logits = logits_model(input_ids=ids).logits[0, len(prompt_ids) - 1:-1, :]
    return float(torch.nn.functional.cross_entropy(logits, torch.tensor(label_ids)).item())


class GenericCausalLMTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
            from tokenizers import Tokenizer
            from tokenizers.models import WordLevel
            from tokenizers.pre_tokenizers import Whitespace
        except ImportError:
            self.skipTest("torch and transformers are required")
        self.torch = torch
        self.AutoModelForCausalLM = AutoModelForCausalLM
        self.AutoTokenizer = AutoTokenizer
        self.GPT2Config = GPT2Config
        self.GPT2LMHeadModel = GPT2LMHeadModel
        self.PreTrainedTokenizerFast = PreTrainedTokenizerFast
        self.Tokenizer = Tokenizer
        self.WordLevel = WordLevel
        self.Whitespace = Whitespace
        torch.set_num_threads(1)

    def save_tiny(self, root, *, chat_template=None, eos_token=None):
        backend = self.Tokenizer(self.WordLevel({"<unk>": 0, "a": 1, "b": 2, "c": 3}, unk_token="<unk>"))
        backend.pre_tokenizer = self.Whitespace()
        kwargs = {"tokenizer_object": backend, "unk_token": "<unk>"}
        if eos_token:
            kwargs["eos_token"] = eos_token
        tokenizer = self.PreTrainedTokenizerFast(**kwargs)
        if chat_template:
            tokenizer.chat_template = chat_template
        config = self.GPT2Config(vocab_size=4, n_positions=32, n_embd=16, n_layer=1, n_head=2, n_inner=32,
                                 bos_token_id=None, eos_token_id=None, pad_token_id=None)
        model = self.GPT2LMHeadModel(config)
        raw = Path(root) / "raw"
        model.save_pretrained(raw, safe_serialization=True)
        tokenizer.save_pretrained(raw)
        snap = snapshot(root)
        for path in raw.iterdir():
            if path.is_file():
                shutil.copyfile(path, snap / path.name)
        (snap / "notes.md").write_text("beyond-allowlist", encoding="utf-8")
        nested = snap / "additional_chat_templates"
        nested.mkdir()
        (nested / "extra.jinja").write_text("not the default template", encoding="utf-8")
        return snap

    def independent_smollm_id(self, snap, target):
        import transformers
        model = self.AutoModelForCausalLM.from_pretrained(snap, local_files_only=True, trust_remote_code=False,
                                                          dtype=self.torch.float32, attn_implementation="eager").cpu().eval()
        tensors, aliases, storage = {}, {}, {}
        pointers = set()
        for name, tensor in model.named_parameters(remove_duplicate=False):
            key = (tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()))
            if key in storage:
                aliases[name] = storage[key]
            else:
                pointers.add(tensor.untyped_storage().data_ptr())
                storage[key] = name
                tensors[name] = tensor
        parameters = {name: {"shape": list(tensor.shape), "dtype": "float32", "size": tensor.numel()}
                      for name, tensor in tensors.items()}
        digest = hashlib.sha256(b"weightless.smollm.manifest.v1\0")
        digest.update(json.dumps({"model": target["model"], "revision": target["revision"], "dtype": target["dtype"],
                                  "torch": self.torch.__version__, "transformers": transformers.__version__,
                                  "aliases": aliases, "parameters": parameters}, sort_keys=True).encode())
        for name in sorted(os.listdir(snap)):
            if name.endswith((".json", ".txt", ".model", ".jinja")):
                digest.update(name.encode() + b"\0")
                with open(os.path.join(snap, name), "rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
        manifest = digest.digest()
        digest = hashlib.sha256(b"weightless.smollm.state.v1\0" + manifest)
        for name, tensor in sorted(tensors.items()):
            digest.update(name.encode() + b"\0")
            digest.update(tensor.detach().numpy().tobytes(order="C"))
        for name, tensor in sorted(model.named_buffers(remove_duplicate=False)):
            digest.update(name.encode() + b"\0")
            digest.update(str(tensor.dtype).encode() + b"\0")
            digest.update(json.dumps(list(tensor.shape)).encode() + b"\0")
            digest.update(tensor.detach().contiguous().numpy().tobytes(order="C"))
        return "sha256:" + digest.hexdigest()

    def test_base_model_uses_plain_prompt_ids_and_does_not_invent_eos(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            self.save_tiny(root)
            destination = Path(tmp) / "staged"
            staged = stage_model(root, destination, model="fixture/tiny-base", revision=REV)
            described = run(request(staged))
            self.assertEqual(described["aliases"].get("lm_head.weight"), "transformer.wte.weight")
            tokenizer = self.AutoTokenizer.from_pretrained(destination / "snapshots" / REV, local_files_only=True, trust_remote_code=False)
            model = self.AutoModelForCausalLM.from_pretrained(destination / "snapshots" / REV, local_files_only=True,
                                                              trust_remote_code=False, dtype=self.torch.float32,
                                                              attn_implementation="eager").cpu().eval()
            self.assertIsNone(tokenizer.eos_token_id)
            prompt_ids = tokenizer.encode("b c", add_special_tokens=False)
            label_ids = tokenizer.encode("a", add_special_tokens=False)
            expected = _loss(self.torch, model, prompt_ids, label_ids)
            invented = _loss(self.torch, model, prompt_ids, label_ids + [0])
            self.assertNotEqual(expected, invented)
            result = run(request(staged, action="evaluate", generate=True,
                                 cases=[{"id": "one", "prompt": "b c", "target": "a"}]))
            row = result["cases"][0]
            self.assertEqual(row["target_nll"], expected)
            self.assertFalse(row["stopped_on_eos"])
            generated = model.generate(self.torch.tensor([prompt_ids]), max_new_tokens=16, do_sample=False)[0, len(prompt_ids):].tolist()
            self.assertEqual(row["prediction"], tokenizer.decode(generated, skip_special_tokens=True))
            canonical = "transformer.wte.weight"
            current = run(request(staged, action="inspect", query={"tensor": "lm_head.weight", "indices": [0]}))
            old = current["values"][0]["value"]
            edited = run(request(staged, action="materialize", edits=[{"tensor": "lm_head.weight", "index": 0, "value": old + 1}]))
            self.assertEqual(edited["target_id"], described["target_id"])
            self.assertNotEqual(edited["checkpoint"], described["checkpoint"])
            seen = run(request(staged, action="inspect", patches=edited["patches"], query={"tensor": canonical, "indices": [0]}))
            self.assertNotEqual(seen["values"][0]["value"], old)
            restored = run(request(staged, action="materialize", patches=edited["patches"],
                                   edits=[{"tensor": canonical, "index": 0, "value": old}]))
            self.assertEqual(restored["checkpoint"], described["checkpoint"])
            self.assertEqual(restored["patches"], [])

    def test_chat_template_loss_includes_eos_and_differs_from_plain_encoding(self):
        template = "{% for message in messages %}{{ message['content'] }}{% endfor %}{% if add_generation_prompt %} c{% endif %}"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            self.save_tiny(root, chat_template=template, eos_token="a")
            destination = Path(tmp) / "staged"
            staged = stage_model(root, destination, model="fixture/tiny-chat", revision=REV)
            tokenizer = self.AutoTokenizer.from_pretrained(destination / "snapshots" / REV, local_files_only=True, trust_remote_code=False)
            model = self.AutoModelForCausalLM.from_pretrained(destination / "snapshots" / REV, local_files_only=True,
                                                              trust_remote_code=False, dtype=self.torch.float32,
                                                              attn_implementation="eager").cpu().eval()
            prompt_ids = tokenizer.apply_chat_template([{"role": "user", "content": "b"}], tokenize=True,
                                                       add_generation_prompt=True, return_dict=False)
            label_ids = tokenizer.encode("c", add_special_tokens=False) + [tokenizer.eos_token_id]
            plain_ids = tokenizer.encode("b", add_special_tokens=False)
            expected = _loss(self.torch, model, prompt_ids, label_ids)
            plain = _loss(self.torch, model, plain_ids, tokenizer.encode("c", add_special_tokens=False))
            self.assertNotEqual(prompt_ids, plain_ids)
            self.assertNotEqual(expected, plain)
            result = run(request(staged, action="evaluate", cases=[{"id": "chat", "prompt": "b", "target": "c"}]))
            self.assertEqual(result["cases"][0]["target_nll"], expected)

    def test_generation_uses_the_same_eos_as_the_tokenizer_and_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            snap = self.save_tiny(root, eos_token="b")
            model = self.AutoModelForCausalLM.from_pretrained(snap, local_files_only=True)
            with self.torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
                model.transformer.ln_f.bias.fill_(1)
                model.transformer.wte.weight[1].fill_(1)
            model.generation_config.eos_token_id = 1
            model.save_pretrained(snap)
            target = stage_model(root, Path(tmp) / "staged", model="fixture/eos-mismatch", revision=REV)
            result = run(request(target, action="evaluate", generate=True, query={"max_new_tokens": 4},
                                 cases=[{"id": "eos", "prompt": "c", "target": "a"}]))
            self.assertEqual(result["cases"][0]["prediction"], "a a a a")
            self.assertFalse(result["cases"][0]["stopped_on_eos"])

    def test_saved_smollm_identity_ignores_non_allowlisted_sidecars(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            snap = self.save_tiny(root)
            target = {"kind": "smollm", "source": str(root.resolve()), "revision": REV,
                      "model": "fixture/tiny", "dtype": "float32"}
            described = run(request(target))
            self.assertEqual(described["target_id"], self.independent_smollm_id(snap, target))
            generic = {"kind": "huggingface", "source": str(root.resolve()), "revision": REV,
                       "model": "fixture/tiny", "dtype": "float32"}
            first = run(request(generic))["target_id"]
            self.assertNotEqual(first, described["target_id"])
            notes = snap / "notes.md"
            notes.chmod(0o644)
            notes.write_text("changed sidecar", encoding="utf-8")
            self.assertEqual(run(request(target))["target_id"], described["target_id"])
            self.assertNotEqual(run(request(generic))["target_id"], first)


if __name__ == "__main__":
    unittest.main()
