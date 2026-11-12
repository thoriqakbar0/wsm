"""Create a synthetic local benchmark configuration without downloading models."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

from wsm.benchmark import FORMAT, digest, tree_digest


def create(directory):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    root = Path(directory).resolve()
    root.mkdir(mode=0o700)
    source = root / "model"
    torch.manual_seed(7)
    backend = Tokenizer(WordLevel({"<unk>": 0, "a": 1, "b": 2, "c": 3, "d": 4}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", eos_token="d", pad_token="d")
    GPT2LMHeadModel(GPT2Config(vocab_size=5, n_positions=32, n_embd=16, n_layer=1, n_head=2,
                             bos_token_id=4, eos_token_id=4, pad_token_id=4)).save_pretrained(source)
    tokenizer.save_pretrained(source)

    def save(name, value):
        path = root / name
        path.write_text(json.dumps(value, indent=2) + "\n")
        return {"path": str(path), "sha256": digest(path)}

    public = save("public.json", [{"prompt": "a", "target": "b"}])
    private = save("private.json", [{"prompt": "a b", "target": "c"}, {"prompt": "b", "target": "c d"}])
    regression = save("regression.json", [{"prompt": "c", "target": "d"}])
    script = root / "script.py"
    harmful = """def edit(model):
    model.weights['transformer.ln_f.weight'].zero_()
    model.weights['transformer.ln_f.bias'].fill_(1)
    model.weights['lm_head.weight'].zero_()
    model.weights['lm_head.weight'][0].fill_(100)
"""
    script.write_text(harmful)
    undo = ("def edit(model):\n    from wsm import WeightModel\n    original = WeightModel.load(" + repr(str(source))
            + ")\n    for name in model.weights:\n        model.weights[name].copy_(original.weights[name])\n")
    actions = save("actions.json", [{"action": "python", "source": harmful},
                                    {"action": "python", "source": undo}, {"action": "finish"}])
    lock = root / "dependencies.txt"
    lock.write_text("\n".join(sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())) + "\n")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    config = {"format": FORMAT, "name": "synthetic-mechanics", "repetitions": 1,
              "model": {"path": str(source), "id": "synthetic-tiny-gpt2", "revision": "fixture-v1", "sha256": tree_digest(source)},
              "provenance": {"code_revision": git.stdout.strip(), "dependency_lock": {"path": str(lock), "sha256": digest(lock)}},
              "runtime": {"backend": "local_trusted", "grader_python": sys.executable, "actions": actions},
              "limits": {"turns": 3, "evaluations": 1, "output_tokens": 4096, "wall_seconds": 180,
                         "command_seconds": 60, "grader_seconds": 60},
              "budget": {"max_usd": 0, "prior_reserved_usd": 0, "price_checked_on": None,
                         "rates": {"build_usd": 0, "sandbox_hourly_usd": 0, "egress_per_gib": 0, "teacher_request_usd": 0}},
              "statistics": {"seed": 7, "bootstrap_samples": 1000},
              "thresholds": None,
              "review": {"data_rights": False, "semantic_overlap": False, "power": False,
                         "owner": None, "private_artifact_store": None},
              "regression": regression,
              "tasks": [{"id": "completion", "family": "synthetic", "instruction": "Exercise edit and explicit undo",
                         "public": public, "private": private, "script": {"path": str(script), "sha256": digest(script)}}]}
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    return config_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(create(args.output))
