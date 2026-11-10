"""Prove direct Python tensor editing and source-free checkpoint reload on Modal."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fingerprints(model):
    return {name: hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, tensor in model.model.state_dict().items()}


def file_fingerprints(directory):
    result = {}
    for path in sorted(directory.rglob('*')):
        if path.is_file():
            with path.open('rb') as handle:
                result[str(path.relative_to(directory))] = hashlib.file_digest(handle, 'sha256').hexdigest()
    return result


def worker():
    import torch
    from wsm import WeightModel, execute_program

    torch.use_deterministic_algorithms(True)
    source = Path('/target/snapshots') / REVISION
    source_files = file_fingerprints(source)
    cases = json.loads(Path('/examples/cases.json').read_text())
    control = WeightModel.load(source, device='cuda')
    before = fingerprints(control)
    norm = control.weights['model.norm.weight'].clone()
    baseline = control.evaluate(cases)
    result = execute_program('/examples/norm_scale.py', model_path=source,
                             output='/tmp/edit-run', device='cuda', cases=cases)
    assert result['state'] == 'published', result
    changed = WeightModel.load('/tmp/edit-run/checkpoint', device='cuda')
    after = fingerprints(changed)
    names = sorted(name for name in before if before[name] != after[name])
    assert names == ['model.norm.weight'], names
    expected = norm.clone()
    expected[:16].mul_(0.95)
    assert torch.equal(changed.weights['model.norm.weight'], expected)
    assert fingerprints(control) == before
    assert control.evaluate(cases) == baseline
    candidate = changed.evaluate(cases)
    assert abs(candidate['loss'] - baseline['loss']) > 1e-8
    source_reload = WeightModel.load(source, device='cuda')
    assert fingerprints(source_reload) == before
    assert file_fingerprints(source) == source_files
    proof = {'model': MODEL, 'revision': REVISION, 'device': str(next(changed.model.parameters()).device),
             'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
             'changed_tensors': names, 'changed_elements': int((norm != expected).sum().item()),
             'edited_norm': expected.cpu().tolist(),
             'baseline': baseline, 'candidate': candidate,
             'unchanged_control_verified': True, 'source_unchanged_verified': True,
             'weights': after, 'run': result,
             'source_files': source_files,
             'scope': 'Deterministic SDK mechanics check. No capability improvement claim.'}
    import shutil
    shutil.copytree('/tmp/edit-run', '/proof/run')
    write(Path('/proof/edit-proof.json'), proof)
    return proof


def reload_worker():
    import torch
    from wsm import WeightModel

    assert not Path('/target').exists(), 'Independent reload image contains source model'
    assert not Path('/root/.cache/huggingface').exists(), 'Independent reload has an HF cache'
    proof = json.loads(Path('/proof/edit-proof.json').read_text())
    model = WeightModel.load('/proof/run/checkpoint', device='cpu')
    assert fingerprints(model) == proof['weights']
    norm = model.weights['model.norm.weight']
    assert torch.equal(norm, torch.tensor(proof['edited_norm'], dtype=norm.dtype))
    cases = json.loads(Path('/examples/cases.json').read_text())
    evaluation = model.evaluate(cases)
    assert abs(evaluation['loss'] - proof['candidate']['loss']) < 1e-4
    assert model.weights['lm_head.weight'].data_ptr() == model.weights['model.embed_tokens.weight'].data_ptr()
    return {'source_absent': True, 'cache_absent': True, 'all_tensors_exact': True,
            'tied_embeddings': True, 'evaluation': evaluation, 'device': 'cpu'}


def launch(args):
    import modal
    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository / 'packages/python/src'))
    from wsm import stage_smollm

    args.output.mkdir(parents=True, exist_ok=False)
    stage_smollm(args.model_source, args.output / 'source')
    base = (modal.Image.debian_slim(python_version='3.12')
            .uv_pip_install('torch==2.10.0+cu128', index_url='https://download.pytorch.org/whl/cu128')
            .uv_pip_install('transformers==5.22.2', 'safetensors==0.8.0')
            .env({'PYTHONPATH': '/opt/wsm', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                  'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'TOKENIZERS_PARALLELISM': 'false'})
            .add_local_dir(repository / 'packages/python/src/wsm', '/opt/wsm/wsm',
                           ignore=['**/__pycache__/**', '**/*.pyc'])
            .add_local_dir(repository / 'examples/coding', '/examples')
            .add_local_file(Path(__file__), '/coding_modal.py'))
    editing_image = base.add_local_dir(args.output / 'source', '/target')
    app = modal.App('wsm-coding-proof')
    manifest = {'model': MODEL, 'revision': REVISION, 'modal': modal.__version__,
                'sdk_files': file_fingerprints(repository / 'packages/python/src/wsm'),
                'program_sha256': hashlib.sha256((repository / 'examples/coding/norm_scale.py').read_bytes()).hexdigest(),
                'proof_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'invocation': {'edit_calls': 1, 'reload_calls': 1, 'gpu': 'T4',
                               'timeout_seconds_per_call': 600, 'retries': 0},
                'spending_policy': 'Open. These invocation bounds are not a dollar spending cap.'}
    write(args.output / 'launch.json', manifest)
    with modal.Volume.ephemeral() as volume:
        @app.function(image=editing_image, serialized=True, gpu='T4', cpu=2, memory=8192, timeout=600,
                      retries=0, max_containers=1, volumes={'/proof': volume})
        def edit_remote():
            completed = subprocess.run(['python', '/coding_modal.py', '--worker'],
                                       capture_output=True, text=True, timeout=570)
            if completed.returncode:
                raise RuntimeError(completed.stderr[-8000:])
            volume.commit()
            return json.loads(completed.stdout)

        @app.function(image=base, serialized=True, cpu=2, memory=4096, timeout=600, retries=0,
                      max_containers=1, volumes={'/proof': volume.with_mount_options(read_only=True)})
        def reload_remote():
            completed = subprocess.run(['python', '/coding_modal.py', '--reload-worker'],
                                       capture_output=True, text=True, timeout=570)
            if completed.returncode:
                raise RuntimeError(completed.stderr[-8000:])
            return json.loads(completed.stdout)

        try:
            with modal.enable_output(), app.run():
                manifest['app_id'] = app.app_id
                write(args.output / 'launch.json', manifest)
                edit_proof = edit_remote.remote()
                write(args.output / 'edit-proof.json', edit_proof)
                reload_proof = reload_remote.remote()
                write(args.output / 'reload-proof.json', reload_proof)
                checkpoint = args.output / 'checkpoint'
                checkpoint.mkdir()
                from modal.volume import FileEntryType
                for entry in volume.listdir('/run/checkpoint', recursive=True):
                    if entry.type != FileEntryType.FILE:
                        continue
                    relative = Path('/' + entry.path.lstrip('/')).relative_to('/run/checkpoint')
                    destination = checkpoint / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open('xb') as handle:
                        for chunk in volume.read_file(entry.path):
                            handle.write(chunk)
                write(args.output / 'summary.json', {'passed': True, 'edit': edit_proof,
                                                    'independent_reload': reload_proof})
        except BaseException as error:
            write(args.output / 'failure.json', {'type': type(error).__name__, 'message': str(error)[-8000:]})
            raise
    write(args.output / 'cleanup.json', {'app_context_closed': True, 'ephemeral_volume_context_closed': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--reload-worker', action='store_true')
    parser.add_argument('--model-source', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(), allow_nan=False))
    elif args.reload_worker:
        print(json.dumps(reload_worker(), allow_nan=False))
    elif args.model_source is None or args.output is None:
        parser.error('--model-source and --output are required')
    else:
        launch(args)
