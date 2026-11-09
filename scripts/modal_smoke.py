"""Run one trusted, offline SmolLM2 CUDA lifecycle in an ephemeral Modal Sandbox."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import tempfile
import time


REVISION = '12fd25f77366fa6b3b4b768ec3050bf629380bac'
LIMITS = {'gpu': 'T4', 'cpu': [1, 2], 'memory_mib': [2048, 4096],
          'sandbox_seconds': 600, 'worker_seconds': 540, 'generation_tokens': 4,
          'sandbox_creations': 1, 'retries': 0}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def hashes(directory):
    result = {}
    for path in sorted(directory.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            with path.open('rb') as stream:
                result[str(path.relative_to(directory))] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return result


def worker():
    started = time.monotonic()
    signal.alarm(LIMITS['worker_seconds'])
    import torch
    import transformers
    import wsm
    from wsm import Session, Verifier, replay_session, verify_session
    from wsm.target_worker import run

    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    source = Path(wsm.__file__).parent
    model = Path('/target/snapshots') / REVISION
    source_hashes, model_hashes = hashes(source), hashes(model)
    runtime = {'torch': torch.__version__, 'transformers': transformers.__version__,
               'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
               'device': 'cuda:0', 'deterministic_algorithms': True, 'tf32': False,
               'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'source_hashes': source_hashes, 'model_hashes': model_hashes}
    target = {'kind': 'smollm', 'source': '/target', 'revision': REVISION,
              'model': 'HuggingFaceTB/SmolLM2-135M-Instruct', 'dtype': 'float32'}

    class TrustedRunner:
        config = {'mode': 'trusted-modal-smoke', 'runtime': runtime, 'limits': LIMITS,
                  'assurance': 'Trusted orchestration; no untrusted-agent isolation claim.'}

        def __init__(self):
            self.receipts = []

        def execute(self, action, target, *, patches=None, edits=None, cases=None,
                    query=None, generate=False, purpose='development', timeout=None, on_dispatch=None,
                    forward_limit=None):
            if timeout is not None and timeout <= 0:
                raise TimeoutError('Operation deadline exhausted')
            if on_dispatch:
                on_dispatch()
            torch.cuda.synchronize()
            before = time.monotonic()
            request = {'action': action, 'target': target, 'patches': patches or [],
                       'edits': edits or [], 'cases': cases,
                       'query': query or {'max_new_tokens': LIMITS['generation_tokens']},
                       'generate': generate, 'purpose': purpose}
            if forward_limit is not None:
                request['forward_limit'] = forward_limit
            result = run(request, device='cuda')
            torch.cuda.synchronize()
            execution = result.get('execution')
            if execution:
                assert execution['parameter_devices'] == ['cuda:0']
                for placement in execution.get('cases', []):
                    assert placement['finite_logits']
                    assert all(placement[key] == 'cuda:0' for key in ('input_device', 'label_device', 'logits_device'))
                    if generate:
                        assert placement['generation_input_device'] == 'cuda:0'
            if action == 'evaluate':
                assert execution and math.isfinite(result['metrics']['loss'])
                assert all(math.isfinite(row['target_nll']) for row in result['cases'])
            self.receipts.append({'action': action, 'purpose': purpose, 'checkpoint': result.get('checkpoint'),
                                  'execution': execution, 'seconds': time.monotonic() - before})
            return result

    runner = TrustedRunner()
    descriptor = runner.execute('describe', target)
    assert descriptor['aliases']['lm_head.weight'] == 'model.embed_tokens.weight'
    public = [{'id': 'public-hello', 'prompt': 'Say hello.', 'target': 'Hello!'}]
    private = {'task': [{'id': 'final-hello', 'prompt': 'Please greet me.', 'target': 'Hello!'}]}
    declaration = {'hypothesis': 'A finite scalar edit can be evaluated and selected on a CUDA target.',
                   'arms': ['scripted'], 'editable': ['model.norm.weight'], 'wall_seconds': 450,
                   'budgets': {'attempts': 2, 'evaluations': 3, 'inspections': 1,
                               'teacher_requests': 0, 'final_evaluations': 1},
                   'max_edits': 1, 'max_inspect': 1, 'claim': 'exploratory'}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        verifier = Verifier(root / 'private.db', runner=runner, datasets={'final': private})
        session = Session.create(root / 'session.db', runner=runner, target=target,
                                 declaration=declaration, development=public,
                                 verifier=verifier, final_dataset='final')
        try:
            view = session.observe('scripted')
            read = session.inspect({'checkpoint': view['base'], 'tensor': 'model.norm.weight', 'indices': [0]},
                                   key='inspect', arm='scripted')
            original = read['result']['values'][0]['value']
            proposal = {key: view[key] for key in ('session', 'arm', 'declaration', 'observation', 'base')}
            proposal['edits'] = [{'tensor': 'model.norm.weight', 'index': 0, 'value': original + .125}]
            edited = session.submit(proposal, key='edit')
            assert edited['status'] == 'committed', edited
            result = edited['result']
            assert result['checkpoint'] != edited['base']
            expected = result['metrics']['loss'] < result['baseline']['metrics']['loss']
            assert edited['selection']['accepted'] == expected
            selected_loss = result['metrics']['loss'] if expected else result['baseline']['metrics']['loss']
            selected = session.observe('scripted')
            selected_value = original + .125 if expected else original
            same = {key: selected[key] for key in ('session', 'arm', 'declaration', 'observation', 'base')}
            same['edits'] = [{'tensor': 'model.norm.weight', 'index': 0, 'value': selected_value}]
            tie = session.submit(same, key='same-state')
            assert tie['status'] == 'committed' and not tie['selection']['accepted']
            assert tie['result']['checkpoint'] == selected['base']
            assert tie['result']['metrics']['loss'] == selected_loss
            frozen = session.freeze(key='freeze')
            final = session.finalize(key='final')
            assert frozen['complete'] and final['status'] == 'completed'
            bundle = session.export()
            assert replay_session(bundle) == frozen
            reconstruction = verify_session(bundle, runner=runner, target=target)
            retained = frozen['arms']['scripted']
            reevaluated = runner.execute('evaluate', target, patches=retained['patches'], cases=public, generate=True)
            assert reevaluated['checkpoint'] == retained['checkpoint']
            assert math.isclose(reevaluated['metrics']['loss'], retained['loss'], rel_tol=0, abs_tol=1e-6)
            assert runner.execute('describe', target)['checkpoint'] == descriptor['checkpoint']
            assert hashes(model) == model_hashes
        finally:
            session.close()
            verifier.close()
    torch.cuda.synchronize()
    assert bundle['used']['teacher_requests'] == 0
    return {'passed': True, 'runtime': runtime, 'limits': LIMITS, 'teacher_requests': 0,
            'selection_rule_verified': True, 'tie_rejected': True, 'tied_embeddings': True,
            'source_unchanged': True, 'bundle': bundle, 'reconstruction': reconstruction,
            'selected_reevaluation': reevaluated, 'receipts': runner.receipts,
            'seconds': time.monotonic() - started,
            'peak_vram_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_vram_reserved_bytes': torch.cuda.max_memory_reserved(),
            'scope': 'One trusted CUDA mechanics smoke; no efficacy, autonomy, or untrusted-agent isolation claim.'}


def launch(args):
    import modal
    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository / 'packages/python/src'))
    from wsm import stage_smollm

    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    stage_smollm(args.model_source, args.output / 'model')
    image = (modal.Image.debian_slim(python_version='3.12')
             .uv_pip_install('torch==2.10.0+cu128', index_url='https://download.pytorch.org/whl/cu128')
             .uv_pip_install('transformers==5.22.2', 'safetensors==0.8.0')
             .env({'PYTHONPATH': '/opt/wsm', 'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1',
                   'TRANSFORMERS_OFFLINE': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
                   'TOKENIZERS_PARALLELISM': 'false'})
             .add_local_dir(repository / 'packages/python/src/wsm', '/opt/wsm/wsm',
                            ignore=['**/__pycache__/**', '**/*.pyc'])
             .add_local_file(Path(__file__), '/smoke.py')
             .add_local_dir(args.output / 'model', '/target'))
    app = modal.App('wsm-gpu-smoke-single')
    sandbox = None
    metadata = {'limits': LIMITS, 'modal_version': modal.__version__,
                'source_hashes': hashes(repository / 'packages/python/src/wsm'),
                'assurance': 'Dollar estimate is not an account-level enforced budget.'}
    write(args.output / 'launch.json', metadata)
    try:
        with modal.enable_output(), app.run():
            metadata['app_id'] = app.app_id
            write(args.output / 'launch.json', metadata)
            try:
                sandbox = modal.Sandbox.create('python', '/smoke.py', '--worker', app=app, image=image,
                                               gpu=LIMITS['gpu'], cpu=tuple(LIMITS['cpu']),
                                               memory=tuple(LIMITS['memory_mib']), timeout=LIMITS['sandbox_seconds'],
                                               block_network=True)
                metadata['sandbox_id'] = sandbox.object_id
                metadata['image_id'] = image.object_id
                write(args.output / 'launch.json', metadata)
                sandbox.wait(raise_on_termination=False)
                stdout, stderr = sandbox.stdout.read(), sandbox.stderr.read()
                (args.output / 'worker.stdout').write_text(stdout)
                (args.output / 'worker.stderr').write_text(stderr)
                assert len(stdout.encode()) <= 4 * 1024**2, 'Worker result exceeds evidence limit'
                assert sandbox.returncode == 0, f'Worker exit code {sandbox.returncode}'
                result = json.loads(stdout)
                assert result['passed']
                write(args.output / 'result.json', result)
                write(args.output / 'bundle.json', result['bundle'])
            finally:
                if sandbox is not None:
                    code = sandbox.terminate(wait=True)
                    confirmed = sandbox.poll() is not None
                    write(args.output / 'cleanup.json', {'sandbox_id': sandbox.object_id,
                                                        'exit_code': code, 'termination_confirmed': confirmed})
                    assert confirmed, 'Sandbox termination unconfirmed'
    except BaseException as error:
        write(args.output / 'failure.json', {'type': type(error).__name__, 'message': str(error)[:1000]})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--model-source', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(), allow_nan=False))
    elif args.model_source is None or args.output is None:
        parser.error('--model-source and --output are required')
    else:
        launch(args)
