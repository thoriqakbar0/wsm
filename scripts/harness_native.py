"""Verify budgeted real-model edits and a standalone checkpoint in Docker."""
import argparse
import json
from pathlib import Path
import subprocess
import uuid

from wsm import DockerRunner, Session, Verifier, WeightlessError, stage_smollm


_EXPORT = '''
import json,os,sys
from wsm.checkpoint_export import export_checkpoint
request=json.load(sys.stdin)
os.chdir('/output')
os.environ['PYTHONPATH']='relative-import-path-that-does-not-exist'
receipt=export_checkpoint(request['bundle'],target=request['target'],arm='editor',
    destination='selected',max_bytes=request['max_bytes'])
print(json.dumps(receipt,allow_nan=False))
'''

_RELOAD = '''
import json,sys
from pathlib import Path
from wsm.target_worker import ModelTarget
request=json.load(sys.stdin)
assert not Path(request['original_source']).exists(), 'Original source is visible'
target={**request['target'],'source':'/artifact'}
model=ModelTarget(target)
assert model.identity()==request['checkpoint'], 'Exported numerical state differs'
assert model.aliases['lm_head.weight']=='model.embed_tokens.weight'
metrics,rows=model.evaluate(request['cases'],True,None,17*len(request['cases']))
assert metrics==request['expected']['metrics'], 'Reloaded metrics differ'
assert rows==request['expected']['cases'], 'Reloaded generations differ'
print(json.dumps({'checkpoint':model.identity(),'original_source_unavailable':True,
    'tied_embeddings':True,'metrics':metrics,'cases':rows,
    'forward_calls':model.forward_calls},allow_nan=False))
'''


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def proposal(session, arm, edits):
    view = session.observe(arm)
    return {k: view[k] for k in ('session', 'arm', 'declaration', 'observation', 'base')} | {'edits': edits}


def operator(image, code, request, mounts, *, file_limit):
    name = 'wsm-export-proof-' + uuid.uuid4().hex
    package = Path(__file__).resolve().parents[1] / 'packages/python/src'
    command = ['docker', 'run', '--rm', '-i', '--name', name, '--network', 'none',
               '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
               '--user', '65534:65534', '--cpus', '2', '--memory', '3g', '--pids-limit', '128',
               '--ulimit', f'fsize={file_limit}:{file_limit}',
               '--tmpfs', '/tmp:rw,noexec,nosuid,nodev,size=64m',
               '--mount', f'type=bind,src={package},dst=/code,readonly',
               '--env', 'PYTHONPATH=/code', '--env', 'PYTHONDONTWRITEBYTECODE=1']
    for source, destination, readonly in mounts:
        command += ['--mount', f'type=bind,src={source},dst={destination}' + (',readonly' if readonly else '')]
    command += [image, 'python', '-c', code]
    try:
        result = subprocess.run(command, input=json.dumps(request), capture_output=True,
                                text=True, timeout=180, check=True)
        return json.loads(result.stdout)
    finally:
        subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--model-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    target = stage_smollm(args.model_source, root / 'model')
    runner = DockerRunner(args.image, timeout=90)
    descriptor = runner.execute('describe', target)
    public = [{'id': 'duck-public', 'prompt': 'What sound does a duck make?', 'target': 'quack quack turnip'},
              {'id': 'cow-public', 'prompt': 'What sound does a cow make?', 'target': 'moo moo comet'}]
    retention = [{'id': 'public-language', 'prompt': 'Say hello politely.', 'target': 'Hello!'}]
    private = {'task': [{'id': 'unseen-task', 'prompt': 'Please imitate the noise of a duck.', 'target': 'quack quack turnip'}],
               'retention': [{'id': 'unseen-language', 'prompt': 'Greet a new friend.', 'target': 'Hello!'}]}
    declaration = {'hypothesis': 'A scripted scalar edit exercises export and budget mechanics on a real language model.',
                   'arms': ['editor', 'control'], 'editable': ['model.norm.weight'],
                   'max_edits': 1, 'max_inspect': 1, 'wall_seconds': 900,
                   'budgets': {'attempts': 3, 'evaluations': 12, 'inspections': 1,
                               'diagnostics': 1, 'teacher_requests': 0, 'final_evaluations': 4,
                               'forward_calls': 500}, 'final_forward_calls': 96}
    verifier = Verifier(root / 'private.db', runner=runner, datasets={'heldout': private})
    session = Session.create(root / 'session.db', runner=runner, target=target,
                             declaration=declaration, development=public,
                             development_retention=retention, verifier=verifier, final_dataset='heldout')
    try:
        view = session.observe('editor')
        read = session.inspect({'checkpoint': view['base'], 'tensor': 'model.norm.weight', 'indices': [0]},
                               key='inspect', arm='editor')
        original = read['result']['values'][0]['value']
        control = session.submit(proposal(session, 'control', [{'tensor': 'model.norm.weight', 'index': 0, 'value': original}]),
                                 key='control')
        assert not control['selection']['accepted']
        edited = session.submit(proposal(session, 'editor', [{'tensor': 'model.norm.weight', 'index': 0, 'value': original - .125}]),
                                key='edit-negative')
        if not edited['selection']['accepted']:
            edited = session.submit(proposal(session, 'editor', [{'tensor': 'model.norm.weight', 'index': 0, 'value': original + .125}]),
                                    key='edit-positive')
        assert edited['selection']['accepted'], 'Neither scripted edit improved public loss; no edited artifact proof'
        diagnosis = session.diagnose(key='diagnose', arm='editor')
        assert diagnosis['status'] == 'completed'
        frozen = session.freeze(key='freeze')
        assert frozen['complete'] and frozen['arms']['editor']['patches']
        final = session.finalize(key='final')
        assert final['status'] == 'completed'
        bundle = session.export()
        assert all(case['id'] not in json.dumps(bundle) for cases in private.values() for case in cases)
        assert 'usage' not in json.dumps(final)
        write(root / 'evidence.json', bundle)
        expected = diagnosis['result']
        selected = frozen['arms']['editor']['checkpoint']
        assert expected['checkpoint'] == selected
        assert runner.execute('describe', target)['checkpoint'] == descriptor['checkpoint']
        tiny = runner.execute('evaluate', target, cases=retention, generate=True,
                              query={'max_new_tokens': 1}, forward_limit=2)
        assert tiny['usage']['forward_calls'] == 2
        try:
            runner.execute('evaluate', target, cases=retention, generate=True,
                           query={'max_new_tokens': 1}, forward_limit=1)
        except WeightlessError as error:
            assert error.code == 'forward_limit_exceeded'
        else:
            raise AssertionError('Worker did not enforce its forward limit')
        output = root / 'export'
        output.mkdir(mode=0o777)
        output.chmod(0o777)
        max_bytes = 1024 ** 3
        image = runner.config['image']
        artifact = operator(image, _EXPORT, {'bundle': bundle, 'target': target, 'max_bytes': max_bytes},
                            [(target['source'], target['source'], True), (output, '/output', False)],
                            file_limit=max_bytes)
        write(root / 'export-receipt.json', artifact)
        reloaded = operator(image, _RELOAD,
                            {'original_source': target['source'], 'target': target, 'checkpoint': selected,
                             'cases': public, 'expected': {'metrics': expected['metrics'],
                                                          'cases': [{k: v for k, v in row.items() if k != 'prediction_truncated'}
                                                                    for row in expected['cases']]}},
                            [(output / 'selected', '/artifact', True)], file_limit=max_bytes)
        write(root / 'reload-proof.json', reloaded)
        summary = {'passed': True, 'proposal_mode': 'scripted_plumbing', 'fresh_agent': False,
                   'initial': descriptor['checkpoint'], 'selected': selected, 'source_unchanged': True,
                   'edited_checkpoint_reloaded': True, 'original_source_unavailable': True,
                   'reserved_forward_calls': bundle['used']['forward_calls'],
                   'one_token_forward_calls': tiny['usage']['forward_calls'],
                   'forward_limit_enforced': True, 'final': final,
                   'verification_scope': 'Real CPU SmolLM checkpoint, scripted edit, exact reload and finite-case evaluation; no discovery or general efficacy claim.',
                   'post_freeze_checks': 'One source description, two forward-limit probes, export validation, and one independent reload/evaluation outside the development budget.'}
        write(root / 'summary.json', summary)
        print(json.dumps(summary, allow_nan=False))
    finally:
        session.close()
        verifier.close()


if __name__ == '__main__':
    main()
