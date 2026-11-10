"""Exercise the real Tinker SDK and isolated code edits on an existing model workspace."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import secrets

import tinker

from wsm import replay_session
from wsm.client import Experiment
from wsm.tinker_backend import TinkerServer
from wsm.tinker_client import WeightEditor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace', type=Path)
    parser.add_argument('--arm', default='teacher')
    parser.add_argument('--tensor', default='model.layers.0.self_attn.q_proj.weight')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--tokens', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    experiment = Experiment.open(args.workspace)
    credential = 'tml-' + secrets.token_hex(32)
    try:
        with TinkerServer(experiment.session, arm=args.arm, api_key=credential) as server:
            client = tinker.ServiceClient(base_url=server.base_url, api_key=credential)
            try:
                sampler = client.create_sampling_client(base_model='wsm').result()
                prompt = tinker.types.ModelInput.from_ints(args.tokens)
                sampled = sampler.sample(prompt, num_samples=1,
                    sampling_params=tinker.types.SamplingParams(max_tokens=2, temperature=0),
                    include_prompt_logprobs=True).result(timeout=180)
                assert sampled.sequences and sampled.sequences[0].tokens
                print('SDK sampling passed', flush=True)
                scores = sampler.compute_logprobs(prompt).result(timeout=180)
                assert scores == sampled.prompt_logprobs
                print('SDK prompt scoring passed', flush=True)
                editor = WeightEditor(server.base_url, credential)
                view = editor.observe()
                inspected = editor.inspect(args.tensor, [args.index], key='proof-inspect')
                assert inspected['status'] == 'completed'
                value = inspected['result']['values'][0]['value']
                source = (f"old = weights[{args.tensor!r}][{args.index}]\n"
                          f"weights[{args.tensor!r}][{args.index}] = old * .99 if old else .001\n"
                          f"print(old, weights[{args.tensor!r}][{args.index}])")
                edited = editor.run_code(source, [{'tensor': args.tensor, 'indices': [args.index]}],
                                         view=view, key='proof-code')
                assert edited['status'] == 'committed', edited
                print('Isolated code edit evaluated', flush=True)
                assert edited['result']['evaluation']['checkpoint'] != view['base']
                assert edited['result']['stdout']
                assert editor.run_code(source, [{'tensor': args.tensor, 'indices': [args.index]}],
                                       view=view, key='proof-code') == edited
                refreshed = editor.observe()
                after = editor.inspect(args.tensor, [args.index], key='proof-after')
                assert after['result']['checkpoint'] == edited['selection']['after']
                if not edited['selection']['accepted']:
                    assert after['result']['values'][0]['value'] == value
                report = {'sdk': importlib.metadata.version('tinker'), 'target': experiment.session.target['model'],
                          'generated_tokens': sampled.sequences[0].tokens, 'prompt_logprobs': scores,
                          'code_stdout': edited['result']['stdout'], 'selection': edited['selection'],
                          'candidate_metrics': edited['result']['evaluation']['metrics'],
                          'checkpoint': refreshed['base'], 'idempotent_code_retry': True}
            finally:
                client.close('success').result(timeout=30)
        frozen = experiment.session.freeze(key='proof-freeze')
        assert replay_session(experiment.session.export()) == frozen
        report['replay'] = 'passed'
        report['used'] = experiment.session.export()['used']
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    finally:
        experiment.close()


if __name__ == '__main__':
    main()
