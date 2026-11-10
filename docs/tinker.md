# Tinker capability boundary

## Decision for the Python editing program

Rechecked the official TrainingClient, ServiceClient, RestClient, and supported-model references on 2026-11-10. The concrete program is `model.weights["model.norm.weight"][:16].mul_(0.95)`, followed by evaluation and standalone checkpoint reload.

The documented TrainingClient methods include `forward`, `forward_backward`, `optim_step`, state loading, and external weight saving. None exposes named model tensors for this Python mutation. ServiceClient creates LoRA clients and clients from saved state. SmolLM was not listed on the inspected supported-model page. These observations establish a documented-contract gap, not proof of undocumented service behavior. No authenticated hosted Tinker experiment was run.

A possible compatibility subset would name `ServiceClient.create_training_client_from_state`, `TrainingClient.forward`, and `TrainingClient.save_state`. Implementing it would require real SDK tests of `Datum`, loss outputs, futures, checkpoint identifiers, errors, and reload semantics. It would still lack tensor editing. WSM's evaluation dictionary and Hugging Face directory are different contracts; similar names would not make them compatible.

The implemented experimental coding-first SDK therefore owns `WeightModel.load`, `weights`, `evaluate`, and `save`, with `execute_program` as a separate execution boundary. The local SDK backend below provides a separate sampling and scoring contract with an explicit WSM coefficient-editing extension. It does not expose this `WeightModel` program through hosted Tinker. The earlier SDK-wheel inspection below identifies an experimental adapter import route; it does not establish arbitrary dense edits or hosted support for this program.

- [Supported models](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/).
- [Coding workflow](coding.md).

WSM requires a target backend that can inspect permitted weights, materialize explicit replacements in separate candidates, evaluate those candidates, and export the selected checkpoint. The editing AI is a separate caller. Its provider's inference API does not grant access to the target's weights.

## Local SDK backend

WSM supplies a local service for the official `tinker==0.39.0` client. This is a bounded compatibility subset for a configured WSM experiment. The supported numerical targets are the pinned SmolLM checkpoint and staged local Hugging Face models. The service binds one credential to one experiment arm. It listens on loopback and retains the existing session as the owner of budgets, candidate selection, and evidence.

Supported standard calls are server capabilities, creation of a sampling client, greedy `sample`, and `compute_logprobs` on token-only prompts. Sampling permits one sequence and at most 128 new tokens. Unsupported training, optimizer, LoRA, multimodal, stochastic, and advanced sampling requests fail explicitly. No hosted Tinker connection is made. Compatibility with other SDK versions requires a separate check.

Stopping follows the SDK contract. Omitted `stop` or `stop=None` uses tokenizer EOS. An explicit `stop=[]` disables all stopping until `max_tokens`, while a nonempty integer list uses exactly those stop tokens. The adapter preserves this distinction in retry fingerprints. Completed futures retain their encoded response bytes, so repeated retrieval returns the same result without revalidating or re-encoding numerical output.

Install the optional client dependency:

```sh
python3 -m pip install -e 'packages/python[tinker]'
```

Start the service against an existing `wsm.workspace.v1` experiment. Set `WSM_TINKER_API_KEY` in the operator's environment first. The service does not print it.

Use a local secret starting with `tml-`, which the official SDK requires. It is a credential for this local server, not a hosted Tinker credential.

```sh
python3 -m wsm.tinker_backend ./experiment --arm teacher --port 8000
```

For a new SmolLM workspace, stage the already downloaded pinned snapshot and create an experiment from trusted Python:

```python
from wsm import stage_smollm
from wsm.client import Experiment

target = stage_smollm("/models/smollm-cache", "/models/smollm-staged")
config = {
    "format": "wsm.workspace.v1",
    "target": target,
    "runner": {"backend": "bubblewrap", "python": "/opt/wsm/bin/python",
               "timeout": 90, "memory_bytes": 4 * 1024**3},
    "declaration": {
        "hypothesis": "Test explicit coefficient edits on a short response task",
        "arms": ["teacher"],
        "editable": ["model.layers.0.self_attn.q_proj.weight"],
        "max_edits": 16, "max_inspect": 64, "wall_seconds": 1800,
        "claim": "exploratory",
        "budgets": {"attempts": 8, "evaluations": 32, "inspections": 32,
                    "teacher_requests": 16, "code_executions": 8,
                    "final_evaluations": 0, "forward_calls": 1024},
    },
    "development": [{"id": "hello", "prompt": "Say hello.", "target": "Hello."}],
}
experiment = Experiment.create("experiment", config)
experiment.close()
```

The runner's Python environment must have the target's PyTorch and Transformers dependencies. It must be usable inside the sandbox; a virtual environment that links to an interpreter outside `/usr` and outside its mounted runtime will fail to start. This example has only public development feedback and makes no private correctness claim.

A teacher program can use the standard SDK for inference and the WSM extension for code edits:

```python
import os
import tinker
from transformers import AutoTokenizer
from wsm.tinker_client import WeightEditor

url = "http://127.0.0.1:8000"
key = os.environ["WSM_TINKER_API_KEY"]
client = tinker.ServiceClient(base_url=url, api_key=key)
editor = WeightEditor(url, key)
view = editor.observe()
tensor = "model.layers.0.self_attn.q_proj.weight"
feedback = editor.run_code(
    "old = weights['model.layers.0.self_attn.q_proj.weight'][0]\n"
    "print('before', old)\n"
    "weights['model.layers.0.self_attn.q_proj.weight'][0] = old * .99",
    [{"tensor": tensor, "indices": [0]}],
    view=view, key="experiment-code-1",
)
print(feedback)

# Use the real target tokenizer, not the local service alias 'wsm'.
tokenizer = AutoTokenizer.from_pretrained("/local/smollm-snapshot", local_files_only=True)
tokens = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Say hello."}],
    tokenize=True, add_generation_prompt=True,
)
sampler = client.create_sampling_client(base_model="wsm").result()
prompt = tinker.types.ModelInput.from_ints(tokens)
sample = sampler.sample(
    prompt, num_samples=1,
    sampling_params=tinker.types.SamplingParams(max_tokens=16, temperature=0),
).result()
print(tokenizer.decode(sample.sequences[0].tokens, skip_special_tokens=True))
print(sampler.compute_logprobs(prompt).result())
client.close("success").result()
```

The numerical edit above demonstrates mechanics, not a known improvement. The teacher chooses useful computations and replacements. `observe()` explicitly refreshes inspection bindings for a later turn. `submit` and `run_code` always send the view supplied by the caller. They never retry or silently bind an old edit to a new checkpoint.

The experiment must have evaluation and inspection budgets. Declare `budgets.forward_calls` to bound SDK inference. Each sampling or scoring request consumes one evaluation and, when declared, reserves `max_tokens + 1` forward calls. Reservations and unknown outcomes survive in the session journal. Retrieving a future or retrying the same operation does not execute the target again. Sampling clients bind to the checkpoint that existed when they were created. Create another sampling client after an accepted edit.

The WSM extension provides permitted weight inspection, explicit replacement, and isolated Python execution. A code command receives only the requested coefficients as `weights[tensor][index]`. It can calculate statistics, print bounded feedback, use installed numerical libraries, and change those values. The host validates the diff and submits it through ordinary development evaluation. The Python process receives neither the target directory nor host credentials or private datasets. It cannot make network requests. Declare `budgets.code_executions` to enable it. Each requested tensor consumes one inspection; the total addresses per program cannot exceed `max_inspect`.

The code program is a bounded numerical computation. Calls to the Tinker SDK and WSM extension happen in its controlling teacher program. Network access remains outside the isolated numerical program.

The adapter adds no replacement training loop or second selection policy. A larger implementation with an independent model registry was rejected because it would duplicate the session's checkpoint ownership and admission rules. Ordinary Tinker sampling stays separate from the explicitly named WSM editing extension.

Service handles are local to one running server. Restarting the server invalidates SDK sampling handles; create a new client. The durable WSM session and operation keys remain available. This service is for local experiments and is not a multi-tenant deployment.

`scripts/verify_tinker_backend.py WORKSPACE --output REPORT.json` exercises the actual SDK, scores prompt tokens, executes an isolated coefficient edit, verifies an idempotent retry, freezes the experiment, and checks replay. It consumes the workspace's declared budgets and freezes it. Use a separate verification workspace. The default tensor matches the pinned SmolLM profile; `--tensor`, `--index`, and `--tokens` adapt the proof to another model.

On 2026-11-10, this proof ran with the official SDK 0.39.0 against the pinned SmolLM2-135M-Instruct snapshot in bubblewrap. Sampling and prompt scoring agreed. Python changed the requested coefficient from `-0.0751953125` to a proposed `-0.074443359375`; the worker materialized its float32 replacement and evaluated it. The candidate was rejected, leaving the selected checkpoint unchanged. Retrying the code operation returned its retained result, and frozen evidence replay passed. The experiment used one attempt, one code execution, four evaluations, three inspections, and 38 reserved forward calls. This verifies the integration mechanics, not a model capability improvement.

## Inspected contract

On 2026-11-09, inspection covered the public client interfaces, RestClient, weight resources, and upload types in the published `tinker==0.39.0` Python wheel. It also covered the official API references below. The TrainingClient documentation identified its generated source as SDK 0.38.0. SDK inspection establishes the client contract, not a live service integration.

| Required capability | Public Tinker contract | Consequence for WSM |
| --- | --- | --- |
| Read named parameter tensors or slices | No public tensor-inspection method found in TrainingClient, SamplingClient, or ServiceClient | Remote weight inspection is unverified and unsupported by the inspected contract. |
| Apply explicit numerical tensor replacements | No public tensor-replacement method found | `optim_step` is optimizer-driven training and cannot substitute for direct edits. |
| Load a checkpoint | `load_state`, `load_state_with_optimizer`, and clients created from saved state | Existing Tinker checkpoints can initialize clients. SDK 0.39.0 also exposes the experimental external upload route described below. |
| Export weights | `save_weights_external` and checkpoint-download APIs | Saved weights can leave the service. Export alone does not prove an edited checkpoint can return to it. |
| Import external adapter weights | `RestClient.external_weights_upload_start` and `external_weights_upload_finish` in SDK 0.39.0 | Experimental route with exact base-model, LoRA-configuration, and file-format requirements. Acceptance of edited values needs live verification. |
| Evaluate without gradients | `TrainingClient.forward` | Supported for the documented data and loss formats. |
| Generate and score text | `SamplingClient.sample` and `compute_logprobs` | Supported for available base models and eligible checkpoints. |

The SDK's `ServiceClient.copy_weights` copies an existing Tinker checkpoint into a new session in the client's project. The copy shares storage and cannot be trained on. Its existence does not establish a numerical tensor-write API.

### Experimental adapter import

SDK 0.39.0's `tinker/lib/public_interfaces/rest_client.py` exposes `external_weights_upload_start(base_model, lora_config, destination_project_id=None)` and `external_weights_upload_finish(upload_id)`. Start returns signed HTTP PUT URLs. Finish validates the uploaded files against the exact format exported by `save_weights_external` for that base model and LoRA configuration. It registers an external checkpoint that can be loaded with `load_state` into a matching client. Both methods are explicitly experimental. The inspected generated RestClient documentation did not yet list them.

This is a concrete SDK route worth testing for locally edited adapter weights. It is not evidence that arbitrary dense base-model replacements are accepted. WSM has not uploaded an edited adapter, loaded it, or sampled it through this route. A live round trip must verify file layout, numerical changes, checkpoint loading, and inference before this can be called an integrated backend.

`load_state` is deprecated in SDK 0.39.0 in favor of `ServiceClient.create_training_client_from_state`. The upload-finish documentation still names `load_state`. This mismatch also needs resolution in a live integration check. A complete check must save sampler weights after loading and run inference from that checkpoint, rather than treating upload completion as success.

## Compatibility choices

Using hosted Tinker could start with downloading and editing exported LoRA weights, then testing the experimental import route. That edits adapter factors, which is a narrower contract than arbitrary base-model tensor editing. The operator must choose whether that restriction meets the experiment's needs. WSM currently makes no claim of a live Tinker integration. A downloaded adapter edited locally is a local experiment until remote reloading and inference are verified.

The local service above implements named SDK calls with tests against the real client. Direct tensor editing remains an explicit WSM extension because the inspected Tinker API has no equivalent. It does not establish compatibility with Tinker's training or optimizer APIs.

The local backend implements the compatible-service choice. Hosted-service integration remains unverified. No hosted endpoint, hosted credentials, optimizer calls, or remote fallback are introduced.

## Sources

- [TrainingClient reference](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/trainingclient/).
- [ServiceClient reference](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/serviceclient/).
- [SamplingClient reference](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/samplingclient/).
- [RestClient reference](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/restclient/).
- [Checkpoint management](https://tinker-docs.thinkingmachines.ai/tinker/howto/checkpoints/).
- [Tinker Python distribution](https://pypi.org/project/tinker/0.39.0/).
