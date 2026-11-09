# Tinker capability boundary

WSM requires a target backend that can inspect permitted weights, materialize explicit replacements in separate candidates, evaluate those candidates, and export the selected checkpoint. The editing AI is a separate caller. Its provider's inference API does not grant access to the target's weights.

## Inspected contract

On 2026-11-09, inspection covered the public client interfaces, RestClient, weight resources, and upload types in the published `tinker==0.39.0` Python wheel. It also covered the official API references below. The wheel's SHA-256 is `47c87de4e13ca6eeaecfff55b316bd436344d98685091c3024c44462e946250d`. The TrainingClient documentation identified its generated source as SDK 0.38.0. SDK inspection establishes the client contract, not a live service integration.

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

Implementing a Tinker-compatible service is a different product decision. Compatibility needs named SDK calls, request and response contracts, and tests using the real Tinker client. Direct tensor editing would require an explicit extension where the inspected API has no equivalent. A method with a similar name or a mock response does not establish compatibility.

The operator must choose between hosted-service integration and a compatible service interface. Neither choice is implicit in WSM's local backend. No Tinker endpoints, credentials, optimizer calls, or remote fallback are added by this work.

## Sources

- [TrainingClient reference](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/trainingclient/).
- [ServiceClient reference](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/serviceclient/).
- [SamplingClient reference](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/samplingclient/).
- [RestClient reference](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/restclient/).
- [Checkpoint management](https://tinker-docs.thinkingmachines.ai/tinker/howto/checkpoints/).
- [Tinker Python distribution](https://pypi.org/project/tinker/0.39.0/).
