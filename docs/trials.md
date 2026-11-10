# Run a coding-agent trial on Modal

This experimental PoC connects a Codex teacher through a CLI proxy to one Modal editing sandbox, then distills the teacher's actions into an open-weight student through Spindle and the real Tinker SDK. The target is `HuggingFaceTB/SmolLM2-135M-Instruct` at revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`.

## Use the bounded Spindle proof

Use Python 3.12 for Spindle's serialized runtime images. Install the pinned server tooling:

```sh
python -m pip install -e 'packages/python[spindle]'
```

The [single-client recipe](../examples/spindle/qwen3_4b.py) uses Qwen3-4B LoRA as the student. Codex makes the edits through the CLI proxy. The target remains SmolLM2. The recipe limits the server to one H100 trainer and permits at most one sampler with zero minimum replicas. Teacher distillation never creates a student sampler. This overrides the upstream preset's multiple trainer GPUs.

The default task improves short English completions. Hidden target-token negative log likelihood supplies the final reward. Student training retokenizes the exact teacher prompts and responses with the student tokenizer, masks prompt tokens, appends EOS, and applies cross entropy to response tokens with weight `exp(clamp(reward, -20, 0))`. Training uses rank 8, learning rate `1e-5`, and at most three teacher turns. This is supervised teacher distillation weighted by the final score. The teacher's tokens are not an on-policy student rollout. The closed Codex model itself is not updated by Spindle.

Provision the recipe's named API and proxy secrets with the [Spindle setup instructions](https://github.com/modal-projects/spindle/tree/v0.1.6). The API secret contains a newly generated `tml-` prefixed key. Store the same key in a private local JSON file with a `TINKER_API_KEY` field. Use the existing `huggingface-secret` for model downloads. A workspace-scoped proxy token does not require the environment `allow` command. An environment-scoped token must be allowed for `main`.

For the teacher, supply a separate private JSON file containing `model` and `api_key` for the local CLI proxy. Keep its opaque route out of source and public records. The controller exposes it only through `WSM_EDITOR_MODEL` and `WSM_EDITOR_API_KEY`; recorded artifacts use the display label `Codex` and a route fingerprint. The proxy endpoint must support streaming Responses calls with tools disabled. The transport retains exact prompt and response text, sanitized stream events, and uncertain-send receipts.

Run the bounded owner:

```sh
python scripts/spindle_poc.py --model-source /path/to/staged/source \
  --grader-python /path/to/model-runtime/bin/python \
  --api-key-file /private/spindle-api.json --editor-config-file /private/editor.json \
  --prior-reserved-usd 0 --output runs/spindle-proof
```

The owner deploys only the task-owned `wsm-spindle-poc` apps, resolves their endpoint, runs the trial, and stops those apps. It rejects a pre-existing live deployment under that name. CPU image preparation has a 30-minute deadline; the GPU trial and training phase then has 840 seconds, with another 360 seconds reserved for shutdown, including Spindle's sampler exit grace. GPU containers have zero minimums and are requested only by the training client. A separate watchdog stops owned child processes and apps if the owner dies or loses its deadline. Cleanup stops apps by ID, so interrupted deployments are covered even before a name is registered, and confirms zero tasks including apps already marked stopped. The proof requires an agent-authored publication and trusted final tensor changes before reporting success.

The owner sets `MODAL_IMAGE_BUILDER_VERSION=2026.10` for its process tree. This avoids stale Debian package indexes from the legacy builder without changing workspace configuration.

The US$10 reservation includes earlier work. `--prior-reserved-usd` is required. Zero in the example means a fresh allocation with no prior work. Supply the known cumulative total before another run. New reservations include CPU image builds, the complete student trainer resource window, target sandbox time, and retained-artifact egress. The default US$8/hour student allowance covers one H100 and its CPU/memory with headroom at the rates inspected on 2026-11-10. Teacher CLI-proxy usage is separately bounded by request and token limits against the configured Codex access; it is not billed as Modal GPU compute. Verify current prices before rerunning. These reservations are conservative allowances, not reconciled invoices or provider billing caps.

Use `--student-hourly-usd` and `--student-seconds` to set a checked resource allowance. The work deadline is the smaller of 840 seconds and the student envelope minus 360 seconds for cleanup. The final recovery proof used US$6/hour for 600 seconds, which exceeds the inspected one-H100 trainer resource rate while fitting the remaining reservation. This shorter envelope is intended for cached deployments.

The watchdog cannot survive loss of the whole host. The trainer's native timeout remains a second bound. For stronger billing enforcement, use account-side controls. No automatic retry follows an uncertain update or a failed live proof.

## Configure the task and training policy

Install the controller and local grading dependencies in your Python environment:

```sh
python -m pip install -e 'packages/python[editing,trial]'
```

Authenticate Modal using its CLI. Set the Responses-proxy `policy` and Spindle `student` endpoints explicitly. Configure `WSM_EDITOR_MODEL`, `WSM_EDITOR_API_KEY`, and `WSM_SPINDLE_API_KEY` on the controller. Spindle never falls back to the hosted service's key. None of these credentials enters the editing sandbox.

Copy [the configuration template](../examples/coding/trial-config.json) outside the repository and fill the local paths and deployed endpoint. It contains the Spindle PoC defaults above. Resolve the relative public-case path when moving the file.

Set `model_source` to a local directory containing `snapshots/<revision>`. Use the source staged for the [existing Modal proof](coding.md#reproduce-the-modal-proof), or stage the same pinned download. Each trial creates a fresh sandbox from that unchanged source. It never starts from another trial's final checkpoint.

Put public and held-out cases in separate JSON files. Both use the coding API's list of `{"prompt": "...", "target": "..."}` records. Keep the private file outside the model source and package directories. The controller rejects overlapping cases and private files under image inputs.

Choose one final scoring rule explicitly:

- `negative_loss` returns the negative final mean target-token NLL.
- `loss_improvement` returns the untouched model's held-out loss minus the final model's held-out loss.

The teacher/student template selects reward-weighted cross entropy. Student tokenization and prompt masks are retained in `training/datums.json`; the receipt binds them to the teacher trajectory and final reward. This is one update, not a claim of learned editing ability. The separate direct-policy API still requires its own explicit Tinker-compatible sampling provider and uses sampled token log probabilities for policy-gradient updates.

## Set conservative cost reservations

The budget journal reserves costs before cloud operations and never refunds uncertain requests. Set `max_usd` to your authorized total. For the initial PoC the authorized ceiling is US$10.

Enter current rates from [Modal pricing](https://modal.com/pricing). `modal_hourly_usd` must cover the T4, two CPU cores, and 8 GiB RAM for the full sandbox timeout. Sandbox CPU and memory pricing differs from ordinary Functions. `modal_egress_per_gib` covers downloading each retained checkpoint. For Spindle, `spindle_hourly_usd` and `spindle_seconds` reserve the entire server envelope before creating the training client; subsequent token operations consume that prepaid window without token billing. For the hosted provider, use the `tinker_*` fields with model-specific [token prices](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/) and storage allowances instead. The client requests a one-hour training-checkpoint TTL; provider expiration behavior requires separate verification.

These are controller reservations based on supplied prices, not a provider-enforced billing cap. Image builds, startup overhead, delayed billing, and inaccurate allowances can make billed cost differ. Set account-side spending controls when you require a hard dollar stop. A run whose next reservation exceeds the configured total stops before dispatching that operation. Do not run with unverified price allowances.

Validate configuration without making cloud calls:

```sh
wsm trial preflight --config /path/to/trial.json
```

Preflight checks paths, task splits, explicit training choices, and price fields. It does not authenticate providers, verify model availability, or prove the supplied prices.

## Run and watch

Run one trial and one training update:

```sh
wsm trial run --config /path/to/trial.json --output runs/trial-001
```

In another terminal, watch the public journal after the trial directory appears:

```sh
wsm trial view runs/trial-001/trial --follow
```

Omit `--follow` to read retained events. The viewer has no pause, edit, or submission controls. It escapes terminal control characters in recorded output.

List tensor shapes in a retained checkpoint, or inspect numerical values by flat index:

```sh
wsm trial weights runs/trial-001/checkpoints/edit-EXAMPLE.safetensors
wsm trial weights runs/trial-001/checkpoints/edit-EXAMPLE.safetensors \
  --tensor model.norm.weight --indices 0 1 2
```

Use the artifact path recorded in the trial history. These commands open inert tensor data without executing model code and have no mutation controls.

The agent returns JSON actions for `python`, `evaluate`, `history`, and `finish`. A `python` action contains a complete file defining `edit(model)`. The existing `execute_program` runs that file against the previous checkpoint. Every successful publication becomes current, regardless of public score. The agent can undo by copying tensors from a previous checkpoint in a later file.

Python globals do not survive between files. A failed or interrupted file does not publish its partial in-memory changes, and the trial stops. WSM never silently retries an uncertain command. Turn limits also stop malformed actions and repeated history requests.

## Inspect the retained evidence

The output directory contains these records:

- `trial/events.jsonl` contains controller-recorded teacher messages, commands, source, outputs, public evaluations, and checkpoint references.
- `trial/actions/` preserves the submitted Python files.
- `checkpoints/` retains safetensors for every successfully completed file. These remain inspectable after sandbox termination. Checkpoint records include the host-computed artifact hash and file size.
- `final.safetensors` is the frozen final artifact supplied to the private grader.
- `trial/grade.json` contains the private final result. It never enters public history or later agent prompts.
- `training/receipt.json` records the student model, teacher trajectory hash, final reward, response weight, exact student datum hash, API progress, metrics, and saved training state path.
- `budget.jsonl` records pre-dispatch reservations. `poc.json` records the controller's overall outcome.

The agent receives public action history through its history action and a readable projection at `/trial/history.json`. Tool feedback longer than 6,000 characters is explicitly shortened in the next model prompt; complete results remain in history and can be read selectively through Python. The controller's canonical journal stays outside the sandbox. Source-level evaluation calls and tensor fingerprints are worker observations. Arbitrary Python can alter its own runtime, so these observations are not authenticated measurements. WSM records changes at file boundaries, not every individual tensor assignment or custom computation inside a file.

After freezing, the controller exports weights and terminates the sandbox before grading. A fresh local process loads the original trusted model configuration and tokenizer, validates the exported tensor names and layouts, and applies the safetensors. Agent-provided configuration and Python never enter that process. Held-out cases are supplied only there. Only after grading does the controller create the student's Spindle training client and use the final scalar reward to weight its supervised update.

If training reports `outcome_unknown`, inspect the retained receipt and provider session. Do not resubmit the same trajectory automatically. Completed state saving proves an API update completed; it does not prove model quality improved.

## Recover a frozen teacher trial

If grading fails before any training attempt, pass `--resume-trial /path/to/original/run` to the bounded owner. It verifies the retained trajectory against canonical messages and the final weights against the captured checkpoint hash. It grades that artifact without rerunning Codex. The grading executable retains its virtualenv symlink so its installed Torch runtime is used.

An explicit `--retry-pretraining` is permitted only when the prior receipt proves a local failure before training, its durable budget journal contains no training or save dispatch, and prior cleanup confirmed zero tasks. The prior attempt remains recorded. Recovery rechecks the independent grade and refuses a different result. It never repeats an uncertain optimizer operation. Jinja2 is an explicit runtime dependency; student chat tokenization requests a plain token list across Transformers versions.

## Verify locally

Run the focused tests in a runtime with Torch, Transformers, safetensors, and Tinker installed:

```sh
PYTHONPATH=packages/python/src:packages/python/tests \
python -m unittest test_trial test_trial_poc test_trial_training -q
```

The tests use a tiny local model to check persistent worse edits, explicit undo, fresh reset, complete source and output recording, a separate grader, hostile configuration exclusion, BF16 source to float32 publication, and token-loss alignment with real SDK types. Simulated transport failures do not establish hosted Tinker integration. The completed Codex/Spindle live proof is recorded in [verification](verification.md#modal-teacher-and-spindle-student-proof). Reruns still require valid credentials and checked prices.

To verify sandbox mechanics using the existing deterministic example without choosing a training task, run:

```sh
python scripts/verify_trial_modal.py --model-source /path/to/staged/source \
  --output /tmp/wsm-trial-modal-proof
```

This script reserves up to US$2, runs two sequential 300-second T4 sandboxes, and checks a worse edit, persistent evaluation, Python undo, agent history access, retained tensor artifacts, fresh reset, and confirmed termination. Its conservative allowances must still be checked against current provider prices. It does not run an editing-agent rollout or a Tinker training update.

General Hugging Face dataset ingestion, arbitrary Harbor task ingestion, concurrent trials, persistent Python processes, and multi-update training loops are outside this PoC.
