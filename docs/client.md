# Durable local client

The local client adds `Session`, `Verifier`, `BubblewrapRunner`, `DockerRunner`, and `ResponsesTeacher` beside the existing `SearchRun` API. The existing checkpoint and run formats retain their meaning. Session exports use a separate format.

## Ownership and supported execution

`BubblewrapRunner` provides Docker-free execution on the current Linux host using bubblewrap namespaces and systemd cgroups. Its [sandbox boundary and native checks](sandbox.md) describe the required system permissions, filesystem grants, resource limits, and teardown guarantees. The acceptance driver defaults to this runner. The Docker profile below remains available when explicitly selected.

A trusted host owns the SQLite session, private verifier registry, teacher credentials, and Docker socket. An untrusted agent receives an arm-scoped JSON observation and submits commands through `Session.dispatch`. Do not give the agent a Python session object, host shell, database path, or Docker socket. These objects are privileged orchestration APIs.

`DockerRunner.probe` executes agent bridge code without host mounts or network access. The acceptance driver passes model responses through this isolated JSON bridge before dispatch. The numerical worker runs in another container and receives only the pinned public model directory as a read-only mount. Both containers use an unprivileged UID, a read-only root filesystem, no network, no capabilities, no new privileges, bounded processes and memory, and a bounded temporary filesystem. The host enforces deadlines and removes containers after execution. Teardown can take an additional bounded grace period. Cancellation freezes selection and reports that remote termination is unconfirmed.

The current numerical backends are binary64 affine/ReLU checkpoints and the CPU float32 `HuggingFaceTB/SmolLM2-135M-Instruct` checkpoint at revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`. The worker pins greedy generation, eager attention, one CPU thread, offline loading, and no remote model code. It records parameter aliases, runtime identity, literal generations, and sandbox observations. Scalar replacements preserve tied embeddings. Unsupported overlapping storage views fail closed. `stage_smollm` dereferences a cached snapshot into a new self-contained directory. It never changes the source cache.

This profile trusts the host administrator, Docker daemon, image, and numerical runtime. It does not protect against a compromised kernel or runtime. The private verifier database must remain outside every agent container. Docker isolation is enforced by the runner, not by Python object permissions.

## Session contract

`Session.create(path, runner=..., target=..., declaration=..., development=..., verifier=..., final_dataset=...)` binds the target, runner configuration, public cases, and immutable declaration. `Session.open` resumes the persisted state. Each arm has its own selected checkpoint, observation version, pending operation, and history.

A declaration contains `hypothesis`, `arms`, canonical `editable` tensor names, `max_edits`, `max_inspect`, positive `wall_seconds`, and integer `budgets`. Budget names are `attempts`, `evaluations`, `inspections`, `teacher_requests`, and `final_evaluations`. Optional `diagnostics` also consumes the evaluation budget. The current selection policy minimizes development loss strictly. Optional `development_retention_loss_max` requires public retention cases. Model families, resource ceilings, and acceptable regressions remain operator choices within the implemented profile.

### Bound target forward calls

The optional `budgets.forward_calls` dimension bounds target execution separately from evaluation requests. A model forward call means one top-level target model invocation. One loss evaluation uses one call per case; greedy generation can use up to one more per generated token. The worker measures those invocations with a scoped hook and checks its assigned limit before each call. Binary64 counts one network prediction per case.

Session evaluations use a fixed 16-token generation cap. A budgeted SmolLM evaluation therefore reserves 17 calls per case before dispatch, including the first unchanged baseline, candidate, public retention, and diagnostics. Binary64 reserves one call per case. Unused allowance is not refunded. `used.forward_calls` is the reserved total. Successful public feedback records measured `usage.forward_calls` and the assigned `usage.forward_limit` separately. Public retention usage appears in `retention_usage`. Failed or uncertain execution retains its whole reservation; absence of an observed count does not mean zero calls.

A budgeted session with a private verifier also requires a nonnegative `declaration.final_forward_calls`. This is an operator-chosen padded envelope for all final task and retention evaluations across all arms. Finalization reserves it once before execution. The verifier rejects an insufficient envelope before private dispatch and checks each numerical result against its own bound. Exact private measured counts remain in the verifier database and do not enter agent feedback or the public evidence bundle. A failed preflight retains the reservation, and retrying its idempotency key does not charge it again.

These units differ from the batched evaluation budgets in the research. They do not measure tokens, FLOPs, cost, or provider billing. Token-length, CPU, memory, and wall-time limits still apply. Describe, inspect, materialize, and checkpoint serialization consume no target forward calls but still use compute resources. Post-freeze reload checks and independent regrading must disclose their own execution separately from the closed development budget.

`observe(arm)` returns current identity fields, permitted parameter metadata, public cases, history, remaining counts, and phase-appropriate actions. `inspect` requires a checkpoint, tensor, and bounded index list. A submission contains exactly `session`, `arm`, `declaration`, `observation`, `base`, and `edits`. Each edit contains `tensor`, `index`, and finite `value`. Every numerical operation has an idempotency key. Repeating the same key and request returns the same outcome. Changing a request under an existing key is an error. A stale checkpoint or observation is rejected before candidate evaluation.

The first submission establishes an unchanged baseline. Candidate materialization verifies the base identity before evaluation. Effective float32 replacements are recorded after rounding, so replay reproduces the resulting weights. Each candidate starts from copied base weights. Rejected edits never replace the selected checkpoint. Failures and uncertain outcomes remain in the journal. An uncertain external operation is never silently resent.

`freeze(key=...)` records a cutoff, drains durable results that precede it, and prevents later selection. An incomplete freeze cannot finalize. `cancel` uses the same selection barrier. `finalize(key=...)` grades only frozen candidates through the separately bound verifier. `export()` contains the declaration, operations, literal public observations, journal, frozen selection, and final receipt. It excludes private cases and teacher credentials.

## Evaluation and conclusions

The verifier registry stores private datasets separately from the session. A dataset can contain `task` and `retention` cases. The registry binds grading to the frozen runner and target identity, serializes concurrent grading, and records release intent before disclosing a receipt. An interrupted grading operation becomes a stable unknown outcome instead of a fresh hidden evaluation. Dataset claim identities use a private keyed digest.

Target-conditioned language loss and exact free generation are different measurements. A lower loss does not establish the intended behavior. `diagnose` records literal generation, EOS behavior, and loss under the same resource budgets. Retention grading reports unrelated language loss. A single retention fixture is a compatibility check, not a comprehensive capability assessment.

An exploratory declaration leaves criteria unset and produces an inconclusive efficacy conclusion. A confirmatory declaration requires both `criteria.final_exact_min` and `criteria.retention_loss_max`. The verifier checks both on the frozen finite cases. Its supported or bounded-negative conclusion applies only to that declared experiment. It does not establish general weight-space programming or authenticate the runtime's arithmetic.

`replay_session(bundle)` checks journal consistency and reconstructs selection without executing a model. `verify_session(bundle, runner=..., target=...)` independently rematerializes candidate checkpoints and checks numerical identities. It does not independently re-evaluate every recorded score. Hashes detect inconsistency; they do not authenticate a producer. A separate operator can regrade fresh private cases with a trusted runtime.

## Export the selected model

Session evidence and model checkpoints are separate artifacts. `Session.export()` retains the declaration, journal, selected patches, and final receipt. A patch recipe still needs the original base model. Checkpoint export must include the selected numerical model and its tokenizer metadata so another process can load it without the original cache.

Checkpoint export is a trusted operator operation. It is outside `Session.dispatch` and does not give the editing AI a writable host directory. When using a host Python process, its model libraries and filesystem access are trusted. To preserve process isolation, run the operator exporter in a separately limited container with the source mounted read-only and a dedicated output directory. The live editing workers retain their existing read-only model mounts.

Only a complete frozen arm is eligible. Export must check the supplied base identity, apply the selected effective patches, and check the selected identity. Its provenance belongs outside the model snapshot because the numerical identity also binds runtime versions and metadata bytes. A matching set of weight values with different tokenizer or model metadata is not the same WSM checkpoint. Rejected candidates must never replace the selected artifact.

Choose explicit export byte and execution limits. Serialization and reload checks are operator verification work, separate from the frozen development budget. A receipt with file hashes establishes consistency; an independent offline reload and matched evaluation establish that the files can execute. Neither proves a fresh editing AI discovered the patch or that the patch improves general capability.

Use the operator API with the same trusted target descriptor supplied to the session:

```python
from wsm import export_checkpoint

receipt = export_checkpoint(bundle, target=target, arm="editor",
                            destination="/trusted/output/selected", max_bytes=1024**3)
```

The unused destination receives `receipt.json` and either `checkpoint.json` for binary64 or a self-contained `snapshots/<revision>/` Hugging Face model directory. The receipt binds the evidence, frozen arm, runner hash, numerical identities, runtime versions, and relative file hashes. Its `total_bytes` sums model files; the size check also includes the receipt. Source paths, private cases, and credentials are excluded. Load a Hugging Face export directly from its snapshot directory with offline loading and `trust_remote_code=False`.

The exporter preserves the original metadata bytes, applies alias-aware patches, and reloads the serialized model before publication. Source metadata and runtime versions affect WSM identities, so use the same numerical runtime as the experiment. The current atomic no-overwrite publication requires Linux `renameat2`. An existing destination is preserved, including one created concurrently. Failed exports remove only their own temporary directory.

`max_bytes` bounds the completed directory. Preflight accounts for float32 weights even when the source uses a narrower dtype. A child process enforces a per-file `RLIMIT_FSIZE`; this is not a strict aggregate temporary-disk quota. Set CPU, memory, wall-time, and temporary-storage limits in the operator's supervisor. The native verification script uses explicit Docker limits and an independent source-free reload.

## Run a configured experiment

Start with [examples/drive-session.json](../examples/drive-session.json). Run it with `python3 scripts/drive_session.py CONFIG.json` from the repository root after installing WSM. Paths are relative to that working directory. The command creates a new output directory and writes `evidence.json`, `target.json`, and `driver.json` beside the session database. An existing output directory is rejected.

Declare exactly one arm for this driver. Use the Session API directly for multi-arm comparisons. Required configuration fields are `target`, `runner`, `declaration`, `development`, `output`, `arm`, and `max_new_tokens`. The current Session profile requires `max_new_tokens: 16`, `selection: "loss"`, an explicit `claim`, and `budgets.forward_calls`. Supported targets remain binary64 and the pinned SmolLM profile. The initial target and model provider are explicit inputs, with no fallback.

Choose one proposal source:

- `proposals` contains objects with `key` and explicit `edits`. Reports label these runs `scripted_plumbing`.
- `teacher` contains the behavioral `instructions`, `endpoint`, public `model` ID, `api_key_env`, `directory`, `timeout`, and `max_output_tokens`. The environment variable supplies the credential. Alternatively, replace endpoint/model/environment fields with a `credentials` path in the existing teacher-file format. Reports label these runs `fresh_agent`.

The driver supplies the command protocol and passes the literal permitted result into the next prompt. Invalid replies receive bounded feedback and consume their already charged teacher request. Duplicate JSON fields, nonfinite values, invalid payload types, and reserved operation keys are rejected. `driver-freeze`, `driver-finalize`, and keys beginning `teacher-` belong to the driver. `report` ends teacher turns; it cannot set measured scores. The host then attempts to freeze and finalize without reopening selection.

Optional `development_retention` contains public retention cases. A private `verifier` object supplies `database`, `dataset`, and a `cases` file path. The case file contains either a task list or an object with `task` and `retention` lists. Keep verifier files and teacher artifacts outside the output directory. Output and private files must also stay outside the public model mount. Private case text never enters teacher prompts. A budgeted verifier requires the predeclared `final_forward_calls` envelope.

The CLI exits nonzero if freezing is incomplete or a configured final verification does not complete. Its partial evidence remains available. A completed exploratory run can still have an inconclusive efficacy result. The driver does not automatically claim an improvement, and a protocol test with a scripted HTTP responder is not a live editing-AI observation.

## Teacher transport

`ResponsesTeacher` receives an explicit endpoint and model ID. It never selects a fallback. The broker sends `tools: []` and `tool_choice: none`, checks the completed response identity and declarations, and rejects observed tool invocation events. It retains raw request bodies, stream events, responses, and failed or uncertain sends in a private directory. Authorization headers are excluded. These checks establish the observed API boundary, not upstream execution or billing attestation.

Teacher requests reserve session budget before dispatch. Session identity namespaces idempotency artifacts. A saved response can be replayed after a crash. A possible send without a saved response cannot be repeated automatically. A separately charged report request may orchestrate final grading after freeze or summarize a closed experiment without reopening selection.

## Reproduce the native checks

Build from the repository root with Docker available.

```sh
docker build -f examples/client-sandbox/Dockerfile -t weightless-client:local .
PYTHONPATH=packages/python/src python3 scripts/client_native.py \
  --runner docker \
  --image weightless-client:local \
  --model-source /path/to/cached/model-repository \
  --output /tmp/weightless-native-check
```

The output directory must not exist. To resume a teacher acceptance session after a driver interruption, repeat the command with `--resume` and the same directory. The session retains its original deadline, counters, and responses. A possible prior send remains unknown and cannot be retried automatically. The driver tests isolation, timeout cleanup, binary64 final grading, current-client selection conformance, SmolLM aliases, float32 edit replay, restoration, and private final grading. It retains bundles and failure records. `--docker` and `--docker-host` select an explicit Docker installation.

For a fresh teacher run, add `--teacher-config /path/to/private.yaml`. Install PyYAML in the trusted driver environment only. The private configuration contains `teacher.base_url`, `teacher.model`, and `access.api-keys`, whose first entry is the credential. Configure the intended CLIProxy route explicitly. The driver permits at most nine responses, injects a stale-checkpoint fault, withholds private cases, and requires a final report consistent with the verifier receipt. It does not repair malformed model output or substitute another route.

The driver compares identical binary64 proposal streams under the existing API and the durable client. That check holds the method fixed and establishes selection conformance. A repeated, randomized unfamiliar-agent comparison is still required to measure improvements in autonomy, recovery, or usability. One completed fresh-agent run cannot establish those rates.

Retention thresholds currently apply to absolute loss. To estimate a regression, declare an unchanged control arm and grade both arms on the same private retention cases. The client does not infer an acceptable tradeoff from a candidate's loss alone. The fresh acceptance driver uses one candidate arm and reports absolute retention loss; that fixture does not estimate a capability regression.
