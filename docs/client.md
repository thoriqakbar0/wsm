# Durable local client

The local client adds `Session`, `Verifier`, `DockerRunner`, and `ResponsesTeacher` beside the existing `SearchRun` API. The existing checkpoint and run formats retain their meaning. Session exports use a separate format.

## Ownership and supported execution

A trusted host owns the SQLite session, private verifier registry, teacher credentials, and Docker socket. An untrusted agent receives an arm-scoped JSON observation and submits commands through `Session.dispatch`. Do not give the agent a Python session object, host shell, database path, or Docker socket. These objects are privileged orchestration APIs.

`DockerRunner.probe` executes agent bridge code without host mounts or network access. The acceptance driver passes model responses through this isolated JSON bridge before dispatch. The numerical worker runs in another container and receives only the pinned public model directory as a read-only mount. Both containers use an unprivileged UID, a read-only root filesystem, no network, no capabilities, no new privileges, bounded processes and memory, and a bounded temporary filesystem. The host enforces deadlines and removes containers after execution. Teardown can take an additional bounded grace period. Cancellation freezes selection and reports that remote termination is unconfirmed.

The current numerical backends are binary64 affine/ReLU checkpoints and the CPU float32 `HuggingFaceTB/SmolLM2-135M-Instruct` checkpoint at revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`. The worker pins greedy generation, eager attention, one CPU thread, offline loading, and no remote model code. It records parameter aliases, runtime identity, literal generations, and sandbox observations. Scalar replacements preserve tied embeddings. Unsupported overlapping storage views fail closed. `stage_smollm` dereferences a cached snapshot into a new self-contained directory. It never changes the source cache.

This profile trusts the host administrator, Docker daemon, image, and numerical runtime. It does not protect against a compromised kernel or runtime. The private verifier database must remain outside every agent container. Docker isolation is enforced by the runner, not by Python object permissions.

## Session contract

`Session.create(path, runner=..., target=..., declaration=..., development=..., verifier=..., final_dataset=...)` binds the target, runner configuration, public cases, and immutable declaration. `Session.open` resumes the persisted state. Each arm has its own selected checkpoint, observation version, pending operation, and history.

A declaration contains `hypothesis`, `arms`, canonical `editable` tensor names, `max_edits`, `max_inspect`, positive `wall_seconds`, and integer `budgets`. Budget names are `attempts`, `evaluations`, `inspections`, `teacher_requests`, and `final_evaluations`. Optional `diagnostics` also consumes the evaluation budget. The current selection policy minimizes development loss strictly. Optional `development_retention_loss_max` requires public retention cases. Model families, resource ceilings, and acceptable regressions remain operator choices within the implemented profile.

`observe(arm)` returns current identity fields, permitted parameter metadata, public cases, history, remaining counts, and phase-appropriate actions. `inspect` requires a checkpoint, tensor, and bounded index list. A submission contains exactly `session`, `arm`, `declaration`, `observation`, `base`, and `edits`. Each edit contains `tensor`, `index`, and finite `value`. Every numerical operation has an idempotency key. Repeating the same key and request returns the same outcome. Changing a request under an existing key is an error. A stale checkpoint or observation is rejected before candidate evaluation.

The first submission establishes an unchanged baseline. Candidate materialization verifies the base identity before evaluation. Effective float32 replacements are recorded after rounding, so replay reproduces the resulting weights. Each candidate starts from copied base weights. Rejected edits never replace the selected checkpoint. Failures and uncertain outcomes remain in the journal. An uncertain external operation is never silently resent.

`freeze(key=...)` records a cutoff, drains durable results that precede it, and prevents later selection. An incomplete freeze cannot finalize. `cancel` uses the same selection barrier. `finalize(key=...)` grades only frozen candidates through the separately bound verifier. `export()` contains the declaration, operations, literal public observations, journal, frozen selection, and final receipt. It excludes private cases and teacher credentials.

## Evaluation and conclusions

The verifier registry stores private datasets separately from the session. A dataset can contain `task` and `retention` cases. The registry binds grading to the frozen runner and target identity, serializes concurrent grading, and records release intent before disclosing a receipt. An interrupted grading operation becomes a stable unknown outcome instead of a fresh hidden evaluation. Dataset claim identities use a private keyed digest.

Target-conditioned language loss and exact free generation are different measurements. A lower loss does not establish the intended behavior. `diagnose` records literal generation, EOS behavior, and loss under the same resource budgets. Retention grading reports unrelated language loss. A single retention fixture is a compatibility check, not a comprehensive capability assessment.

An exploratory declaration leaves criteria unset and produces an inconclusive efficacy conclusion. A confirmatory declaration requires both `criteria.final_exact_min` and `criteria.retention_loss_max`. The verifier checks both on the frozen finite cases. Its supported or bounded-negative conclusion applies only to that declared experiment. It does not establish general weight-space programming or authenticate the runtime's arithmetic.

`replay_session(bundle)` checks journal consistency and reconstructs selection without executing a model. `verify_session(bundle, runner=..., target=...)` independently rematerializes candidate checkpoints and checks numerical identities. It does not independently re-evaluate every recorded score. Hashes detect inconsistency; they do not authenticate a producer. A separate operator can regrade fresh private cases with a trusted runtime.

## Teacher transport

`ResponsesTeacher` receives an explicit endpoint and model ID. It never selects a fallback. The broker sends `tools: []` and `tool_choice: none`, checks the completed response identity and declarations, and rejects observed tool invocation events. It retains raw request bodies, stream events, responses, and failed or uncertain sends in a private directory. Authorization headers are excluded. These checks establish the observed API boundary, not upstream execution or billing attestation.

Teacher requests reserve session budget before dispatch. Session identity namespaces idempotency artifacts. A saved response can be replayed after a crash. A possible send without a saved response cannot be repeated automatically. A separately charged report request may orchestrate final grading after freeze or summarize a closed experiment without reopening selection.

## Reproduce the native checks

Build from the repository root with Docker available.

```sh
docker build -f examples/client-sandbox/Dockerfile -t weightless-client:local .
PYTHONPATH=packages/python/src python3 scripts/client_native.py \
  --image weightless-client:local \
  --model-source /path/to/cached/model-repository \
  --output /tmp/weightless-native-check
```

The output directory must not exist. To resume a teacher acceptance session after a driver interruption, repeat the command with `--resume` and the same directory. The session retains its original deadline, counters, and responses. A possible prior send remains unknown and cannot be retried automatically. The driver tests isolation, timeout cleanup, binary64 final grading, current-client selection conformance, SmolLM aliases, float32 edit replay, restoration, and private final grading. It retains bundles and failure records. `--docker` and `--docker-host` select an explicit Docker installation.

For a fresh teacher run, add `--teacher-config /path/to/private.yaml`. Install PyYAML in the trusted driver environment only. The private configuration contains `teacher.base_url`, `teacher.model`, and `access.api-keys`, whose first entry is the credential. Configure the intended CLIProxy route explicitly. The driver permits at most nine responses, injects a stale-checkpoint fault, withholds private cases, and requires a final report consistent with the verifier receipt. It does not repair malformed model output or substitute another route.

The driver compares identical binary64 proposal streams under the existing API and the durable client. That check holds the method fixed and establishes selection conformance. A repeated, randomized unfamiliar-agent comparison is still required to measure improvements in autonomy, recovery, or usability. One completed fresh-agent run cannot establish those rates.

Retention thresholds currently apply to absolute loss. To estimate a regression, declare an unchanged control arm and grade both arms on the same private retention cases. The client does not infer an acceptable tradeoff from a candidate's loss alone. The fresh acceptance driver uses one candidate arm and reports absolute retention loss; that fixture does not estimate a capability regression.
