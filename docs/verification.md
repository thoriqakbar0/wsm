# Verify the experiment and grading boundaries

## Python package migration

The Python distribution and source package are now `wsm`. No `weightless` import shim is included. Checkpoint and evidence format names, hash-domain inputs, and `WeightlessError` remain unchanged.

All 95 tests passed with `WEIGHTLESS_LOCAL_SANDBOX=1` after the move. The installed wheel was checked outside the source tree for `import wsm`, the installed CLI, bundled task loading, and a real bubblewrap session that accepted an edit, froze, replayed, and reconstructed the selected checkpoint. Its archive contains `wsm/` and no `weightless/` package. After the distribution rename, the same wheel smoke passed with `wsm-0.1.0-py3-none-any.whl`; installed metadata reports the name `wsm`, the SDK version, and the `wsm.cli:main` console entry point.

```sh
WEIGHTLESS_LOCAL_SANDBOX=1 PYTHONPATH=packages/python/src \
  python3 -m unittest discover -s packages/python/tests -q
WEIGHTLESS_LOCAL_SANDBOX=1 python3 scripts/package_smoke.py
PYTHONPATH=packages/python/src python3 scripts/golden_identity.py
```

A frozen run captured before the move replayed after it to the identical checkpoint. Fresh runs before and after the move had identical selected tensors, evaluator configuration, policy, metrics, and reject/accept/reject decisions. Golden checkpoint identity is unchanged. Comparing the moved source against the uncommitted pre-migration snapshot found only resource lookup and sandbox package-path changes in `cli.py`, `local_sandbox.py`, and `sandbox.py`; the existing harness implementation was preserved.

Prepared Harbor packages now contain `wsm/`. Their oracle and trusted grading scripts ran locally with only the generated package on `PYTHONPATH`. Development grading returned reward one, and independent final grading passed all six heldout examples. This verifies package transfer and grader imports, not Docker or native Harbor execution. The optional Harbor dependency remains unavailable on this host.

## Prompt-first WSM harness

On 2026-11-09, all 95 tests passed with `WEIGHTLESS_LOCAL_SANDBOX=1`. This includes the prompt-first CLI experiment through real bubblewrap and systemd execution. Its teacher endpoint was a local scripted Responses server, not a live language model. The experiment inspected coefficients, rejected a worse candidate, accepted an improvement, graded private task and retention cases, reconstructed a standalone binary64 checkpoint, and replayed the evidence. Resume made no extra teacher requests or numerical evaluations. Changing private final data was rejected.

```sh
WEIGHTLESS_LOCAL_SANDBOX=1 PYTHONPATH=packages/python/src \
  python3 -m unittest discover -s packages/python/tests -v
python3 scripts/package_smoke.py
PYTHONPATH=packages/python/src python3 scripts/golden_identity.py
```

Harness tests also cover malformed commands, private-data exclusion, stale observation rejection, exhausted limits, unknown teacher and target outcomes, saved-response recovery, committed-edit recovery, incomplete freezes, uncertain final grading, and restoration of missing terminal artifacts. A CLI regression reproduces a failed initial metadata request leaving an empty session database and proves that resume can initialize it without a development evaluation. Failed session construction now closes its database connection.

The wheel check passed outside the source tree. It exercised the installed `wsm` command, the packaged addition profile, and the existing SDK and replay checks without optional imports. Golden checkpoint identity is unchanged. The optional Harbor smoke check remains blocked by `ModuleNotFoundError: No module named 'harbor'` on this host. This change does not claim a live teacher run, new SmolLM validation, or a hosted service.

## Earlier client and grading checks

The migration checks passed locally on 2026-11-09. Core checks used Python 3.12.3. Integration checks used Python 3.13.11 with Harbor 0.25.0, Verifiers 0.4.1, and JSON Schema 4.26.0. These are synthetic signed-addition results, not research or model-discovery results.

## Direct-editing checkpoint round trip

PR #6's update onto `main` retained the consolidated prompt harness, installed CLI, generic Hugging Face backend, and explicit CUDA path. Atomic checkpoint publication, receipt provenance, incomplete-run exit status, and replay completeness checks were carried forward onto that shared implementation. Merge validation passed 161 Python tests with native sandbox checks enabled and six environment-dependent skips. All six exporter tests passed separately in the model-runtime container, including generic-model and cached SmolLM source-free reloads. The installed-wheel/CLI smoke, stale-build packaging regression, Rust client build, golden identity check, and Harbor schema/task-loading/local-grading smoke also passed. CUDA hardware execution and live hosted providers were not rerun.

On 2026-11-09, `scripts/harness_native.py` completed a real CPU SmolLM2-135M experiment with a scripted scalar edit and an unchanged control. The model revision was `12fd25f77366fa6b3b4b768ec3050bf629380bac`. The numerical image used Python 3.12.16, Torch 2.15.0+cpu, Transformers 5.22.2, and Safetensors 0.8.0. Host orchestration used Python 3.12.3.

```sh
docker build -f examples/client-sandbox/Dockerfile -t wsm-direct-harness:local .
PYTHONPATH=packages/python/src python3 scripts/harness_native.py \
  --image wsm-direct-harness:local \
  --model-source /path/to/cached/models--HuggingFaceTB--SmolLM2-135M-Instruct \
  --output /tmp/wsm-harness-complete
```

The source must contain the pinned snapshot. The output directory must not exist. This command stages a separate read-only model, runs declared public and private cases, and writes evidence and proof files. It requires Docker and enough memory and disk for the staged and exported model.

The run selected a nontrivial final-norm edit using actual measured public loss. It reserved 300 forward calls across the baseline, candidate, public retention, diagnostics, control, and private grading. A separate one-token probe measured two target invocations. Reducing its assigned allowance to one returned `forward_limit_exceeded`. These are invocation counts, not the research's batched-pass units.

The exported model files occupied 541,526,255 bytes before the receipt. A fresh container mounted only the exported artifact and package code, with neither the original source nor its cache available. It loaded the same full numerical checkpoint identity, retained tied embeddings, and reproduced the development metrics and literal generations. The source identity remained unchanged. Separate export checks passed with a relative destination and an invalid caller `PYTHONPATH`; child imports and paths no longer depend on the caller's working directory.

Independent final task and retention grading completed for both arms. Its efficacy conclusion was `inconclusive`; neither arm generated the requested unusual output exactly on the finite heldout task. No acceptable retention threshold was declared. Lower target-conditioned loss is not a behavioral success or a capability-improvement claim. The final-norm edit is an engineering check and differs from the original paper's allowed editing scope.

Post-freeze verification has separate execution costs. The source check, forward-limit probes, serialization checks, and independent reload do not reopen the development budget. CPU, memory, file-size, and wall-time limits apply to the operator container; the exporter itself does not impose a global temporary-disk quota or wall-time deadline.

The configuration-driven CLI also completed a real SmolLM run with the same pinned target, actual scalar selection, and separate private grading. Driver transport tests use a local scripted HTTP responder to check inspection and diagnostic feedback, malformed-response recovery, reserved keys, and private-data exclusion. They do not constitute a live editing-AI discovery run. No live Tinker service round trip was performed; its [capability record](tinker.md) distinguishes SDK evidence from integration proof.

Current local proof files are under `/tmp/wsm-harness-complete-20261109`, including `summary.json`, `evidence.json`, `export-receipt.json`, `reload-proof.json`, `relative-path-proof.json`, and `configured-driver/driver.json`. The rerunnable script is the durable verification artifact; these local output paths are working evidence.

The full suite ran 102 tests with `WEIGHTLESS_LOCAL_SANDBOX=1` and passed with one skipped optional host-model test. All eight exporter tests then passed inside the model-runtime container, including the optional real-model test with genuine evaluation scores. That test checks metadata and reload identity; the separate source-free container check establishes absence of the original source. The installed-wheel smoke and independent golden-identity check passed. Regression coverage includes literal teacher feedback, malformed command recovery, private path separation, reserved keys, failed CLI completion, replayed budget/completeness tampering, relative paths, and concurrent destination preservation.

## Docker-free Linux execution

On 2026-11-09, all 75 tests passed with `WEIGHTLESS_LOCAL_SANDBOX=1`, including 14 real bubblewrap and systemd tests. The host used Python 3.12.3, bubblewrap 0.9.0 from Ubuntu package `0.9.0-1ubuntu0.3`, and systemd 255. No Docker daemon or KVM device was used.

```sh
WEIGHTLESS_LOCAL_SANDBOX=1 PYTHONPATH=packages/python/src \
  python3 -m unittest discover -s packages/python/tests -v
PYTHONPATH=packages/python/src python3 scripts/client_native.py \
  --binary64-only --output /tmp/weightless-local-proof
python3 scripts/package_smoke.py
```

The native checks denied reads of host canaries, host service connections, inherited environment credentials, model writes, and writes to the root and runtime. They exhausted memory, process counts, temporary storage, and both output streams. They checked the actual kernel `memory.max`, `memory.swap.max`, `pids.max`, and `cpu.max` values. Each launched unit and its workload cgroup disappeared after execution, including timeout, overflow, and callback cancellation. A detached child did not survive its workload.

Two additional regressions failed before their fixes and passed afterward. Service-manager expansion changed literal `${PATH}` inside submitted Python source into a manager environment value. Disabling expansion preserves the exact source. A virtual environment under `/tmp` was hidden by the private temporary mount. Mounting the configured environment after that temporary filesystem preserves its imports without exposing sibling host files.

The standalone binary64 driver passed isolation, strict selection, independent private final grading, replay, and reconstruction. It reported `smollm: null` and no fresh teacher run. The wheel smoke passed outside the source tree without optional imports. The [sandbox boundary](sandbox.md) records the threat model, upstream source revisions, and remaining host trust.

The repository-wide `scripts/verify.sh` passed golden identity and its unit phase, then stopped at the optional Harbor check because `harbor` is not installed on this host. Generated Harbor packages were prepared and imported successfully with the new module inventory. Native Harbor execution was not rerun. Torch and Transformers are also absent, so this change does not claim native SmolLM validation.

## Dependency-free behavior and installed package

```sh
INTEGRATION_PYTHON="$PWD/.venv/bin/python" bash scripts/verify.sh
python3 scripts/package_smoke.py
```

The seventeen unit groups passed. They cover unchanged checkpoint golden identity, independent addition predictions across signed inputs, row-major arithmetic, strict proposal validation, stale bases, copied state, foreign or modified candidates, modified results, lazy baseline evaluation, retained baseline failure, genuine graded-zero reward, full request bindings including direct numerical calls, configuration drift, malformed metrics, one-attempt accounting, pending and closed runs, reentrant mutation, training-loss tie-break, development priority, numerical failure, history tampering, bounded history, nested record admission, and oversized evaluator output.

The three example decisions match the pre-migration output exactly. They remain `not_improved`, `improved`, and `not_improved`. Golden checkpoint bytes remain unchanged. The separate identity script encodes the published fixture without importing the SDK.

The package smoke built a wheel, installed it outside the source tree, and ran the new search caller and pure replay. It also replayed a Harbor-produced frozen run without importing Harbor or Verifiers. Pass such a run as an optional argument to `scripts/package_smoke.py` to repeat that check.

## Schema, task loading, and local verifier scripts

`scripts/harbor_smoke.py` validated the JSON Schema, loaded both purpose-specific task packages through actual Harbor and Verifiers models, and checked separate no-network verification and the single declared submission artifact. It checked that the agent build context excludes grader, oracle, and heldout data. The development task excludes heldout data entirely.

The actual trusted final verifier script graded the oracle with reward one and the unchanged network with reward zero. It rejected malformed or duplicate JSON, checkpoint tampering, changed public configuration, a different declared development split, wrong initial checkpoint, wrong policy, and an oversized file. The trusted development script accepted a bound checkpoint submission and rejected a frozen run. These local checks use temporary directory overrides. They do not prove container execution.

## Actual optional adapter in Docker

After installing `packages/python[integration]`, prepare and build the image:

```sh
.venv/bin/python scripts/prepare_harbor.py
docker build -t weightless-signed-addition:local examples/harbor/signed-addition/environment
.venv/bin/python scripts/harbor_native.py --failures --output /tmp/weightless-native-proof
```

The adapter ran the installed Verifiers `HarborTask`, `DockerConfig`, `Deployment`, and isolated `grade_task(on_score=...)` in real Docker. No agent or model rollout occurred. The baseline returned completed native reward zero. The three candidate evaluations returned completed native rewards zero, one, zero, with the original reject, accept, reject decisions and native train/development metrics. Independent final grading of the frozen Harbor-produced run returned reward one with all six heldout examples exact.

Native failure checks passed:

- Nonfinite numerical inference returned `task_failure`, native reward zero, structured failure detail, and verifier exit one.
- A structurally different valid checkpoint returned `task_failure` with native rejected status.
- Malformed diagnostics returned `invalid_response` while preserving authoritative native reward one, raw malformed diagnostics, stdout, and exit zero.
- Valid JSON lists, null values, and nonnumeric metric fields also returned `invalid_response` with native reward one retained. Invalid fields stayed in diagnostic evidence and were excluded from usable metrics.
- Deeply nested verifier diagnostics returned `invalid_response` with native reward one retained as well. When structured evidence cannot fit the envelope, the adapter retains bounded canonical raw JSON and an explicit truncation flag. Run admission checks also reserve the nesting context of the frozen artifact.
- Missing reward returned `invalid_response` with native stderr and verifier exit seven.
- An unavailable Docker socket returned `infrastructure_error` with no invented native reward.

The malformed-diagnostics test initially failed because Verifiers wrapped the callback error in `TaskError`. The adapter now retains capture-error evidence inside the callback and classifies from that evidence. The original failing log remains beside the successful rerun. No retry or score repair was used. Docker had no containers left after the native checks.

Local migration evidence is under `/tmp/weightless-api-update/implementation-proof` and independent verification is under `/tmp/weightless-api-update/parent-proof`. The latter retains failing and passing request-binding, malformed native JSON, and declared split controls. The implementation's `native-final.log` retains the original wrapped callback failure. Unit, schema, wheel, image-build, decision-equivalence, and cleanup evidence remain alongside native JSON artifacts. These paths are local working artifacts.

The parent independently checked 1681 signed predictions and all four training/development loss states against Python integer addition and exact fractions. Current native evidence is in `parent-proof/native-complete`, with the current wheel replay in `parent-proof/wheel-complete.log`. The replay Python had neither Harbor nor Verifiers installed.

## Independent native Harbor CLI

```sh
.venv/bin/harbor run --path examples/harbor/signed-addition --agent oracle --env docker \
  --force-build --jobs-dir /tmp/weightless-jobs --job-name oracle
python3 scripts/check_harbor_reward.py /tmp/weightless-jobs/oracle
```

The native CLI oracle trial passed with reward one and zero trial exceptions. A second native CLI task copy whose solution wrote the unchanged frozen run passed through grading with reward zero and zero trial exceptions. Its diagnostics reported graded status, so this is a genuine task-solving zero. Evidence is in `implementation-proof/jobs/api-oracle` and `implementation-proof/jobs/api-unchanged`.

GitHub Actions remains disabled. The offline tutorial rendered at desktop and phone widths without console errors. Its controls remain an explanatory browser calculation, while its Python callers use the actual SDK. Replay verifies unsigned consistency and checkpoint reconstruction. It neither authenticates external scores nor reruns behavioral evaluation.

## Durable client checks

The standard unit suite includes session, verifier, sandbox, and no-tools transport regressions. `scripts/package_smoke.py` imports the new client from a built wheel without Torch, Transformers, Harbor, or Verifiers. The [native client guide](client.md#reproduce-the-native-checks) runs actual Docker isolation and pinned SmolLM editing. The [acceptance specification](design/agent-client-validation.md) retains broader scientific-validity and unfamiliar-agent comparisons as distinct requirements.


## Consolidated implementation verification

The [integration receipt](design/integration-verification.json) records the local combined checks and artifact hashes. The native suite ran 156 tests successfully with six optional model-runtime skips; separate model-environment runs passed 14 generic-target tests, four export tests, and 13 worker tests. Installed wheel, real PTY lifecycle, CLI interruption/resume, Rust bridge parity, and retained evidence replay passed.

A fresh live teacher run initially exhausted its budget while alternating inspections. The harness now reconstructs bounded inspection memory for the current arm and checkpoint from completed session operations. The repeat run selected one edit, reached zero development loss, and passed private grading on the declared synthetic signed-addition cases. This is one live interaction check, not a general model-improvement result.

The combined worker also passed a real Modal T4 smoke with finite CUDA outputs, selection, replay, and reconstruction. Sandbox termination was confirmed. The current worker, session, sandbox, and verifier hashes match that receipt. No live hosted Tinker integration is claimed.

The model export tests force acceptance to isolate changed-state export and fresh-process reload. Their loss fixture is not efficacy evidence. Real measured selection is exercised separately in the CUDA smoke and live synthetic experiment.
