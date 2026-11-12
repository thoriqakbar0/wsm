# Run a repeatable evaluation benchmark

The experimental `wsm benchmark` commands compare unchanged weights, a frozen scripted edit, and an editing policy. They retain independent task and regression grades without training a student. Read the [protocol reference](benchmark-protocol.md) for scoring, configuration, and evidence limits.

Download and open the self-contained [HTML walkthrough](../examples/benchmark/walkthrough.html) for an interactive explanation using the recorded synthetic results. It makes no cloud calls.

## Check the workflow without external services

Create a Python 3.12 environment from the repository root:

```sh
uv venv --python 3.12 .venv
. .venv/bin/activate
uv pip install -e 'packages/python[editing]'
```

Create a tiny random local model and synthetic inputs in an unused directory:

```sh
python scripts/benchmark_fixture.py --output /tmp/wsm-benchmark-inputs
wsm benchmark preflight --config /tmp/wsm-benchmark-inputs/config.json
```

Preflight reports three scheduled trials and zero external calls. It verifies file hashes and partition boundaries without loading model tensors or contacting a provider.

Review the generated `script.py` and `actions.json`. They deliberately damage weights, then restore the synthetic agent's weights from the original snapshot. These are fixture programs with your process permissions. Local mode provides no isolation from private files or other host data.

Run the trusted fixtures into an unused output directory:

```sh
OMP_NUM_THREADS=1 wsm benchmark run \
  --config /tmp/wsm-benchmark-inputs/config.json \
  --output /tmp/wsm-benchmark-run --local-trusted \
  > /tmp/wsm-benchmark-report.json
```

The report labels the run `synthetic` and `exploratory`. The scripted baseline has worse private loss. The synthetic agent's explicit undo restores its baseline loss. This checks mechanics and provides no evidence of learned editing ability.

## Rebuild the report offline

Run report generation without the editing or cloud dependencies:

```sh
wsm benchmark report /tmp/wsm-benchmark-run > /tmp/wsm-benchmark-report-again.json
cmp /tmp/wsm-benchmark-report.json /tmp/wsm-benchmark-report-again.json
wsm benchmark report /tmp/wsm-benchmark-run --markdown > /tmp/wsm-benchmark-report.md
```

The two JSON files match. Reporting verifies retained grade and checkpoint hashes. It does not load the source model, read the original input files, or call a provider. Copy the complete campaign directory to another machine to rebuild the same numeric report. Retain `restricted/` privately; the report command needs its numeric evidence, even though exported reports omit private case text and per-case scores.

## Independently regrade one artifact

Keep the original trusted model, interpreter, dependencies, and numeric environment available:

```sh
OMP_NUM_THREADS=1 wsm benchmark regrade /tmp/wsm-benchmark-run \
  --trial-id completion.agent.0000
```

Regrading verifies the frozen input and runtime fingerprints before loading the artifact. It compares per-case results with the original receipt using the frozen numeric tolerance, or exact equality when no tolerance was supplied. It never repeats an edit or provider request and never erases a prior failure. A frozen artifact whose grading was interrupted can receive its missing grade receipt, while its earlier outcome remains visible.

## Prepare an isolated campaign

Use the generated configuration as a schema example, then supply your own reviewed datasets and scripted baselines. Set `runtime.backend` to `modal`, replace `runtime.actions` with a digest-pinned `runtime.policy` file, and use the existing pinned SmolLM2 snapshot layout. Install the repository's optional `trial` dependencies.

Use the Responses-proxy policy schema from the [trial guide](trials.md), without a student configuration. Match policy turn and output-token limits to the benchmark limits. Keep credentials in the named host environment variables. The benchmark projects only public task inputs into each network-blocked sandbox and independently grades the exported tensors after confirmed termination.

Supply checked conservative prices and an explicit cumulative allocation. `teacher_request_usd` is an upper allowance for one bounded request, including its maximum input and output, not a measured bill. Include any build and shutdown overhead in your checked allowances. Preflight displays the full schedule's worst-case new reservation. A run reserves each trial before dispatch and records `budget_blocked` when the next envelope does not fit. Reservations never imply provider-enforced billing caps.

Run without `--local-trusted` only after approving the allocation and dataset use. There is no automatic retry or continuation of an existing output directory. Inspect incomplete trials and retained provider intents before making a separate decision to run another campaign. A missing terminal record remains a failure or unknown outcome in offline reports.
