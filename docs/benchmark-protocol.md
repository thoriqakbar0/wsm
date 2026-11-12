# Benchmark protocol reference

`wsm.benchmark.v1` freezes a sequential comparison schedule before execution. The CLI is experimental. The supported top-level Python SDK exports remain `WeightModel` and `execute_program`.

## Ownership and artifacts

The campaign controller owns the manifest, reservations, trial outcomes, and private grading receipts. `Trial` owns public action history and checkpoint progression. Every successful edit remains current, including worse edits. The benchmark never selects a checkpoint using private grades and never trains a student.

`manifest.json` contains public identities, counts, digests, code and environment fingerprints, limits, statistics, and the complete schedule. `seal.json` binds its digest. `restricted/config.json` binds retained input paths and reviewed configuration. Inputs are copied under content digests in `restricted/inputs/` before dispatch. Files use owner-only permissions. These hashes detect inconsistency; they do not authenticate evidence against a malicious host owner who can rewrite the entire campaign.

Each `trials/<id>/` contains a durable reservation, dispatch intent, public trial history when applicable, final safetensors, an editor-closure record, and an outcome. `restricted/<id>/grade.json` binds the independent private and regression grades to the manifest, trial identity, and artifact hash. Detailed errors remain restricted. Public reports omit arbitrary worker output, private texts, and per-case scores.

Missing outcomes are represented as `outcome_unknown` when a dispatch intent exists and `failed` otherwise. Completed scores require intact receipts and artifacts. A malformed retained receipt or missing completed artifact stops report generation rather than presenting its score as valid. Earlier outcomes remain unchanged during regrading. A recovered grade receipt is inspection-only when the original outcome failed or is missing; it does not retroactively enter completed-only summary statistics.

## Configuration

The synthetic fixture generator produces a complete example. Relative input paths resolve against the configuration file. File references have `path` and `sha256` fields. Hashes are lowercase SHA-256 hex strings. A model reference has a directory path, model ID, revision, and a SHA-256 identity of the sorted relative-file-name to file-digest mapping.

| Field | Required contents |
| --- | --- |
| `format`, `name` | Format `wsm.benchmark.v1` and a safe benchmark identifier. |
| `model` | `path`, `id`, `revision`, `sha256`. Modal requires the trial guide's pinned SmolLM2 model and `snapshots/<revision>` layout. |
| `provenance` | Full `code_revision` and digest-pinned `dependency_lock`. The manifest additionally records the actual package source digest and installed versions for both the controller and the independently probed grader interpreter. The supplied revision is attribution; the package digest identifies executed source, including uncommitted changes. |
| `runtime` | `backend`, `grader_python`, and either digest-pinned synthetic `actions` or Responses-proxy `policy`. Proxy endpoints stay in restricted configuration; the public manifest stores policy settings and a route fingerprint. |
| `limits` | Positive integer `turns`, `evaluations`, `output_tokens`, `wall_seconds`, `command_seconds`, and `grader_seconds`. |
| `budget` | `max_usd`, `prior_reserved_usd`, `price_checked_on`, and `rates` containing `build_usd`, `sandbox_hourly_usd`, `egress_per_gib`, `teacher_request_usd`. Modal prices must be positive. |
| `statistics` | Nonnegative integer `seed` and `bootstrap_samples` between 100 and 100,000. |
| `thresholds` | Optional object with nonnegative `improvement`, `regression`, and `numeric_tolerance`, or null. No efficacy threshold is invented when missing. |
| `review` | Boolean `data_rights`, `semantic_overlap`, `power`, plus `owner` and `private_artifact_store`, each a nonempty string or null. These record operator decisions, not automated certifications. |
| `regression` | Digest-pinned nonempty unrelated completion cases. |
| `tasks` | Unique task `id`, `family`, `instruction`, and digest-pinned `public`, `private`, `script`. One frozen script digest per family. Cases contain exactly nonempty `prompt` and `target` strings. |
| `repetitions` | Positive integer number of agent trials per task. Unchanged weights run once per task. Scripted baselines run once by default. Set each task's optional `scripted_stochastic` boolean to true to repeat it on the agent schedule. The script digest and stochastic setting must agree within each family. The setting is an operator declaration; preflight does not prove that a script is deterministic. |

Preflight rejects normalized exact duplicates between public, private, and regression partitions, including Unicode NFC normalization and trimmed edges. It rejects private inputs inside the model or package image inputs, including symlink references. Semantic near-duplicates and contamination require operator review. Isolated preflight also requires the configured model and credential environment variables to be nonempty. It does not authenticate the provider.

`evaluations` caps explicit public evaluation actions. Arbitrary Python can perform additional calculations or invoke evaluation inside a file. Its computation is bounded by command and sandbox timeouts, not an authenticated count of internal model calls. Turns bound action dispatches. Each live request has the existing proxy's prompt-byte ceiling and a maximum output-token allowance. Synthetic tapes use byte allowances and do not represent provider token usage.

## Scoring

The trusted grader uses the original configuration and tokenizer, validates exported tensor names, shapes, allowed dtypes, and finiteness, and loads only inert tensors. It runs in a separate process after the editor closes. Agent configuration and Python never enter that process. Local fixture execution uses host permissions and cannot establish private-data isolation.

Each prompt and target is tokenized separately with special tokens disabled. The grader computes mean target-token NLL for each case, then averages cases equally. Reports also include target-token counts and token-weighted NLL. Pristine-score comparisons and regrading require exact equality when no numeric tolerance was configured.

Task improvement is baseline NLL minus agent NLL. Regression is agent NLL minus unchanged NLL on the unrelated cases. Each agent trial is paired with unchanged weights and its scripted baseline. Stochastic scripted baselines pair by task and repetition. Deterministic scripted baselines share one result per task. The summary averages repetitions within tasks, tasks within families, and families equally. Median, range, and population standard deviation describe completed paired repetitions.

The seeded 95% hierarchical bootstrap resamples tasks within each fixed family and paired repetitions within each selected task. It preserves family and task weighting. It does not treat repeated evaluation of identical cases as new independent tasks. Failed trials have no fabricated score; summaries explicitly use completed pairs. The success denominator includes every scheduled agent trial.

A successful individual edit meets the frozen improvement threshold against unchanged weights and the regression tolerance. A positive campaign decision additionally requires complete evidence, a lower confidence bound above the improvement threshold against both baseline arms, and an upper regression bound no greater than the regression tolerance. A bound entirely on the failing side yields a negative decision. Remaining cases are inconclusive.

All local synthetic campaigns are exploratory. Missing thresholds, missing operator reviews, insufficient coverage, or at most three tasks per family also force exploratory labeling. Coverage floors are two families, three tasks per family, 20 public and 50 private cases per task, 100 regression cases, and five agent repetitions. Those floors do not establish statistical power. More tasks and an explicit power review are required for confirmatory labeling.

## Cost and failure semantics

A live editing trial reserves a build allowance, the complete sandbox resource window, a 600 MiB checkpoint payload plus an 8 MiB diagnostic allowance and one overflow-detection byte per possible successful Python action, and the maximum number of teacher requests for the agent arm. Checkpoint export rejects an initially oversized regular file before sending its contents. The controller independently caps binary output, terminates an oversized producer, and removes partial staging files. Publication requires a successful process exit and a complete export receipt on the diagnostic stream. The controller compares the receipt's byte count and SHA-256 with the bytes it actually received before publishing. Missing, truncated, or mismatched receipts reject the transfer, including SDK timeouts presented as ordinary EOF. Receipts check transfer completeness; the independent grader still validates agent-controlled tensor contents. Successful transfers publish without overwriting an existing destination. SDK framing, in-flight network buffers, and provider billing are not hard-capped by the application payload limit. Unchanged and trusted local arms make no provider reservation. Each reservation is durably recorded before dispatch and remains charged to the allocation after uncertain outcomes. Actual billing is reported as unknown and unreconciled until separately established; zero local reservations do not imply a free hosted service.

Terminal statuses are `completed`, `invalid_artifact`, `failed`, `timed_out`, `budget_blocked`, and `outcome_unknown`. An exception before a durable frozen artifact may leave uncertain editor effects, which take precedence over a timeout classification. A timeout during independent grading has a known frozen artifact and uses `timed_out`. No uncertain external operation is replayed automatically.

Preflight and report generation make no provider calls. Live Modal provisioning, proxy requests, provider cleanup behavior, data suitability, and efficacy remain separate verification steps requiring an explicitly funded pilot.
