# Agent-facing experiment client

Status: reviewed design with an implemented local subset. The [local client reference](../client.md) states current APIs, supported targets, enforcement, and limits. Requirements below define the broader conformance target. They are not all implementation guarantees. The [validation specification](agent-client-validation.md) separates implemented checks from the remaining acceptance program.

## Purpose and recommendation

An unfamiliar agent should be able to state a hypothesis, conduct an authorized weight-editing experiment, recover from interruptions, and export evidence supporting an appropriately limited conclusion. The operator should not have to reconstruct request counts, repair state, or notice that an improving score measures a different behavior.

Recommend a **durable local experiment kernel**, with a separately authorized target runner and independent final verifier. The agent-facing client is a small interface to that kernel. The kernel owns experiment state, permitted observations, operations, accounting, and selection. It does not own the research strategy or assume that an evaluator tells the truth.

This is a recommended ownership model, not a decision to build a daemon, hosted service, general agent framework, or particular model integration. Actual isolation requires restricted identities, storage, and network access. A library in the agent's unrestricted process can offer cooperative bookkeeping but cannot enforce confidentiality against that agent.

## What the pilots establish

Evidence identifiers below refer to records in [agent-client-evidence.json](agent-client-evidence.json). Its local source locators and digests are provenance hints. The extracted observations remain readable without those temporary files; neither the index nor its hashes authenticate the original producers.

| Evidence | Reconstructed decision history | Attribution and design consequence |
| --- | --- | --- |
| E1. Prepared addition | One recorded completed teacher response supplied two output coefficients for an existing addition feature basis. The recorded final set had seven exact answers. | Narrow compatibility evidence. A prepared representation and one response do not establish repeated discovery. Completed proposals and transport requests need different counters. |
| E2. Terminal-agent baseline | The oracle completed. An unchanged target timed out, then a separate longer attempt produced a malformed command. No weight proposals were made. | Target/runtime/harness precondition failure. A native zero reward is insufficient to conclude that weight editing failed. |
| E3. Normalization pilot | A clean first proposal was evaluated. The next request accidentally included random-control feedback and was interrupted. The clean response was reused; the contaminated slot still counted. Eight clean proposals occupied nine request slots. All three selected teacher vectors were identical. | A temporary harness disclosure bug, not a failure of the SDK's existing result binding. Allowlisted views, durable recovery, separate counters, and candidate identities should eliminate manual repair. |
| E3. Final behavior | Normalization edits lowered target loss but produced no exact silly answers. Unrelated reference loss increased for the teacher candidate. | Metric and edit-space limitation. Report selection, behavior, and retention independently. Three arm records are not three distinct learned programs. |
| E4. Activation-guided rows | Training activations constructed directions for nine tied embedding rows. Development sensitivity probes informed coefficient choices. A scripted control and matched random directions were evaluated. Nine proposals were valid; eight final arm records represented seven distinct candidates. | A more informed research method, not a client usability comparison. Basis construction, probes, ties, control dependence, and cache reuse belong in the experiment identity. |
| E4. Final behavior | Strongly improved correct-prefix token loss coexisted with repetition, premature stopping, wrong animal answers, no exact final answers, and larger unrelated reference loss. | Feedback needs generation diagnostics and named metric conditioning. The small synthetic retention set does not measure broad language ability. |

The current SDK already owns copied candidates, strict proposal admission, one pending candidate, bound evaluator results, strict lexicographic selection, bounded history, and pure replay. See [the current contract](../contract.md) and `SearchRun` in [core.py](../../packages/python/src/wsm/core.py). It directly uses the checkpoint and patch operations in [binary64.py](../../packages/python/src/wsm/binary64.py), whose graph is affine/ReLU and whose parameter count is bounded. Changing only the evaluator does not add transformer support.

The SmolLM scripts bypassed that SDK lifecycle and implemented their own. This motivates a reusable contract; it does not prove that the current SDK violated its documented guarantees. Durable external operations, model-scale inspection, target capability negotiation, and protected observation projections are proposed extensions. The experiments also do not show that any client design improves agent performance yet.

## Experimental contract

The following requirements precede API names and deployment choices. `MUST` denotes a proposed conformance invariant, not an existing implementation guarantee. A required capability that is unavailable causes admission refusal or an explicitly weaker execution profile, never an implied guarantee.

### C01. Declaration and claim

An immutable declaration MUST bind the hypothesis, task population, operational target behavior, search method, editable region and operations, control definitions, trial units, information policy, metric definitions, selection policy, conclusion rule, execution identities, and resource policy. Derived activation bases, sensitivity probes, gradients when allowed, and diagnostic interventions are parts of the method.

Drafts may contain unresolved choices. Before dispatch, all resource and access permissions required for the selected execution profile MUST be resolved. Efficacy and retention thresholds may remain explicitly unset in an exploratory experiment. That experiment may report descriptive outcomes but cannot certify threshold-based success or acceptable regression. A confirmatory claim requires its criteria and uncertainty rule before final exposure.

A method, permission, basis, policy, or threshold change after observations creates a linked new declaration. Earlier disclosures and spent datasets remain in its lineage. A new identifier does not erase information already learned.

### C02. Principals and permitted information

Declare the operator, experiment agent, proposal teacher if separate, control owners, runner, and verifier. A model may fill multiple roles only where the information policy permits that sharing. The pilot transport profile continues to use a tool-disabled proposal teacher; it does not imply that a separate experiment controller has no host tools.

Information authorization is a table over **principal, arm, split, operation, and phase**. It specifies whether examples, tensor values, activations, gradients, logits, local sensitivities, generated text, and diagnostics may be disclosed. It MUST distinguish adaptive development retention probes from independent final retention data. Hidden development examples repeatedly queried through metrics are adaptive feedback, not an untouched final set.

Agent observations MUST be constructed from allowlisted domain fields. They MUST NOT be an all-arms history with selected keys removed. Each disclosure records its authorized source references, projection version, exact delivered bytes or bounded digest-backed artifact, and accounting operation. Other-arm scores, final examples, private retention cases, unrestricted paths, evaluator source, and raw privileged logs are unavailable unless expressly authorized for that phase.

Enforced mode requires the agent to lack access to privileged memory, files, credentials, arbitrary runner execution, and bypass network routes. Distinct processes on an unrestricted shared account are insufficient. Cooperative mode labels those restrictions advisory. Host administrators and the declared trusted execution components remain in the trust boundary. Canary tests establish tested-path coverage, not absence of all timing or inference channels.

### C03. Target, inspection, and numerical identity

A target manifest MUST identify actual weight content, topology, parameter storage and aliases, configuration, tokenizer, preprocessing, chat template, runtime, numerical format and rounding, decoding, and relevant stochastic settings. Friendly model names and mutable paths are labels. Provider-reported model names are observations, not independently established implementation identity.

A capability description declares supported patch forms, canonical addressing, inspectable regions, shapes, dtypes, quantization and sharding limits, alias rules, reset behavior, and reproducibility level. No model family is selected by this document. Unsupported combinations fail before an evaluation reservation or target execution.

Inspection uses typed, bounded queries over declared parameters and public case identifiers. It does not execute arbitrary agent code, expressions, or filesystem paths. Snapshot and observation identifiers make stale inspection explicit. Large arrays use scoped artifact references and paged slices, not full-checkpoint prompt dumps. Derived probes spend their declared diagnostic and compute budgets.

### C04. Proposals and canonical materialization

A proposal MUST bind its session, arm, incumbent checkpoint, declaration, edit-space identity, and observation version. It declares whether values replace the incumbent directly or are coefficients relative to an identified original and immutable basis. Basis provenance binds source checkpoint, permitted data split, construction code, arithmetic, and resulting bytes.

Admission validates the complete proposal atomically. Unknown fields, stale bases, nonfinite values, forbidden addresses, duplicate or overlapping canonical writes, conflicting aliases, unsupported formats, and exceeded bounds reject before materialization. Teacher-supplied executable code is not a patch format. Effective rounded no-ops are recorded without inventing a new distinct candidate.

The runner materializes exact effective tensor values and records a canonical patch plus resulting state identity. Tied input/output embeddings are one storage region with two uses; changing either alias changes both. A backend cannot silently untie, cast, reorder, or change architecture. Alias and layout identity are part of conformance. The candidate identity binds full target state, not just the inspected slice or coefficient vector.

### C05. Candidate ownership and isolation

The runner owns immutable source state and isolated candidate state. The agent never edits a shared live incumbent. Each arm/trial has independent state and selection history. A worker reused for efficiency MUST reset weights, aliases, caches, RNG, hooks, mode, and other declared mutable runtime state before another candidate. Reset verification failure quarantines the worker and its unverified results.

Start with one pending candidate per arm as the reference ordering rule. A future concurrent implementation must preserve an explicit selection order and isolated materialization; concurrency is not permission to race incumbent updates.

### C06. Baseline, evaluation, selection, and retention

Each result MUST bind session, arm, attempt, full candidate identity, evaluator, data split, numerical execution settings, seed where meaningful, and purpose. A usable baseline must satisfy the declared execution protocol. Genuine graded zero remains valid evidence. A wrong silly answer can be a legitimate task zero; an invalid terminal-command envelope can fail an explicitly declared agent-usability gate. Neither is automatically an infrastructure failure.

Preserve distinct outcomes for completed grading, target/task failure, infrastructure failure, invalid response, and unknown external completion. Selection cannot use a failure or invented replacement score. The configured selection rule and tie behavior are pure functions of retained, eligible development evidence. Existing strict lexicographic selection is one supported policy model to preserve where chosen.

Metrics MUST name units, conditioning, normalization, aggregation, stopping rules, and split. In the pilots, target loss was the mean of per-case mean target-token loss with correct prefixes and EOS. Free generation did not receive those correct prefixes. Its exact-match normalization, decoding, and token cap are separate parts of the measurement. A selected candidate is not automatically a successful program.

A predeclared development-retention constraint may make a candidate ineligible. Final-retention data cannot be used for development selection. With retention acceptability unset, report measured regression and `acceptability=undetermined`; never insert a permissive default or invent a pass threshold.

### C07. Controls and repeated evidence

Control manifests MUST state what comparison they identify. Teacher-selected coefficients combined with random directions constitute a **dependent matched control**. They do not measure an independently optimized random-search algorithm. The activation-guided pilot also supplied a learned basis and probe results; a scripted no-teacher control is needed to isolate value beyond those supplied ingredients.

Record request slots, usable proposals, admitted attempts, materialized candidates, distinct states, unique executions, cache reuses, and trial/arm records separately. Repeated deterministic evaluation of identical states is not independent replication.

An evaluation cache key MUST include candidate state, runtime, evaluator and split identities, generation settings, seed, and any stochastic replicate identity. Reuse requires declared deterministic equivalence or an authorized stochastic sampling rule. A new bound receipt references reused evidence; it does not pretend another dispatch occurred. Cache metadata and cross-arm identifiers remain private where the arm policy requires it.

### C08. Resource accounting

Budget policy is a vector. Potential dimensions include teacher client requests, valid candidate attempts, target and diagnostic evaluations, control work, basis construction, input/output tokens, compute, elapsed deadline, bytes, storage, and cost. The operator chooses applicable ceilings and whether each is enforced, estimated, or observed-only.

Before side effects, atomically persist an operation identity, immutable request hash, and conservative reservation. Reservations transition to known spend, retained unknown exposure, or a justified release. A teacher request that produces invalid JSON still spends request budget. Rejection before candidate admission preserves the current SDK's no-candidate-attempt charge. Every diagnostic and public probe is accounted work.

Budget exhaustion cannot be hidden by resetting a counter, creating another run ID, or resuming a process. Imported or replayed evidence does not create another live teacher sample. If remote retries or usage cannot be bounded, a required hard financial or token ceiling prevents admission for that route. Otherwise the report labels the dimension uncertain or advisory. A local reservation alone does not constrain provider billing.

### C09. Operations, cancellation, and recovery

Session phase and operation status are separate domains. The reference session phases are `draft`, `developing`, `freezing`, `frozen`, `finalizing`, and `closed`. An operation is `reserved`, `dispatched`, `completed`, `committed`, `cancel_requested`, `cancelled_confirmed`, `outcome_unknown`, or `rejected`. These are proposed contracts, not additions to current run-v2 wire records.

| Session transition | Guard and authority |
| --- | --- |
| `draft -> developing` | Kernel admits capabilities, information and budget policy. The baseline is a separately recorded prerequisite; its absence is not a fabricated zero. |
| `developing -> freezing` | An authorized freeze or stop persists a barrier against new development proposals and dispatch. Separately authorized final dispatch remains possible after a complete freeze. |
| `freezing -> frozen` | Kernel fixes selection and completeness, preserves unresolved obligations, and verifies candidate references. Incomplete freezes are exportable but cannot finalize under this reference profile. |
| `frozen -> finalizing` | Complete eligible comparison and declared verifier authority. Required confirmatory criteria are fixed before exposure. |
| `finalizing -> closed` | Final receipt is committed, or bounded shutdown produces an explicit unknown/failed/inconclusive report. |
| Any active phase `-> closed` | A declared terminal refusal or failure permits partial export. It never erases reservations, disclosures, or unknown external operations. |

There is no transition back from freeze to development. The journal may receive late-evidence supplements after closure, but those cannot change the frozen artifact, reopen search, or silently rewrite the original report.

The reference profile uses a **durable-result cutoff**. The development freeze barrier has journal sequence `K`. Completed evaluation results durably recorded before `K` remain eligible. During `freezing`, the kernel must finish their pure selection decisions in the declared arm/attempt order before committing the frozen artifact, even if a crash interrupted selection. Results received but not durable before `K`, and results first committed after `K`, are ineligible supplements. Eligibility depends on journal order, not a provider timestamp or when a caller happened to receive bytes. No development inference is dispatched while draining those pure decisions. Another drain/cutoff policy would require an explicitly versioned declaration/profile and its own conformance oracle; an implementation cannot choose opportunistically after seeing scores.

Commands carry session-scoped idempotency keys. Same key and request content returns the same operation; changed content under the same key is a conflict. Persist dispatch intent before sending, commit results before selection, and apply selection once. This provides local commit discipline without promising exactly-once external execution.

| Interruption point | Required recovery |
| --- | --- |
| Before durable reservation | No authorized dispatch. Repeating the command may create its reservation. |
| After reservation, before possible send | If non-dispatch is provable, continue or release according to policy. If not, retain unknown exposure. |
| After possible send, before durable response | `outcome_unknown`. Query a trustworthy job/status endpoint when supported. Do not resend automatically. |
| After durable result, before decision | Complete pure selection from that result. If a freeze barrier exists, apply the durable-result cutoff before committing its artifact. Do not call the evaluator again. |
| After committed decision | Return the existing receipt. Never select or charge twice. |
| After frozen-artifact commit | Preserve late evidence in a separate linked supplement. Never change frozen selection. |

Cancellation first prevents new work in its authorized scope, then requests termination. Acknowledged cancellation and proven non-execution are different facts. Unknown requests retain reservations. A replacement remote attempt needs a new authorized reservation and records possible duplicate execution. Cross-restart upstream idempotency is used only when the provider actually guarantees its scope and retention period.

Deadlines stop new dispatches and initiate supported cancellation. Hard termination of local compute requires actual supervisor quotas. Remote termination is best effort unless the route proves otherwise. On restart, inability to establish a trustworthy remaining deadline blocks new dispatch. The client must still export an honest partial result within its own bounded shutdown/report allowance.

### C10. Freeze and final exposure

Freeze stops proposal admission and dispatch, fixes the selected completed candidate for every declared arm, and records unresolved work and accounting. Final grading is admitted only for a complete eligible freeze satisfying the declaration's baseline and comparison requirements. An incomplete freeze remains exportable and supports an inconclusive report, not a silently repaired confirmation.

All arms in a confirmatory comparison freeze before any of their final results are disclosed. A final dataset has an exposure identity and persistent spent status. Any released final example, score, diagnostic, or error information marks it exposed; development cannot resume on it. A newly declared experiment does not reset that status. Adaptive next work needs fresh evaluator-owned confirmation data with declared lineage.

Before possible final execution, the verifier atomically claims the canonical dataset/exposure identity for the declared frozen comparison or predeclared cohort. The claim binds member freezes, allowed grading/re-evaluation purposes, operations, release schema, and attempt allowance. It survives kernel/adapter restarts and session IDs. A second session cannot treat a claimed dataset as unused. Paired cohort membership cannot be expanded after outcomes are disclosed. The trusted evaluator owns identity/lineage registration; hashes alone cannot detect every semantic duplicate or an administrator re-registering spent data under a misleading name.

| Verifier dataset state | Consequence |
| --- | --- |
| `unused` | May be atomically claimed for one declared confirmation cohort. |
| `claimed` | Bound to that cohort before dispatch; competing unrelated claims fail even before disclosure. Private evaluation alone is not public exposure. |
| `evaluation_unknown` | Execution may have occurred. Only safe status/receipt reconciliation within the existing claim is allowed. |
| `disclosure_unknown` | Result bytes may have reached an adaptive actor. Treat as spent for new confirmation and development, even if acknowledgement was lost. |
| `exposed` | A declared result was released; the dataset is permanently spent for adaptive development or a fresh confirmation claim. |
| `retired` | Abandoned or unresolved claim remains unavailable for unrelated new confirmation. |

Persist release intent before crossing the disclosure boundary. A crash or lost acknowledgement after that point produces `disclosure_unknown`, not `unused`. The kernel separately tracks operation completion; an unknown evaluation and an unknown disclosure are different obligations. A claim can return to unused only on trusted evidence that dispatch and release never occurred, with all outstanding authorizations revoked or drained. The verifier serializes claim changes and checks the current claim at execution and release, so delayed jobs cannot execute under a superseded authorization. There is no automatic release on timeout. An unresolved or privately completed abandoned claim is retired under this conservative profile, rather than reassigned opportunistically.

The verifier releases only the declared bounded result schema after freeze. Raw final cases, labels, paths, private diagnostics, and privileged logs remain in verifier custody. Exposure includes the agent and any human/controller whose later experimental decisions can adapt to the result. No adaptive final diagnostics are permitted. A predeclared retry may reconcile the same operation or return its immutable cached receipt. Without safe reconciliation, final completion remains unknown; a new evaluation or newly informative disclosure is not an automatic retry.

Independent re-evaluation of an already frozen artifact may be permitted under the original claim for reproducibility auditing. It does not restore the dataset to unused, authorize development, or provide a new independent confirmation. Releasing the same cached receipt does not pretend another trial or grading execution occurred.

### C11. Diagnostics as observations

Diagnostics are experimental interventions with split/arm provenance and costs. Bounded development feedback should be able to distinguish first token divergence, wrong class, repeated tokens, premature EOS, length-cap termination, correct-prefix likelihood, likelihood along generated prefixes, and per-region update norms. Token-rank or activation detail requires explicit capability and information allowance. Results include the exact public case reference, candidate, execution settings, and truncation indicators.

The agent can use these observations to explain, for example, that `meow meow meow` has improved target-token probability but has not learned `meow meow potato`. The kernel does not diagnose a scientific cause automatically. Adding a previously forbidden diagnostic requires an authorized linked protocol change. Truncated logs cannot establish that an event never occurred.

### C12. Provenance, verification, and conclusions

A portable bundle MUST include the public declaration; target, basis, runtime, data and evaluator references; proposal and materialization records; operation/accounting journal; disclosure receipts; bound evaluation and selection records; freeze; separately bound final evidence; and preserved failures/unknowns. Private verifier material stays separately held. Exported agent reports omit credentials and unrestricted session transcripts. Referenced artifacts have relative locators or a declared retrieval mechanism, digests, versions, producers, and availability status. Temporary absolute paths alone are insufficient.

Verification exposes distinct results:

1. **Replay** reconstructs patches and decisions from recorded evidence without model calls. It proves consistency under a specified parser and arithmetic.
2. **Re-evaluation** executes frozen candidates and recomputes outputs and scores with the declared runtime. Exact equivalence or statistical/numerical tolerance must be defined before observing disagreement.
3. **Independent final grading** uses verifier-owned data and trusted scoring. It establishes only the tested behavior/population under that verifier's trust assumptions.
4. **Attestation**, if offered, authenticates a named signer's observations. A consistent unsigned history or provider-reported model/tool declaration is not an attestation.

Correctly bound fabricated scores can pass replay. Independent re-evaluation or trusted execution is needed to detect them. A client cannot infer score truth from a digest.

The final report separates protocol integrity, information-boundary assurance, target task observations, retention acceptability, agent/client usability, and scientific conclusion. Scientific conclusions are `supported`, `bounded_negative`, `inconclusive`, or `invalid_experiment`, each with claim scope and evidence references. Required unresolved thresholds yield inconclusive certification while preserving descriptive findings. A negative result needs its declared falsification rule or a verifiable finite-space certificate. Failure to discover an existing patch is not proof that no patch exists.

## Architecture alternatives and synthesis

| Architecture | What it owns | Main tradeoff | Disposition |
| --- | --- | --- | --- |
| Minimal SDK extension | Existing in-process candidate and result lifecycle, richer feedback and target adapter hooks | Small delivery, but external accounting/recovery and adversarial information boundaries remain caller responsibilities | Valid cooperative profile. Insufficient alone for the full request. |
| Local session kernel, separate runner and verifier | One experiment authority, durable operations, scoped observations, bounded target execution, independent final custody | More state machinery and real isolation configuration | Recommended contract and ownership boundary. |
| Hosted experiment service | Same authority with managed credentials, queues, quotas, and workers | Operations, tenant security, artifact custody, service availability, and cost | A possible deployment later, not a prerequisite or chosen release scope. |
| General agent orchestration framework | Flexible planning and tool scheduling | Does not itself supply numerical identities, experimental validity, or independent grading | An optional caller of the kernel, not the authority over scientific evidence. |

Two independent design candidates converged on the kernel boundary. The independent cross-judge preferred candidate B's compact ownership/lifecycle as the base. This synthesis adopts that base, adds candidate A's disclosure receipts, diagnostic contract, control-lineage distinctions, and portable evidence requirements, and resolves the judge's late-completion, final-exposure, hard-budget, and conclusion-axis questions above. The primary author initially favored A's operational detail, then accepted B's smaller core after reviewing that rationale. The prose and proposed interfaces were synthesized rather than pasted into parallel architectures.

The extra boundary earns its complexity by owning durable uncertainty and information release. It must not become a generic workflow engine. `Model the Domain` shaped the separate phase and operation records. `Boundary Discipline` places parsing and authorization at the kernel/runner/verifier edges. `Make Operations Idempotent` shaped result-before-selection commit and unknown remote completion. `Separate Before Serializing Shared State` keeps arm state and verifier data out of a shared mutable agent object. `Prove It Works` requires independent re-evaluation rather than score-hash claims.

## Proposed caller experience

Names below are illustrative design vocabulary. They are not a committed public API, wire version, or compatibility decision.

The normal loop is `observe -> submit`, followed by `freeze -> finalize -> export`. `submit` hides admission, reservation, materialization, evaluation, and one policy decision. It returns an operation receipt; callers need not hold framework objects or manually coordinate those stages. The same protocol can have a Python client and a constrained agent tool projection without duplicating policy.

```python
# Design sketch only. No implementation exists.
session = experiments.open(declaration_ref)
view = session.observe(arm="teacher")
details = session.inspect(ParameterSlice(view.checkpoint, allowed_region, page=0),
                          key="inspect-page-0")
operation = session.submit(proposal, key="trial-1-proposal-1")
receipt = session.status(operation.id)
# The agent repeats only while the declared budget and phase allow it.
# Pending or unknown operations require the receipt's permitted recovery action.
frozen = session.freeze(key="freeze-comparison")
if frozen.complete:
    final_operation = session.finalize(frozen.id, key="final-comparison")
bundle = session.export()
```

`finalize` authorizes a verifier-owned operation; it never gives the agent verifier files or an arbitrary grader command. `cancel(operation_id, key)` is available for interruption. Provisioning and credential authority belong to the operator, not arguments supplied by the agent. The pilot's teacher adapter can send an approved view to the confirmed model route with empty tool declarations and retain literal proposals without repair. It must not include ambient host context or unrelated history.

The transport broker accepts registered prompt/view recipes rather than arbitrary history or unapproved provider/tool configuration. An unavailable requested route is an explicit outcome, not permission to substitute another model or provider. Clients that bypass the broker cannot claim that its budget or information controls covered those requests.

| Proposed operation | Result and invariant |
| --- | --- |
| `open(declaration_ref)` | Admitted session or structured refusal; provisioning is operator-authorized |
| `observe(arm)` | `AgentView` with only caller-authorized state and legal next actions |
| `inspect(query, key)` | Bounded inspection receipt with provenance, charged probes/bytes, and truncation status |
| `submit(proposal, key)` | `OperationReceipt`; validation, execution, and selection have one durable owner |
| `status(operation_id)` | Existing receipt and its permitted recovery actions; no new evaluation |
| `cancel(operation_id, key)` | Cancellation-request receipt; never an invented termination confirmation |
| `freeze(key)` | Immutable `FrozenComparison`, complete or explicitly incomplete |
| `finalize(frozen_id, key)` | Bound verifier operation or phase/capability refusal |
| `export()` | Public `EvidenceBundle` plus access-scoped references; no grading side effect |

| Domain record | Required meaning |
| --- | --- |
| `DeclarationRef` | Immutable method, policies, roles, budgets, and execution-profile identity |
| `TargetCapability` | Supported inspection/mutation/numerical semantics, aliases, reset and reproducibility limits |
| `AgentView` | Authorized public state, remaining/unknown budget, metric definitions, phase, next legal actions, and disclosure receipt |
| `Proposal` | Bound base/observation/edit-space and typed non-executable edit payload |
| `OperationReceipt` | Stable operation ID, status, reservation/spend, committed result or failure, and retry/reconciliation permission |
| `Feedback` | Candidate and selection result, actual behavior, named surrogate, retention status, diagnostics, and evidence references |
| `FrozenComparison` | Fixed eligible candidates for all arms, unresolved obligations, exposure lineage, and completeness |
| `EvidenceBundle` | Portable public evidence plus separately referenced verifier evidence and explicit verification levels |

## Responsibility boundaries

| Owner | Authority | Explicit limit |
| --- | --- | --- |
| Kernel | Admission, scoped views, journal, operation IDs, reservations, phase transitions, pure selection, freeze | Does not know arbitrary target arithmetic or authenticate external scores by hashing |
| Target backend/runner | Capability validation, aliases, isolated materialization, inference, numerical receipts, reset and resource enforcement | Cannot expand edit/data permissions or select by undeclared metrics |
| Teacher transport | Approved prompt bytes, configured route, response events, observed model/tools and usage metadata | Cannot attest unobserved provider execution, hidden retries, or billing |
| Verifier | Private final data, scoring authority, bounded final release, re-evaluation, exposure record | Cannot return final feedback to a developing session |
| Thin client | Typed command encoding, bounded presentation, stable errors and available next actions | Contains no second policy engine or private evaluator state |
| Independent auditor | Replay, byte/metric checks, trust classification, claim/evidence reconciliation | Does not treat an agent's narrative or self-consistent bundle as proof of score truth |

Logical ownership is mandatory in the proposal; process/service packaging is an operator decision. In enforced mode those logical boundaries must correspond to tested access restrictions.

## Decisions left open

Supported target families and numerical formats; compatibility with current APIs and run formats; cooperative versus enforced deployment; concrete storage and transport; resource ceilings and cancellation authority; numerical/stochastic tolerances; acceptable behavioral and retention changes; statistical sample sizes and adoption thresholds; initial implementation scope; and release/publication policy remain open.

The implemented slice uses durable local SQLite ownership, a separate verifier registry, binary64 and pinned SmolLM2 targets, and enforced Docker execution. Existing public APIs remain available. Broader model support, resource and regression policy, full acceptance obligations, and release decisions remain open. The [native validation record](client-native-validation.json) records actual checks, failures, and the intervention needed to finish the fresh-context run.
