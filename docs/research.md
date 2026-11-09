# Research scope

WSM's target workflow follows [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/), by Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab. An editing AI reasons about a target network and changes its weights directly. The research workflow excludes gradient descent and optimizer-driven training, and declares the diagnostic and inference access available to the editing AI.

The [Hugging Face demonstration](https://huggingface.co/spaces/ML-Intern-lab/weight-space-programming-demo) compares an existing base checkpoint with a reconstructed edited checkpoint. Its [application source](https://huggingface.co/spaces/ML-Intern-lab/weight-space-programming-demo/blob/main/app.py) loads both checkpoints for generation. Running that comparison does not demonstrate an agent discovering an edit.

The original WSM addition example instead drew on [Weight-Space Programming and Gradient Optimization on Nonconvex Algorithmic Tasks](https://github.com/Fchaubard/WSP-for-turing-complete/blob/main/paper.pdf), a separate research preprint by Francois Chaubard with an AI research collaborator. Its provenance remains below. That example does not reproduce the newly designated source's language-model experiments.

## Primary experiment artifacts

The authors' [Experiment 8 record](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md) describes the permitted tools, exact evaluation protocol, and saved edits. The [p50c edit manifest](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/edit/manifest.json) and [reconstruction script](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/code/apply_edit.py) provide a stronger reproduction starting point than the demo alone. Their base revision is `2fc06364715b967f1860aea9cf38778875588b17` of `Qwen/Qwen3.5-0.8B`. A published-edit replay must check the supplied tensor hashes and preserve its rounding protocol.

WSM's current pinned SmolLM profile and scripted final-norm edit are engineering checks, not that experiment. Experiment 8 prohibited final-norm and embedding edits and allowed a persistent Python weight-analysis kernel with separate batched forward tools. WSM currently exposes bounded scalar replacements and short greedy generation. Its invocation budget, model, editable tensors, dtype, and evaluation protocol differ. Do not compare its proof scores or call counts with the paper's benchmark results.

## Historical addition-example evidence

The initial implementation was grounded against research repository commit `917a213a770abad0cbde31ceceea1d9d9c51eaf2`. The inspected artifacts include:

- `paper.pdf` and `REPRODUCIBILITY.md` describe the experimental scope and numerical limits.
- `focus/arithmetic/fixtures.py` defines a two-input dense ReLU network with a scalar output and row-major parameter arrays.
- `focus/arithmetic/score.py` validates complete finite coefficient arrays and evaluates archived proposals.
- `focus/iterative_eval/check_selection.py` independently checks strict lexicographic selection by development loss, then training loss.

The upstream repository does not declare a project-wide reuse license. WSM uses fresh code and synthetic examples. It does not redistribute upstream source, archived model replies, or saved coefficient arrays.

## What the first example establishes

The first example checks the mechanics of direct numerical weight edits against a fixed network. An external teacher submits edits. The runtime evaluates a separate candidate and selects by development loss, using training loss to break ties. A separate grader scores the final checkpoint against an independent addition reference.

This is a synthetic reproduction of an experimental procedure and a network family. It is not a replication of a paper result. The example uses its own data, initial weights, and scripted proposals. A scripted teacher is an offline test fixture, not a new language-model observation.

## What remains untested

WSM does not establish replication of the linked Sparks experiments or their reported scores. Reproducing a published patch verifies checkpoint and evaluation mechanics. A fresh AI must independently choose an edit before a run can provide evidence of edit discovery. Keep these claims separate in run reports.

The research also reports a language-model case with no established improvement. Removing a repetition failure does not establish acquisition of a new reasoning capability. Model family, editing-AI provider, compute budget, and acceptable retention regressions remain operator choices. [Tinker compatibility](tinker.md) is a separate backend capability question.

WSM does not yet reproduce the paper's 87 settings, optimizer comparisons, reported rankings, archived language-model observations, architecture-selection trials, recurrent networks, or resource-accounting results. The example establishes neither optimizer superiority nor general extrapolation ability.

The interpreter uses binary64 scalar arithmetic. Agreement on synthetic fixtures does not establish agreement with every NumPy, BLAS, accelerator, or fused-arithmetic implementation.

## Held-out evaluation

The teacher receives only training and development examples and their feedback. Final grading takes a frozen artifact and evaluator-owned test data. A local library cannot prevent an unrestricted host process from reading other host files. Harbor must isolate the agent environment from the final verifier and transfer only declared output artifacts.
