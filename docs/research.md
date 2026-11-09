# Research scope

WSM independently implements a small weight-editing experiment inspired by [Weight-Space Programming and Gradient Optimization on Nonconvex Algorithmic Tasks](https://github.com/Fchaubard/WSP-for-turing-complete/blob/main/paper.pdf), a research preprint by Francois Chaubard with an AI research collaborator.

## Source evidence

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

WSM does not yet reproduce the paper's 87 settings, optimizer comparisons, reported rankings, archived language-model observations, architecture-selection trials, recurrent networks, or resource-accounting results. The example establishes neither optimizer superiority nor general extrapolation ability.

The interpreter uses binary64 scalar arithmetic. Agreement on synthetic fixtures does not establish agreement with every NumPy, BLAS, accelerator, or fused-arithmetic implementation.

## Held-out evaluation

The teacher receives only training and development examples and their feedback. Final grading takes a frozen artifact and evaluator-owned test data. A local library cannot prevent an unrestricted host process from reading other host files. Harbor must isolate the agent environment from the final verifier and transfer only declared output artifacts.
