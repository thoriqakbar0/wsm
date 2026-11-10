# Architecture

`WeightModel` owns offline model loading, live PyTorch tensors, public evaluation, and verified checkpoint saving. `execute_program` executes a trusted Python file defining `edit(model)` and records baseline and candidate evaluations in `run.json`.

`wsm.editing` owns models and their publication lifecycle. `wsm.program` owns program execution and receipts. The CLI and terminal call the same execution function. `_publication` provides atomic exclusive rename without loading any experiment machinery.

Python execution has the invoking process's privileges. Public evaluation does not provide private grading or automatic selection. See [the coding guide](coding.md).
