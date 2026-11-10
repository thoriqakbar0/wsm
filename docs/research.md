# Research

WSM is inspired by [Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights](https://rikiyat.github.io/blog/weight-space-programming/), published by Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab at MIT CSAIL on November 2, 2026.

The authors explore whether an LLM can interpret, repair, and write neural network weights directly, without gradient descent. Their experiments span small game policies and language models. WSM supplies a Python workflow for exploring this idea through tensor inspection, explicit edits, evaluation, and checkpoint export.

## Tinker compatibility

WSM's experimental agent training works with the [Tinker SDK](https://thinkingmachines.ai/tinker/) and Tinker-compatible backends such as Modal Spindle. This reuses Tinker's training interface while WSM handles direct edits to the target model through Python.

One student training update has been verified through Spindle. Hosted Tinker support is implemented but has not yet been verified with a hosted training update. See [experimental agent trials](trials.md) for setup and evidence. Tinker is optional for the Python editing SDK.

## What WSM has established

WSM provides Python access to model weights and records evaluation comparisons. A scripted edit or lower public loss does not establish capability improvement, generalization, or independent edit discovery. Private grading requires a separate process with data unavailable to the editing program.
