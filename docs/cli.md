# CLI commands

Create the parent directory with `mkdir -p runs`, then run `wsm execute edit.py --model /path/to/snapshot --cases examples/coding/cases.json --output runs/edit --device cpu`. The file must define `edit(model)`. The output directory must not exist. A successful run saves `checkpoint/` and `run.json`. Failed or interrupted execution returns a nonzero exit status. Inspect the receipt and checkpoint before retrying.

Run `wsm` or `wsm tui` in a terminal and press **p** to review and execute a Python file. Use `wsm --help` or `wsm --version` in scripts.

Legacy experiment commands and workspace options have been removed. See [the coding guide](coding.md).
