# scripts/

**Operational entry points** — commands you run to *operate the pipeline* on a
supplied dataset: preparing data, tuning a prior for a task, training a model.

These are thin Hydra CLIs launched from the project root; the reusable logic
lives in `ebpfn/`, and data-acquisition adapters live in `benchmarks/data/`.

Scientific work — diagnostics, comparisons, ablations, studies that back a
note or paper — does **not** belong here; it lives in [`benchmarks/`](../benchmarks/README.md).

## Commands

| Command | Module |
|---|---|
| `pixi run prepare-data` | `scripts/prepare_data.py` — build a leakage-safe source manifest from OpenML |

Future operational entries (`tune.py`, `train.py`) land flat alongside these.
