# benchmarks/

**Scientific investigation** — diagnostics, comparisons, ablations, and studies
that back a characterization note or paper. Anything whose purpose is to *learn
something we might write up* belongs here, not in [`scripts/`](../scripts/README.md)
(which is for operating the pipeline).

Hydra entry points are allowed here (unlike reusable `ebpfn/` code): a study owns
both its logic and its CLI.

## Layout

| Path | Contents |
|---|---|
| `data/` | Dataset-acquisition adapters at the benchmark boundary (e.g. OpenML) |
| `studies/` | Paper-facing studies — logic + Hydra entry (e.g. `characterization.py` + `characterize.py`) |
| `diagnostics/` | Small quick checks, not paper-facing on their own |
| `results/` | Written artifacts (gitignored) |

Comparisons and ablations join `studies/`/`diagnostics/` as siblings when they arrive.

## Commands

| Command | Module |
|---|---|
| `pixi run characterize` | `benchmarks/studies/characterize.py` — Step 2 characterization study (`characterization_mode=audit` for the evidence gate) |
| `pixi run python -m benchmarks.diagnostics.headroom` | Reducible-vs-aleatoric headroom on the corpus: base PFN vs GP vs CatBoost on the same splits |
| `pixi run python -m benchmarks.diagnostics.target_adapter` | Direct-target-adapter upper bound: adapt the base on each target's own rows and compare to a same-data GP |
| `pixi run python -m benchmarks.diagnostics.feature_probe` | Frozen-feature probe: how much of the base's loss is readout rather than representation |
