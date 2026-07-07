# ebpfn

Empirical-Bayes PFN experiments for tabular regression.

The repository implements a staged system for adapting a hierarchical synthetic
regression-task prior to one supplied tabular dataset. The current implementation
provides strict configuration, deterministic random streams, leakage-safe source
splits, Polars task contracts, fixed-map regression characterization, and OpenML
ingestion at the benchmark boundary.

## Development

Install the Pixi environment:

```bash
pixi install
```

Run the acceptance checks:

```bash
pixi run lint
pixi run test
```

Prepare the configured OpenML source through the Hydra application boundary:

```bash
pixi run prepare-data
```

Run the deterministic Step 2 characterization smoke study:

```bash
pixi run characterize
```

Use `characterization_mode=audit` for the five-repeat evidence gate. An audit
remains `incomplete` until representative real-task repeats are supplied; its
validated row/feature applicability bounds are recorded explicitly. Study tables,
schemas, configuration, provenance, and decisions are written below
`benchmarks/results/characterization`.

Run the Step 3 hierarchical-prior p-complexity audit:

```bash
pixi run prior-audit
```

Use `prior_mode=audit` for the denser feature grid. The generator samples varied
task mechanisms over four routes (SCM, BNN, tree, compositional) at a requested
shape; the audit checks route-frequency convergence, reproducibility, and that
per-route complexity and realized SNR stay controlled across the feature grid.
The joint-Sobol identifiability study is deferred to Step 4. Artifacts are written
below `benchmarks/results/prior`.

OpenML acquisition needs network access the first time a task is cached under
`data/raw/openml`. Reusable package code does not import OpenML or Hydra.
