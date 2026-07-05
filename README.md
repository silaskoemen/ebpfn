# ebpfn

Empirical-Bayes PFN experiments for tabular regression.

The repository implements a staged system for adapting a hierarchical synthetic
regression-task prior to one supplied tabular dataset. The current foundation
provides strict configuration, deterministic random streams,
leakage-safe source splits, Polars task contracts, explicit rotations, and
OpenML ingestion at the benchmark boundary.

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

OpenML acquisition needs network access the first time a task is cached under
`data/raw/openml`. Reusable package code does not import OpenML or Hydra.
