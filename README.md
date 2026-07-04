# ebpfn

Empirical-Bayes PFN experiments for tabular regression.

The repository tests whether dataset-level learning geometry can be diagnosed
from real tabular tasks and used as a coverage surrogate for tuning PFN-style
priors. The current codebase contains:

- Gate 0 toy prior and distance/MMD diagnostics.
- Gate 1 PFN/prior pairing and calibration checks.
- Gate 2 conditional-structure descriptors, descriptor-cloud coverage, and
  across-prior fixed-effects ablations.

## Development

Install the Pixi environment:

```bash
pixi install
```

Run tests:

```bash
pixi run test
```

Run the Gate-2 quick wiring check:

```bash
pixi run gate2-quick
```

Run the full Gate-2 experiment:

```bash
pixi run gate2 --name gate2_full --steps 4000
```

The OpenML-backed corpus loader needs network access the first time datasets are
cached under `data/raw/openml`.
