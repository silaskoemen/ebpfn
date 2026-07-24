"""Direct-target-adapter upper bound (V2 P0 diagnostic (b) — see PLAN.md).

The complement to ``feature_probe.py``. That probe froze the base and read its
features; this one lets *gradients flow* and adapts the base **on each target's own
rows**, then asks the localizing question:

  * adapted ≈ GP  → the PFN/arch is CAPABLE of the target; the base only lacked the
    right *prior source*. The empirical-Bayes / prior-family line is alive → fix the
    prior (P3), because adapting on the ideal source (the target itself) closes to GP.
  * adapted ≫ GP  → even the loosest possible adaptation (a full fine-tune on the
    target itself, the strongest conceivable "source") cannot reach GP. The ceiling is
    the ADAPTATION MECHANISM / PFN capacity, not the prior source → prior tuning is moot.

This is an *upper bound*: full-model fine-tune (not a restricted adapter) on the
target's own data is the most generous test of capacity. If it fails, no smaller
adapter or better prior can rescue that target within this arch.

This is the DATASET-ONLY-TUNING oracle — the baseline prior-tuning must beat. Adapting on
the target's own training rows is the *intent*, not leakage; the held-out TEST split only
buys honest scoring (never grade on rows you trained on). Each dataset's finite rows split
into a disjoint TEST and an ADAPT pool; the transform is fit on ADAPT only; adaptation
tasks resample (support, query) splits *within* ADAPT; base/adapted/GP are scored on the
same TEST. The GP ceiling is fit on the SAME ADAPT pool as the adaptation (see gp_on_task)
so the comparison is same-data. RMSE localizes; NLL flags the E3 calibration failure mode
(RMSE down, NLL up = overfit specialist).

Run from project root (long; wrap in caffeinate):
    caffeinate -i pixi run python -m benchmarks.diagnostics.target_adapter [--datasets naval] [--steps 200]
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from benchmarks.baselines import gaussian_nll, gp_fit_predict, rmse
from benchmarks.diagnostics.reachability import DEFAULT_BASE
from benchmarks.studies.characterization import _load_openml_regression_dataset
from ebpfn.config import PfnArchConfig, PfnTrainConfig, PreprocessingConfig
from ebpfn.data import FeatureSchema, TaskPartition, TuningTask
from ebpfn.data.preprocessing import fit_feature_transform
from ebpfn.pfn.data import TaskBatch, collate_tasks
from ebpfn.pfn.train import load_checkpoint, select_device, train_pfn
from ebpfn.priors import hyperprior_from_dict
from ebpfn.utils import RandomStreams

OUT_DIR = Path("benchmarks/results/diagnostics/target_adapter")
DATASETS = [
    "airfoil",
    "concrete",
    "energy_cooling",
    "energy_heating",
    "kin8nm",
    "naval",
    "protein",
    "superconduct",
    "yacht",
]
MAX_FEATURES = 100
ADAPT_CAP = 4000  # bound the ADAPT pool (GP + memory); big datasets are subsampled
TEST_FRACTION = 5  # held-out TEST = min(200, n_finite // TEST_FRACTION)


@dataclass
class TargetData:
    """Leakage-clean transformed split of one real dataset."""

    name: str
    adapt_x: np.ndarray  # (n_adapt, p) transformed float32
    adapt_y: np.ndarray  # (n_adapt,) raw target
    adapt_ids: np.ndarray  # original row indices (globally unique)
    test_x: np.ndarray  # (n_test, p) transformed float32
    test_y: np.ndarray  # (n_test,) raw target
    test_ids: np.ndarray  # original row indices (globally unique)
    schema: FeatureSchema
    transform_id: str
    missing: tuple[float, ...]
    n_support: int  # eval + train support size


def prepare(name: str, seed: int) -> TargetData:
    dataset = _load_openml_regression_dataset(name)
    finite = np.flatnonzero(np.isfinite(dataset.y.astype(float, copy=False)))
    perm = np.random.default_rng(seed).permutation(finite)
    n_test = min(200, len(finite) // TEST_FRACTION)
    test_idx, adapt_idx = perm[:n_test], perm[n_test : n_test + ADAPT_CAP]

    schema = dataset.schema
    numeric = tuple(n for n, k in zip(schema.names, schema.kinds, strict=True) if k != "categorical")[:MAX_FEATURES]
    raw_schema = schema.select(numeric)
    adapt_frame, test_frame = dataset.X[adapt_idx].select(numeric), dataset.X[test_idx].select(numeric)
    preprocessing = PreprocessingConfig(
        max_features=MAX_FEATURES,
        clip=4.0,
        constant_atol=1e-12,
        constant_rtol=1e-12,
        scale_epsilon=1e-12,
        version="target-adapter-preprocess-1",
    )
    transform = fit_feature_transform(adapt_frame, raw_schema, preprocessing)  # fit on ADAPT only
    adapt_x = transform.apply(adapt_frame).to_numpy().astype(np.float32)
    test_x = transform.apply(test_frame).to_numpy().astype(np.float32)
    # eval/train support: the dataset's characterization scale (480), capped to leave a query slice
    n_support = min(480, len(adapt_idx) - min(160, len(adapt_idx) // 3))
    return TargetData(
        name,
        adapt_x,
        dataset.y[adapt_idx].astype(float),
        adapt_idx,
        test_x,
        dataset.y[test_idx].astype(float),
        test_idx,
        transform.output_schema,
        transform.transform_id,
        transform.probe_fit_missing_rates,
        n_support,
    )


class RealTaskSource:
    """Train source that resamples (support, query) in-context tasks from ONE target's
    ADAPT rows — the seam train_pfn drives to adapt the base on the target itself."""

    def __init__(self, data: TargetData, n_query: int, seed: int, eta) -> None:
        self.data = data
        self.n_query = n_query
        self.names = list(data.schema.names)
        self.streams = RandomStreams(seed)
        self.eta = eta  # base eta, provenance only — tasks come from the target, not this eta

    @property
    def stream_provenance(self) -> dict[str, str | int | bool | None]:
        return {
            "version": "real-adapt-task-source-1",
            "base_seed": self.streams.base_seed,
            "common_random_numbers": False,
            "pairing_id": None,
        }

    def _task(self, rng: np.random.Generator) -> TuningTask:
        d = self.data
        take = min(d.n_support + self.n_query, len(d.adapt_y))
        idx = rng.choice(len(d.adapt_y), size=take, replace=False)
        s, q = idx[: d.n_support], idx[d.n_support :]
        fit = TaskPartition(pl.DataFrame(d.adapt_x[s], schema=self.names), d.adapt_y[s], d.adapt_ids[s])
        score = TaskPartition(pl.DataFrame(d.adapt_x[q], schema=self.names), d.adapt_y[q], d.adapt_ids[q])
        return TuningTask(
            "adapt",
            f"adapt-{d.name}",
            "regression",
            "adapt-split",
            "adapt-split",
            fit,
            score,
            d.schema,
            d.transform_id,
            d.missing,
        )

    def sample_batch(self, batch_size: int, shape, *identity) -> list[TuningTask]:
        rng = np.random.default_rng(abs(hash(("real-adapt", self.data.name, *identity))) % (2**32))
        return [self._task(rng) for _ in range(batch_size)]

    def tensor_batch(self, batch_size: int, shape, *identity) -> TaskBatch:
        return collate_tasks(self.sample_batch(batch_size, shape, *identity))


def eval_task(data: TargetData) -> TuningTask:
    """Fixed eval task: support = first n_support ADAPT rows, query = all held-out TEST."""
    names = list(data.schema.names)
    s = slice(0, data.n_support)
    fit = TaskPartition(pl.DataFrame(data.adapt_x[s], schema=names), data.adapt_y[s], data.adapt_ids[s])
    score = TaskPartition(pl.DataFrame(data.test_x, schema=names), data.test_y, data.test_ids)
    return TuningTask(
        "eval",
        f"eval-{data.name}",
        "regression",
        "eval-split",
        "eval-split",
        fit,
        score,
        data.schema,
        data.transform_id,
        data.missing,
    )


@torch.no_grad()
def score_model(model, device, batch: TaskBatch) -> tuple[float, float]:
    logits = model.predict_logits(batch.x, batch.y_train_std)[0]
    y = batch.y_test_std[0]
    r = float(torch.sqrt(torch.mean((model.distribution.mean(logits) - y) ** 2)))
    nll = float(model.distribution.nll(logits, y).mean())
    return r, nll


def gp_on_task(data: TargetData, task: TuningTask, gp_max: int) -> tuple[float, float]:
    """Fair GP ceiling: fit on the SAME ADAPT pool the PFN adapts on (subsampled to
    ``gp_max``), not just the eval support — otherwise the PFN (weights trained on the
    whole pool) is compared to a GP that saw ~8x fewer rows on the big datasets, which
    spuriously flatters the PFN. Standardized by the eval support's mean/std to match the
    model's collate standardization; scored on the same held-out TEST query."""
    y_fit = task.probe_fit.y.astype(np.float64)
    mu, sd = float(y_fit.mean()), max(float(y_fit.std()), 1e-6)
    x_tr, y_tr = data.adapt_x.astype(np.float64), (data.adapt_y.astype(np.float64) - mu) / sd
    x_te = task.probe_score.X.to_numpy().astype(np.float64)
    mean, std = gp_fit_predict(x_tr, y_tr, x_te, max_samples=gp_max)
    y_te = (task.probe_score.y.astype(np.float64) - mu) / sd
    return rmse(y_te, mean), gaussian_nll(y_te, mean, std)


def adapt_one(name: str, checkpoint: Path, steps: int, gp_max: int, seed: int) -> dict:
    device = select_device("auto")
    base_model, ck = load_checkpoint(checkpoint, map_location=device)
    base_model.to(device).eval()
    arch = PfnArchConfig.model_validate(ck["arch"])
    train = PfnTrainConfig.model_validate(ck["train"]).model_copy(
        update={
            "steps": steps,
            "seed": seed,
            "warmup_steps": min(20, steps // 4),
            "checkpoint_interval": max(1, steps // 6),
        }
    )

    data = prepare(name, seed)
    n_query = min(160, len(data.adapt_y) - data.n_support)
    print(
        f"[{name}] adapt_pool={len(data.adapt_y)} test={len(data.test_y)} p={data.adapt_x.shape[1]} "
        f"support={data.n_support} query={n_query} | fine-tune {steps} steps"
    )

    eval_batch = collate_tasks([eval_task(data)]).to(device)
    base_rmse, base_nll = score_model(base_model, device, eval_batch)
    gp_rmse, gp_nll = gp_on_task(data, eval_task(data), gp_max)
    del base_model
    if device.type == "mps":
        torch.mps.empty_cache()

    ckpt_dir = OUT_DIR / "checkpoints" / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for stale in ckpt_dir.glob("checkpoint_step_*.pt"):  # fresh run per invocation (seed/steps may change)
        stale.unlink()
    source = RealTaskSource(data, n_query, seed, hyperprior_from_dict(ck["source_eta"]))
    train_pfn(
        arch,
        train,
        source=source,  # ty:ignore[invalid-argument-type]
        checkpoint_dir=ckpt_dir,
        init_weights_from=checkpoint,
        log_every=max(1, steps // 5),
    )

    # eval every saved checkpoint -> pick the best-NLL and best-RMSE adaptation point
    curve = []
    for path in sorted(ckpt_dir.glob("checkpoint_step_*.pt"), key=lambda p: int(p.stem.rsplit("_", 1)[-1])):
        m, mck = load_checkpoint(path, map_location=device)
        m.to(device).eval()
        r, nll = score_model(m, device, eval_batch)
        curve.append({"step": int(mck["step"]), "rmse": r, "nll": nll})
        del m
        if device.type == "mps":
            torch.mps.empty_cache()

    for path in ckpt_dir.glob("checkpoint_step_*.pt"):  # bound disk: ~110MB/checkpoint
        path.unlink()
    best_rmse = min(curve, key=lambda c: c["rmse"])
    best_nll = min(curve, key=lambda c: c["nll"])
    row = {
        "dataset": name,
        "base_rmse": base_rmse,
        "adapt_rmse": best_rmse["rmse"],
        "adapt_rmse_step": best_rmse["step"],
        "gp_rmse": gp_rmse,
        "base_nll": base_nll,
        "adapt_nll": best_nll["nll"],
        "adapt_nll_step": best_nll["step"],
        "gp_nll": gp_nll,
        "adapt_gp_rmse_gap": best_rmse["rmse"] - gp_rmse,
        "curve": curve,
    }
    print(
        f"[{name}] base rmse={base_rmse:.3f} nll={base_nll:+.3f} | adapt rmse={best_rmse['rmse']:.3f}@{best_rmse['step']} "
        f"nll={best_nll['nll']:+.3f}@{best_nll['step']} | gp rmse={gp_rmse:.3f} nll={gp_nll:+.3f}"
    )
    return row


def verdict(row: dict) -> str:
    reaches_gp = row["adapt_gp_rmse_gap"] < 0.10
    adapted = row["base_rmse"] - row["adapt_rmse"] > 0.03
    if reaches_gp:
        return "PRIOR-SOURCE (adapt reaches GP -> PFN capable, base lacked the source)"
    if adapted:
        return "PARTIAL (adapt helps but stalls above GP -> capacity + source)"
    return "PFN-CAPACITY (adapt can't move off base / stays >> GP -> arch/head ceiling)"


def run(checkpoint: Path, datasets: list[str], steps: int, gp_max: int, seed: int) -> pl.DataFrame:
    rows = [adapt_one(name, checkpoint, steps, gp_max, seed) for name in datasets]
    df = pl.DataFrame([{k: v for k, v in r.items() if k != "curve"} for r in rows]).sort(
        "adapt_gp_rmse_gap", descending=True
    )

    print(f"\n{'dataset':<16}{'base':>7}{'adapt':>7}{'gp':>7}{'Δnll':>8}   verdict")
    tally: dict[str, int] = {}
    for r in rows:
        v = verdict(r)
        tally[v.split()[0]] = tally.get(v.split()[0], 0) + 1
    for r in sorted(rows, key=lambda x: x["adapt_gp_rmse_gap"], reverse=True):
        dnll = r["adapt_nll"] - r["base_nll"]  # >0 = adaptation hurt calibration (E3 mode)
        print(
            f"{r['dataset']:<16}{r['base_rmse']:>7.3f}{r['adapt_rmse']:>7.3f}{r['gp_rmse']:>7.3f}{dnll:>+8.3f}   {verdict(r)}"
        )
    print("\nsummary: " + " | ".join(f"{k} {v}/{len(rows)}" for k, v in sorted(tally.items())))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_DIR / "target_adapter.parquet")
    (OUT_DIR / "summary.json").write_text(json.dumps({"steps": steps, "seed": seed, "rows": rows}, indent=2))
    print(f"\nartifacts -> {OUT_DIR}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--gp-max", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.checkpoint, args.datasets, args.steps, args.gp_max, args.seed)


if __name__ == "__main__":
    main()
