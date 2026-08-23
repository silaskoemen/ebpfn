"""Reference row: the released TabICLv2 base against a same-data GP on the corpus.

Every later claim is a delta against this. The P2 null band asks how far the nine
targets drift when the base is adapted on the origin prior; P3's specialization prize
asks whether an eta aimed at a target beats that band. Both need one committed,
re-runnable measurement of where the *unadapted* base sits -- so this is a script and an
artifact rather than a number in a plan.

NLL and CRPS are the decision metrics; RMSE is reported to locate a target, never to
judge one. The GP is fit on the same context rows the model conditions on, so neither
side is flattered by seeing more data than the other.

Run from project root:
    pixi run python -m benchmarks.studies.base_eval [--gp-max 3000]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from benchmarks.base import released_base_checkpoint
from benchmarks.baselines import gaussian_coverage, gaussian_crps, gaussian_nll, gp_fit_predict, rmse
from benchmarks.corpus import CHAR_DIR, iter_real_tasks
from ebpfn.data import TuningTask
from ebpfn.pfn.data import collate_tasks
from ebpfn.pfn.metrics import compute_metrics
from ebpfn.pfn.train import load_checkpoint, select_device

OUT_DIR = Path("benchmarks/results/base_eval")
COVERAGE_LEVELS = (0.5, 0.8, 0.9, 0.95)


def _mean(frame: pl.DataFrame, column: str) -> float:
    """Column mean, refusing to summarize an empty corpus as a number."""
    value = frame[column].mean()
    if not isinstance(value, (int, float)):
        raise ValueError(f"{column} has no numeric mean ({value!r}); the corpus may be empty")
    return float(value)


@torch.no_grad()
def base_row(model, device: torch.device, task: TuningTask) -> dict[str, float]:
    """Metrics for the frozen base on one task, in standardized target space."""
    batch = collate_tasks([task]).to(device)
    logits = model.predict_logits(batch.x, batch.y_train_std)[0]
    metrics = compute_metrics(model.distribution, logits, batch.y_test_std[0], coverage_levels=COVERAGE_LEVELS)
    row = {"nll": metrics.nll, "crps": metrics.crps, "rmse": metrics.rmse}
    return row | {f"coverage_{level}": value for level, value in metrics.coverage.items()}


def gp_row(task: TuningTask, gp_max: int) -> dict[str, float]:
    """Same-data GP on the same standardization the model's collate applies."""
    y_fit = task.probe_fit.y.astype(np.float64)
    mu, sd = float(y_fit.mean()), max(float(y_fit.std()), 1e-6)
    x_tr = task.probe_fit.X.to_numpy().astype(np.float64)
    x_te = task.probe_score.X.to_numpy().astype(np.float64)
    y_te = (task.probe_score.y.astype(np.float64) - mu) / sd
    mean, std = gp_fit_predict(x_tr, (y_fit - mu) / sd, x_te, max_samples=gp_max)
    row = {
        "nll": gaussian_nll(y_te, mean, std),
        "crps": gaussian_crps(y_te, mean, std),
        "rmse": rmse(y_te, mean),
    }
    return row | {f"coverage_{level}": gaussian_coverage(y_te, mean, std, level) for level in COVERAGE_LEVELS}


def run(checkpoint: Path, gp_max: int) -> pl.DataFrame:
    device = select_device("auto")
    model, checkpoint_state = load_checkpoint(checkpoint, map_location=device)
    model.to(device).eval()
    print(f"base={checkpoint} step={checkpoint_state['step']} | device={device} | gp_max={gp_max}\n")

    rows: list[dict] = []
    for dataset, repeat, task in iter_real_tasks(CHAR_DIR):
        identity = {"dataset": dataset, "repeat": repeat, "n_query": task.probe_score.X.height}
        rows.append(identity | {"method": "base"} | base_row(model, device, task))
        rows.append(identity | {"method": "gp"} | gp_row(task, gp_max))
        print(f"  {dataset:<16} repeat={repeat}", end="\r")

    frame = pl.DataFrame(rows)
    metric_names = ["nll", "crps", "rmse", *(f"coverage_{level}" for level in COVERAGE_LEVELS)]
    per_dataset = (
        frame.group_by("dataset", "method")
        .agg(pl.col(name).mean() for name in metric_names)
        .pivot(on="method", index="dataset", values=metric_names)
        .with_columns(
            (pl.col("nll_base") - pl.col("nll_gp")).alias("nll_vs_gp"),
            (pl.col("crps_base") - pl.col("crps_gp")).alias("crps_vs_gp"),
        )
        .sort("nll_vs_gp", descending=True)
    )

    print(f"\n{'dataset':<16}{'nll':>8}{'crps':>8}{'rmse':>8}{'cov90':>7}  |{'gp nll':>8}{'crps':>8}  {'Δnll':>7}")
    for row in per_dataset.iter_rows(named=True):
        print(
            f"{row['dataset']:<16}{row['nll_base']:>8.3f}{row['crps_base']:>8.3f}{row['rmse_base']:>8.3f}"
            f"{row['coverage_0.9_base']:>7.3f}  |{row['nll_gp']:>8.3f}{row['crps_gp']:>8.3f}  {row['nll_vs_gp']:>+7.3f}"
        )

    beats_gp_nll = int((per_dataset["nll_vs_gp"] < 0).sum())
    beats_gp_crps = int((per_dataset["crps_vs_gp"] < 0).sum())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(OUT_DIR / "base_eval.parquet")
    per_dataset.write_parquet(OUT_DIR / "per_dataset.parquet")
    (OUT_DIR / "summary.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "base_step": int(checkpoint_state["step"]),
                "n_datasets": per_dataset.height,
                "n_tasks": frame.filter(pl.col("method") == "base").height,
                "gp_max_samples": gp_max,
                "coverage_levels": list(COVERAGE_LEVELS),
                "beats_gp_nll": beats_gp_nll,
                "beats_gp_crps": beats_gp_crps,
                "mean_nll_base": _mean(per_dataset, "nll_base"),
                "mean_crps_base": _mean(per_dataset, "crps_base"),
                "mean_coverage_90_base": _mean(per_dataset, "coverage_0.9_base"),
            },
            indent=2,
        )
    )
    print(f"\nbase beats GP on NLL {beats_gp_nll}/{per_dataset.height}, CRPS {beats_gp_crps}/{per_dataset.height}")
    print(f"artifacts -> {OUT_DIR}")
    return per_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None, help="defaults to the released base")
    parser.add_argument("--gp-max", type=int, default=3000, help="max GP train rows")
    args = parser.parse_args()
    run(args.checkpoint or released_base_checkpoint(), args.gp_max)


if __name__ == "__main__":
    main()
