"""P0 corpus-headroom check (V2, see ``PLAN.md``).

Question: do our real datasets contain *reducible* (prior-addressable) headroom, or is
their loss mostly aleatoric? If a strong probabilistic baseline can't beat the base PFN
(and both sit near the marginal, RMSE~1), that target's loss is irreducible and it
cannot answer the prior-tuning question — the corpus, not the method, is the problem.

Compares base PFN vs a GP (sklearn, subsampled) vs CatBoost (RMSEWithUncertainty) on
the *same* task splits, in standardized-target space (NLL comparable to the bar head).

Run from project root:
    pixi run python -m benchmarks.diagnostics.headroom
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from benchmarks.baselines import catboost_fit_predict, coverage90, gaussian_nll, gp_fit_predict, rmse
from benchmarks.diagnostics.reachability import CHAR_DIR, DEFAULT_BASE, iter_real_tasks
from ebpfn.pfn.data import collate_tasks
from ebpfn.pfn.train import load_checkpoint, select_device

OUT_DIR = Path("benchmarks/results/diagnostics/headroom")

try:
    import catboost  # noqa: F401

    HAVE_CATBOOST = True
except Exception:
    HAVE_CATBOOST = False

METHODS = ("pfn", "gp", "catboost") if HAVE_CATBOOST else ("pfn", "gp")


def _features(partition) -> np.ndarray:
    return partition.X.to_numpy().astype(np.float64)


@torch.no_grad()
def pfn_metrics(model, device, task) -> tuple[float, float, float]:
    batch = collate_tasks([task]).to(device)
    logits = model.predict_logits(batch.x, batch.y_train_std)[0]
    y = batch.y_test_std[0]
    nll = float(model.distribution.nll(logits, y).mean())
    point_rmse = float(torch.sqrt(torch.mean((model.distribution.mean(logits) - y) ** 2)))
    lo = model.distribution.icdf(logits, 0.05)
    hi = model.distribution.icdf(logits, 0.95)
    cov = float(((y >= lo) & (y <= hi)).float().mean())
    return nll, point_rmse, cov


def run(checkpoint: Path, iterations: int, gp_max: int) -> pl.DataFrame:
    device = select_device("auto")
    model, ck = load_checkpoint(checkpoint, map_location=device)
    model.to(device).eval()
    print(f"base step={ck['step']} | device={device} | gp_max={gp_max} | catboost_iters={iterations}\n")

    by_dataset: dict[str, list] = {}
    for name, _repeat, task in iter_real_tasks(CHAR_DIR):
        by_dataset.setdefault(name, []).append(task)

    rows: list[dict] = []
    for name, tasks in by_dataset.items():
        acc = {m: {"nll": [], "rmse": [], "cov": []} for m in METHODS}
        for task in tasks:
            x_tr, x_te = _features(task.probe_fit), _features(task.probe_score)
            y_fit = task.probe_fit.y.astype(np.float64)
            y_score = task.probe_score.y.astype(np.float64)
            mu, sd = float(y_fit.mean()), max(float(y_fit.std()), 1e-6)
            y_tr, y_te = (y_fit - mu) / sd, (y_score - mu) / sd

            n, r, c = pfn_metrics(model, device, task)
            acc["pfn"]["nll"].append(n)
            acc["pfn"]["rmse"].append(r)
            acc["pfn"]["cov"].append(c)
            gm, gs = gp_fit_predict(x_tr, y_tr, x_te, max_samples=gp_max)
            acc["gp"]["nll"].append(gaussian_nll(y_te, gm, gs))
            acc["gp"]["rmse"].append(rmse(y_te, gm))
            acc["gp"]["cov"].append(coverage90(y_te, gm, gs))
            if HAVE_CATBOOST:
                cm, cs = catboost_fit_predict(x_tr, y_tr, x_te, iterations=iterations)
                acc["catboost"]["nll"].append(gaussian_nll(y_te, cm, cs))
                acc["catboost"]["rmse"].append(rmse(y_te, cm))
                acc["catboost"]["cov"].append(coverage90(y_te, cm, cs))

        row = {"dataset": name, "n_repeats": len(tasks)}
        for m in METHODS:
            for k in ("nll", "rmse", "cov"):
                row[f"{m}_{k}"] = float(np.mean(acc[m][k]))
        # reducible headroom: does the best baseline beat the PFN?
        base_methods = [m for m in METHODS if m != "pfn"]
        best_base_nll = min(row[f"{m}_nll"] for m in base_methods)
        best_base_rmse = min(row[f"{m}_rmse"] for m in base_methods)
        row["nll_headroom"] = row["pfn_nll"] - best_base_nll  # >0 => baseline beats PFN
        row["rmse_headroom"] = row["pfn_rmse"] - best_base_rmse
        rows.append(row)
        base_str = "  ".join(f"{m}(nll={row[f'{m}_nll']:+.3f} rmse={row[f'{m}_rmse']:.3f})" for m in base_methods)
        print(f"{name:<16} pfn(nll={row['pfn_nll']:+.3f} rmse={row['pfn_rmse']:.3f})  {base_str}")

    df = pl.DataFrame(rows).sort("rmse_headroom", descending=True)
    print(f"\n{'dataset':<16}{'pfn_rmse':>9}{'best_base':>11}{'Δrmse':>8}{'Δnll':>8}   read")
    for r in df.iter_rows(named=True):
        best_base = min(r[f"{m}_rmse"] for m in METHODS if m != "pfn")
        if r["rmse_headroom"] > 0.05:
            read = "REDUCIBLE (baseline beats PFN)"
        elif best_base > 0.9:
            read = "aleatoric floor (all near marginal)"
        else:
            read = "PFN ~ floor"
        print(
            f"{r['dataset']:<16}{r['pfn_rmse']:>9.3f}{best_base:>11.3f}{r['rmse_headroom']:>+8.3f}{r['nll_headroom']:>+8.3f}   {read}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_DIR / "headroom.parquet")
    (OUT_DIR / "summary.json").write_text(
        json.dumps(
            {
                "base_step": int(ck["step"]),
                "n_reducible": int((df["rmse_headroom"] > 0.05).sum()),
                "n_datasets": df.height,
            },
            indent=2,
        )
    )
    print(f"\nartifacts -> {OUT_DIR}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--iterations", type=int, default=500, help="catboost iterations")
    ap.add_argument("--gp-max", type=int, default=3000, help="max GP train samples")
    args = ap.parse_args()
    run(args.checkpoint, args.iterations, args.gp_max)


if __name__ == "__main__":
    main()
