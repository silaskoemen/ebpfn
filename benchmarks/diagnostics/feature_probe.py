"""Frozen-feature probe (V2 P0 follow-up — see PLAN.md).

Localizes the P0 gap (a GP beats the base PFN everywhere) to REPRESENTATION vs HEAD.
Per dataset, on the same standardized query targets, compare:
  * PFN head RMSE  — what the trained bar head actually outputs;
  * ridge probe    — a fresh linear read-out of the PFN's own pre-head features
                     (`model.embed`, i.e. `out_ln(emb)` — the exact head input), 5-fold CV;
  * GP on raw X    — the P0 ceiling.

Reads:
  * probe << head  → HEAD / readout bottleneck: the features carry signal the head fails
    to output (undertrained / miscalibrated head). Cheap to fix. If probe ≈ GP the
    representation is as good as raw X.
  * probe ≈ head (both ≫ GP) → REPRESENTATION bottleneck: the features lack the signal a
    GP extracts from raw X → backbone / prior / training. The expensive path.

Run from project root:
    pixi run python -m benchmarks.diagnostics.feature_probe
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from benchmarks.baselines import gp_fit_predict, rmse
from benchmarks.diagnostics.reachability import CHAR_DIR, DEFAULT_BASE, iter_real_tasks
from ebpfn.pfn.data import collate_tasks
from ebpfn.pfn.train import load_checkpoint, select_device
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT_DIR = Path("benchmarks/results/diagnostics/feature_probe")


def ridge_probe(features: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """5-fold CV linear read-out of frozen features -> (rmse, r2), out-of-fold."""
    pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    oof = cross_val_predict(pipe, features, y, cv=KFold(5, shuffle=True, random_state=0))
    return float(np.sqrt(np.mean((oof - y) ** 2))), float(r2_score(y, oof))


@torch.no_grad()
def head_and_features(model, device, task) -> tuple[float, np.ndarray, np.ndarray]:
    batch = collate_tasks([task]).to(device)
    logits = model.predict_logits(batch.x, batch.y_train_std)[0]
    y = batch.y_test_std[0]
    head_rmse = float(torch.sqrt(torch.mean((model.distribution.mean(logits) - y) ** 2)))
    feats = model.embed(batch.x, batch.y_train_std)[0].float().cpu().numpy()  # (n_test, icl_dim) = head input
    return head_rmse, feats, y.cpu().numpy()


def run(checkpoint: Path, gp_max: int) -> pl.DataFrame:
    device = select_device("auto")
    model, ck = load_checkpoint(checkpoint, map_location=device)
    model.to(device).eval()
    print(f"base step={ck['step']} | device={device}\n")

    by_dataset: dict[str, list] = {}
    for name, _repeat, task in iter_real_tasks(CHAR_DIR):
        by_dataset.setdefault(name, []).append(task)

    rows: list[dict] = []
    for name, tasks in by_dataset.items():
        head, probe, probe_r2, gp = [], [], [], []
        for task in tasks:
            head_rmse, feats, y = head_and_features(model, device, task)
            p_rmse, p_r2 = ridge_probe(feats, y)
            x_tr = task.probe_fit.X.to_numpy().astype(np.float64)
            x_te = task.probe_score.X.to_numpy().astype(np.float64)
            y_fit = task.probe_fit.y.astype(np.float64)
            mu, sd = float(y_fit.mean()), max(float(y_fit.std()), 1e-6)
            gm, _ = gp_fit_predict(x_tr, (y_fit - mu) / sd, x_te, max_samples=gp_max)
            head.append(head_rmse)
            probe.append(p_rmse)
            probe_r2.append(p_r2)
            gp.append(rmse((task.probe_score.y.astype(np.float64) - mu) / sd, gm))
        row = {
            "dataset": name,
            "head_rmse": float(np.mean(head)),
            "probe_rmse": float(np.mean(probe)),
            "probe_r2": float(np.mean(probe_r2)),
            "gp_rmse": float(np.mean(gp)),
        }
        row["head_minus_probe"] = row["head_rmse"] - row["probe_rmse"]  # >0 => features beat the head
        rows.append(row)
        print(
            f"{name:<16} head={row['head_rmse']:.3f}  probe={row['probe_rmse']:.3f} (r2={row['probe_r2']:+.2f})  "
            f"gp={row['gp_rmse']:.3f}"
        )

    df = pl.DataFrame(rows).sort("head_minus_probe", descending=True)
    print(f"\n{'dataset':<16}{'head':>7}{'probe':>8}{'gp':>7}   verdict")
    n_head, n_repr = 0, 0
    for r in df.iter_rows(named=True):
        beats_head = r["head_rmse"] - r["probe_rmse"] > 0.05
        near_gp = r["probe_rmse"] - r["gp_rmse"] < 0.10
        if beats_head:
            verdict = "HEAD/readout (features carry signal head misses" + (", ≈GP)" if near_gp else ")")
            n_head += 1
        else:
            verdict = "REPRESENTATION (features lack the signal GP gets)"
            n_repr += 1
        print(f"{r['dataset']:<16}{r['head_rmse']:>7.3f}{r['probe_rmse']:>8.3f}{r['gp_rmse']:>7.3f}   {verdict}")
    print(f"\nsummary: HEAD-bottleneck {n_head}/{df.height} | REPRESENTATION-bottleneck {n_repr}/{df.height}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_DIR / "feature_probe.parquet")
    (OUT_DIR / "summary.json").write_text(
        json.dumps(
            {"base_step": int(ck["step"]), "n_head_bottleneck": n_head, "n_representation_bottleneck": n_repr}, indent=2
        )
    )
    print(f"\nartifacts -> {OUT_DIR}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--gp-max", type=int, default=3000)
    args = ap.parse_args()
    run(args.checkpoint, args.gp_max)


if __name__ == "__main__":
    main()
