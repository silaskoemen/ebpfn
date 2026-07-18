"""E1 reachability diagnostic (V2, see ``PLAN.md``).

Question: can the prior generate datasets that reach real datasets' points in the
frozen base PFN's embedding space? If a real point is off the prior's reachable
manifold, no amount of knob-tuning gets there -- that is a prior-*support* gap,
not a knob-position one.

Method (forward-pass only, no training):
  * frozen base -> ``model.embed(x, y_train)`` -> per-task ``z`` (mean over test rows);
  * reachability is PER-SHAPE -- a real dataset's feature count is fixed by the
    data, so the prior cloud is drawn at each real dataset's exact
    ``characterization_shape`` (the same exact-match convention as
    ``ebpfn/tune/evaluate.py``), never a single anchor shape;
  * compare each real dataset's ``z`` points (one per corpus repeat) to the prior
    ``z``-cloud with a whitened (diagonal-Mahalanobis) kNN ratio: ~1 => the real
    point sits within the cloud's natural spread; >>1 => off-manifold.

Run from project root:
    pixi run python -m benchmarks.diagnostics.reachability [--n-prior 200] [--checkpoint PATH]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from benchmarks.studies.characterization import make_task
from ebpfn.config import CharacterizationStudyConfig
from ebpfn.data import characterization_shape
from ebpfn.data.types import CharacterizationShape
from ebpfn.pfn.data import PriorTaskSource, collate_tasks
from ebpfn.pfn.train import load_checkpoint, select_device
from ebpfn.priors import hyperprior_from_dict, hyperprior_to_dict
from ebpfn.utils import RandomStreams

DEFAULT_BASE = Path(
    "benchmarks/results/offline_validation/learning_curve_pilot/checkpoints/"
    "28c6eac5e7702ca9/seed_0/1ab3167a4351f79906e3/checkpoint_step_00000950.pt"
)
CHAR_DIR = Path("benchmarks/results/characterization")
OUT_DIR = Path("benchmarks/results/diagnostics/reachability")


def pooled_embed(model, batch, device: torch.device) -> np.ndarray:
    """z per task = mean over test rows of the frozen embedding -> (B, D)."""
    z = model.embed(batch.x.to(device), batch.y_train_std.to(device))
    return z.mean(dim=1).float().cpu().numpy()


def iter_real_tasks(char_dir: Path):
    """Yield ``(dataset, repeat, TuningTask)`` for every audit-corpus repeat."""
    for manifest_path in sorted(char_dir.glob("*mode_audit*/task_manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        rows = manifest.get("tasks") or []
        if not rows:
            continue
        study = CharacterizationStudyConfig.model_validate_json((manifest_path.parent / "config.json").read_text())
        for row in rows:
            yield str(manifest["dataset"]), int(row["repeat"]), make_task(str(row["label"]), study, int(row["repeat"]))


def prior_cloud_z(model, device, eta, shape, n_tasks: int, batch_size: int, tag: str) -> np.ndarray:
    """Embed ``n_tasks`` prior draws at ``shape`` -> (n_tasks, D)."""
    source = PriorTaskSource(eta, RandomStreams(base_seed=0))
    out: list[np.ndarray] = []
    done = 0
    while done < n_tasks:
        take = min(batch_size, n_tasks - done)
        batch = source.tensor_batch(take, shape, "e1-reachability", tag, done)
        out.append(pooled_embed(model, batch, device))
        done += take
    return np.concatenate(out, axis=0)


def whitened_knn(query: np.ndarray, ref: np.ndarray, mean, std, k: int, drop_self: bool) -> np.ndarray:
    """Mean distance from each whitened query row to its k nearest whitened refs."""
    q = (query - mean) / std
    r = (ref - mean) / std
    dist = np.linalg.norm(q[:, None, :] - r[None, :, :], axis=2)
    dist.sort(axis=1)
    start = 1 if drop_self else 0  # drop the self-match (distance 0) for intra-cloud
    return dist[:, start : start + k].mean(axis=1)


def run(checkpoint: Path, n_prior: int, k: int, batch_size: int) -> pl.DataFrame:
    device = select_device("auto")
    model, ck = load_checkpoint(checkpoint, map_location=device)
    model.to(device).eval()  # load_checkpoint leaves params on cpu
    eta = hyperprior_from_dict(ck["source_eta"])
    corr = hyperprior_to_dict(eta).get("corr_strength_mean")
    print(f"base step={ck['step']} corr={corr} | device={device} | n_prior={n_prior}/dataset | k={k}\n")

    # group real tasks by dataset (all repeats)
    by_dataset: dict[str, list] = {}
    shapes: dict[str, CharacterizationShape] = {}
    for name, _repeat, task in iter_real_tasks(CHAR_DIR):
        by_dataset.setdefault(name, []).append(task)
        shapes[name] = characterization_shape(task)

    records: list[dict] = []
    emb_clouds: list[np.ndarray] = []
    emb_reals: list[np.ndarray] = []
    emb_names: list[str] = []
    emb_sizes: list[int] = []
    for name, tasks in by_dataset.items():
        shape = shapes[name]
        try:
            prior_z = prior_cloud_z(model, device, eta, shape, n_prior, batch_size, name)
        except Exception as error:
            print(f"{name:<16} PRIOR-SAMPLE-FAIL: {type(error).__name__}: {error}")
            continue
        real_z = np.stack([pooled_embed(model, collate_tasks([t]), device)[0] for t in tasks])
        emb_clouds.append(prior_z)
        emb_reals.append(real_z.mean(0))
        emb_names.append(name)
        emb_sizes.append(len(prior_z))
        mean, std = prior_z.mean(0), prior_z.std(0) + 1e-6
        intra = float(np.median(whitened_knn(prior_z, prior_z, mean, std, k, drop_self=True)))
        d_real = whitened_knn(real_z, prior_z, mean, std, k, drop_self=False)  # (n_repeats,)
        ratio = float(np.median(d_real)) / intra
        records.append(
            {
                "dataset": name,
                "p": shape.p_numeric,
                "n_train": shape.n_probe_fit,
                "n_test": shape.n_probe_score,
                "n_repeats": len(tasks),
                "intra_prior": intra,
                "d_real_to_prior": float(np.median(d_real)),
                "ratio": ratio,
            }
        )
        print(
            f"{name:<16} p={shape.p_numeric:<3} intra={intra:.2f} d(real->prior)={np.median(d_real):.2f} ratio={ratio:.2f}"
        )

    df = pl.DataFrame(records).sort("ratio")
    print(f"\n{'dataset':<16}{'p':>4}{'ratio':>8}   reachability (~1 inside, >3 far)")
    for r in df.iter_rows(named=True):
        verdict = "inside" if r["ratio"] <= 1.5 else ("far" if r["ratio"] > 3 else "edge")
        bar = "#" * min(36, int(r["ratio"] * 8))
        print(f"{r['dataset']:<16}{r['p']:>4}{r['ratio']:>8.2f}   {bar} {verdict}")
    ratios = df["ratio"].to_numpy()
    print(
        f"\nsummary: median ratio={np.median(ratios):.2f} | "
        f"inside(<=1.5)={int((ratios <= 1.5).sum())}/{len(ratios)} | far(>3)={int((ratios > 3).sum())}/{len(ratios)}"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_DIR / "reachability.parquet")
    np.savez(
        OUT_DIR / "embeddings.npz",
        prior_z=np.concatenate(emb_clouds),
        real_z=np.stack(emb_reals),
        names=np.array(emb_names),
        prior_sizes=np.array(emb_sizes),
    )
    (OUT_DIR / "summary.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "base_step": int(ck["step"]),
                "corr_strength_mean": corr,
                "n_prior": n_prior,
                "k": k,
                "median_ratio": float(np.median(ratios)),
                "n_inside": int((ratios <= 1.5).sum()),
                "n_far": int((ratios > 3).sum()),
                "n_datasets": len(ratios),
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"\nartifacts -> {OUT_DIR}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_BASE, help="frozen base PFN checkpoint")
    ap.add_argument("--n-prior", type=int, default=200, help="prior draws per real dataset")
    ap.add_argument("--k", type=int, default=5, help="neighbours for the kNN distance")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    run(args.checkpoint, args.n_prior, args.k, args.batch_size)


if __name__ == "__main__":
    main()
