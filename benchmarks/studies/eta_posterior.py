"""Clean marginal likelihood: does the real corpus carry information about eta at all?

Empirical Bayes over the generator needs p(D | eta). We get it without retraining the
PFN per eta from an amortized posterior q(eta | D) trained on simulations with
eta ~ U(cube): under that proposal ``p_psi(D) / p_U(D) = E_q[psi(eta)]`` for any
hyperprior psi, so every Bayes factor below is a reweighting, not a training run.

Three questions, in the order they can kill the idea:

1. **Identifiability** (held-out simulations). Which coordinates does a dataset inform?
   Information gain over the uniform proposal and posterior contraction, per coordinate.
2. **Calibration** (held-out simulations). Is q a posterior or just a regressor? PIT of
   the true eta under q should be uniform; central-80% coverage should be 0.8.
3. **Real data**. Is each real target inside the simulated summary distribution (else q
   is extrapolating), where does its posterior sit, and what is the log Bayes factor of
   the origin -- the base's own pretraining prior -- against uniform and against the
   empirical-Bayes hyperprior fitted across the target's repeats?

Run from project root:
    pixi run python -m benchmarks.studies.eta_posterior simulate --n 40000
    pixi run python -m benchmarks.studies.eta_posterior control --n 120
    pixi run python -m benchmarks.studies.eta_posterior analyze
"""

import argparse
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
import torch
from benchmarks.corpus import CHAR_DIR, iter_real_tasks
from ebpfn.data import CharacterizationShape, TuningTask
from ebpfn.posterior.estimator import N_BINS, fit_posterior, posterior_probs, to_bins
from ebpfn.posterior.summaries import SUMMARY_VERSION, summarize
from ebpfn.priors import EtaSpace, sample_task
from ebpfn.utils import RandomRole, RandomStreams

OUT_DIR = Path("benchmarks/results/eta_posterior")
SIM_DIR = OUT_DIR / "simulations"
CONTROL_PATH = OUT_DIR / "control.parquet"
CHUNK = 500
MAX_ATTEMPTS = 50
SEED = 0
VAL_FRACTION = 0.15
COVERAGE = 0.8
# A real summary is out of support when its nearest simulation is farther than this
# quantile of simulation-to-simulation nearest-neighbour distances.
OOD_QUANTILE = 0.99
EM_STEPS = 200
# Control targets mimic the corpus: repeats are row subsamples of one larger draw.
N_REPEATS = 5
POOL_FACTOR = 3


def _xy(task: TuningTask) -> tuple[np.ndarray, np.ndarray]:
    x = np.vstack([task.probe_fit.X.to_numpy(), task.probe_score.X.to_numpy()]).astype(np.float64)
    return x, np.concatenate([task.probe_fit.y, task.probe_score.y]).astype(np.float64)


def corpus_shapes() -> list[CharacterizationShape]:
    """One shape per real dataset, so simulations match the corpus's own (n, p)."""
    shapes: dict[str, CharacterizationShape] = {}
    for dataset, _, task in iter_real_tasks(CHAR_DIR):
        shapes.setdefault(
            dataset,
            CharacterizationShape(
                task.probe_fit.X.height, task.probe_score.X.height, task.probe_fit.X.width, 0, "regression"
            ),
        )
    return list(shapes.values())


def _simulate_chunk(chunk: int, shapes: list[CharacterizationShape]) -> Path:
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    space = EtaSpace()
    streams = RandomStreams(SEED)
    rows: list[dict[str, float]] = []
    for index in range(chunk * CHUNK, (chunk + 1) * CHUNK):
        rng = streams.generator(RandomRole.SEARCH, "eta-posterior", index)
        vector = rng.uniform(size=space.dimension)
        shape = shapes[int(rng.integers(len(shapes)))]
        row = {"index": float(index)} | {
            f"eta.{name}": float(v) for name, v in zip(space.coordinate_names, vector, strict=True)
        }
        try:
            task = sample_task(
                shape, streams, "eta-posterior", index, eta=space.decode(vector), max_attempts=MAX_ATTEMPTS
            ).tuning
            rows.append(row | {"ok": 1.0} | summarize(*_xy(task), seed=index))
        except RuntimeError:
            rows.append(row | {"ok": 0.0})
    path = SIM_DIR / f"chunk_{chunk:04d}.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return path


def simulate(n: int, workers: int) -> None:
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    shapes = corpus_shapes()
    todo = [c for c in range(-(-n // CHUNK)) if not (SIM_DIR / f"chunk_{c:04d}.parquet").exists()]
    print(f"{len(todo)} chunks of {CHUNK} to simulate over {len(shapes)} corpus shapes, {workers} workers")
    with ProcessPoolExecutor(workers) as pool:
        for done, path in enumerate(pool.map(_simulate_chunk, todo, [shapes] * len(todo)), start=1):
            print(f"[{done}/{len(todo)}] {path.name}", flush=True)


def _control_target(target: int, shapes: list[CharacterizationShape]) -> list[dict[str, float]]:
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    space = EtaSpace()
    streams = RandomStreams(SEED)
    rng = streams.generator(RandomRole.SEARCH, "eta-control", target)
    vector = rng.uniform(size=space.dimension)
    shape = shapes[int(rng.integers(len(shapes)))]
    n = shape.n_probe_fit + shape.n_probe_score
    pool = CharacterizationShape(n * POOL_FACTOR - 1, 1, shape.p_numeric, 0, "regression")
    try:
        task = sample_task(
            pool, streams, "eta-control", target, eta=space.decode(vector), max_attempts=MAX_ATTEMPTS
        ).tuning
    except RuntimeError:
        return []
    x, y = _xy(task)
    eta_row = {f"eta.{name}": float(v) for name, v in zip(space.coordinate_names, vector, strict=True)}
    rows = []
    for repeat in range(N_REPEATS):
        rows_index = rng.choice(len(y), size=n, replace=False)
        rows.append(
            {"target": float(target), "repeat": float(repeat)}
            | eta_row
            | summarize(x[rows_index], y[rows_index], seed=SEED)
        )
    return rows


def control(n: int, workers: int) -> None:
    shapes = corpus_shapes()
    with ProcessPoolExecutor(workers) as pool:
        rows = [row for target in pool.map(_control_target, range(n), [shapes] * n) for row in target]
    pl.DataFrame(rows).write_parquet(CONTROL_PATH)
    print(f"{len(rows) // N_REPEATS} control targets -> {CONTROL_PATH}")


def _load_simulations() -> tuple[np.ndarray, np.ndarray, list[str], list[str], float]:
    frame = pl.concat([pl.read_parquet(p) for p in sorted(SIM_DIR.glob("chunk_*.parquet"))], how="diagonal")
    accept = float(frame["ok"].to_numpy().mean())
    frame = frame.filter(pl.col("ok") == 1.0)
    eta_cols = [c for c in frame.columns if c.startswith("eta.")]
    summary_cols = [c for c in frame.columns if c not in eta_cols and c not in ("index", "ok")]
    return frame.select(summary_cols).to_numpy(), frame.select(eta_cols).to_numpy(), summary_cols, eta_cols, accept


def _identifiability(probs: np.ndarray, eta: np.ndarray, names: list[str]) -> pl.DataFrame:
    """Per-coordinate information gain, contraction, PIT uniformity and coverage."""
    centers = (np.arange(N_BINS) + 0.5) / N_BINS
    true_bin = to_bins(eta)
    picked = np.take_along_axis(probs, true_bin[..., None], axis=-1)[..., 0]
    gain = np.log(N_BINS) + np.log(np.clip(picked, 1e-12, None))
    mean = (probs * centers).sum(-1)
    var = (probs * centers**2).sum(-1) - mean**2 + 1.0 / (12 * N_BINS**2)
    cdf = np.cumsum(probs, -1)
    within = eta * N_BINS - true_bin
    lower = np.take_along_axis(np.concatenate([np.zeros_like(cdf[..., :1]), cdf], -1), true_bin[..., None], -1)[..., 0]
    pit = lower + picked * within
    tail = (1 - COVERAGE) / 2
    rows = []
    for c, name in enumerate(names):
        hist = np.histogram(pit[:, c], bins=N_BINS, range=(0, 1))[0] / len(pit)
        rows.append(
            {
                "coordinate": name.removeprefix("eta."),
                "info_gain_nats": float(gain[:, c].mean()),
                "contraction": float(1 - var[:, c].mean() * 12),
                "pit_max_dev": float(np.abs(hist - 1 / N_BINS).max() * N_BINS),
                f"coverage_{COVERAGE}": float(np.mean((pit[:, c] > tail) & (pit[:, c] < 1 - tail))),
            }
        )
    return pl.DataFrame(rows).sort("info_gain_nats", descending=True)


def _em_hyperprior(probs: np.ndarray) -> np.ndarray:
    """NPMLE histogram psi per coordinate across repeats: ``psi_b <- mean_r psi_b q_rb / sum``."""
    psi = np.full(probs.shape[1:], 1.0 / N_BINS)
    for _ in range(EM_STEPS):
        weighted = psi[None] * probs
        psi = (weighted / weighted.sum(-1, keepdims=True)).mean(0)
    return psi


def _score_target(
    probs: np.ndarray, origin_bin: np.ndarray, true_bin: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Per-coordinate log Bayes factors for one target's repeats, under the factorized posterior.

    ``log p_psi(D_r) / p_U(D_r) = log(B * sum_b psi_b q_rb)``, summed over repeats. The EB
    hyperprior is fitted on the other repeats and scored on the held-out one; an in-sample
    NPMLE rewards any non-flat posterior, noise included.
    """

    def at(bins: np.ndarray) -> np.ndarray:
        index = np.broadcast_to(bins[None, :, None], (len(probs), probs.shape[1], 1))
        return np.log(N_BINS * np.clip(np.take_along_axis(probs, index, -1)[..., 0], 1e-12, None)).sum(0)

    held_out = [_em_hyperprior(np.delete(probs, r, axis=0)) for r in range(len(probs))]
    eb = np.stack([np.log(N_BINS * (probs[r] * held_out[r]).sum(-1)) for r in range(len(probs))]).sum(0)
    origin = at(origin_bin)
    scores = {
        "log_bf_origin_vs_uniform": origin,
        "log_bf_eb_loo_vs_uniform": eb,
        "log_bf_eb_loo_vs_origin": eb - origin,
    }
    if true_bin is not None:
        scores["log_bf_truth_vs_uniform"] = at(true_bin)
    return scores


def analyze() -> None:
    summaries, eta, summary_cols, eta_cols, accept = _load_simulations()
    print(
        f"{len(eta)} accepted simulations (acceptance {accept:.3f}), {len(summary_cols)} summaries, {len(eta_cols)} coords"
    )
    model, scaler, fit_stats = fit_posterior(summaries, eta, val_fraction=VAL_FRACTION, seed=SEED)
    print(f"fit: {fit_stats}")

    # The same split fit_posterior held out, so identifiability is out-of-sample.
    order = np.random.default_rng(SEED).permutation(len(eta))
    held = order[: int(len(eta) * VAL_FRACTION)]
    ident = _identifiability(posterior_probs(model, scaler, summaries[held]), eta[held], eta_cols)
    ident.write_csv(OUT_DIR / "identifiability.csv")
    with pl.Config(tbl_rows=-1, float_precision=3):
        print(ident)

    # Real corpus: support check, posterior, and Bayes factors against the origin.
    space = EtaSpace()
    origin_bin = to_bins(space.origin_vector)
    scaled_sims = scaler(summaries)
    sample = scaled_sims[np.random.default_rng(SEED).permutation(len(scaled_sims))[:2000]]
    sim_nn = np.sort(np.linalg.norm(sample[:, None] - scaled_sims[None, :5000], axis=-1), axis=1)[:, 1]
    threshold = float(np.quantile(sim_nn, OOD_QUANTILE))

    names = [c.removeprefix("eta.") for c in eta_cols]
    centers = (np.arange(N_BINS) + 0.5) / N_BINS
    real: dict[str, list[np.ndarray]] = {}
    for dataset, _, task in iter_real_tasks(CHAR_DIR):
        real.setdefault(dataset, []).append(np.array(list(summarize(*_xy(task), seed=SEED).values())))
    rows, coords = [], []
    for dataset, vectors in real.items():
        stacked = np.stack(vectors)
        nn_dist = np.linalg.norm(scaler(stacked)[:, None] - scaled_sims[None], axis=-1).min(1)
        probs = posterior_probs(model, scaler, stacked)
        scores = _score_target(probs, origin_bin)
        rows.append(
            {"dataset": dataset, "repeats": len(vectors), "nn_dist_over_threshold": float(nn_dist.max() / threshold)}
            | {k: float(v.sum()) for k, v in scores.items()}
        )
        psi = _em_hyperprior(probs)
        for c, name in enumerate(names):
            coords.append(
                {
                    "dataset": dataset,
                    "coordinate": name,
                    "origin": float(space.origin_vector[c]),
                    "post_mean": float((probs[:, c] * centers).sum(-1).mean()),
                    "eb_mode": float(centers[psi[c].argmax()]),
                    "log_bf_eb_loo_vs_origin": float(scores["log_bf_eb_loo_vs_origin"][c]),
                }
            )

    # The same procedure on simulated targets with known eta: the null the real rows are read against.
    control_rows = []
    if CONTROL_PATH.exists():
        control = pl.read_parquet(CONTROL_PATH)
        for target, group in control.group_by("target", maintain_order=True):
            probs = posterior_probs(model, scaler, group.select(summary_cols).to_numpy())
            true_bin = to_bins(group.select(eta_cols).to_numpy()[0])
            scores = _score_target(probs, origin_bin, true_bin)
            control_rows.append({"target": int(target[0])} | {k: float(v.sum()) for k, v in scores.items()})
        control_frame = pl.DataFrame(control_rows)
        control_frame.write_csv(OUT_DIR / "control_bayes_factors.csv")
        with pl.Config(float_precision=3, tbl_cols=-1):
            print("simulated control (known eta):")
            print(control_frame.drop("target").describe())

    real_frame = pl.DataFrame(rows)
    coord_frame = pl.DataFrame(coords)
    real_frame.write_csv(OUT_DIR / "real_bayes_factors.csv")
    coord_frame.write_csv(OUT_DIR / "real_coordinates.csv")
    with pl.Config(tbl_rows=-1, float_precision=3, tbl_cols=-1):
        print(real_frame)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(
            {
                "summary_version": SUMMARY_VERSION,
                "space": space.schema()["hash"],
                "acceptance": accept,
                "n_simulations": len(eta),
                "fit": fit_stats,
                "ood_threshold": threshold,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sim = sub.add_parser("simulate")
    sim.add_argument("--n", type=int, required=True)
    sim.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ctl = sub.add_parser("control")
    ctl.add_argument("--n", type=int, required=True)
    ctl.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    sub.add_parser("analyze")
    args = parser.parse_args()
    if args.command == "simulate":
        simulate(args.n, args.workers)
    elif args.command == "control":
        control(args.n, args.workers)
    else:
        analyze()


if __name__ == "__main__":
    main()
