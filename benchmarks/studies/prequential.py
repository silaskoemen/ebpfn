"""Prequential prior mass: how much predictive mass does each model's prior put on a dataset?

A PFN approximates the posterior predictive of its prior, so chaining its predictions
row by row approximates the prior's marginal likelihood of the dataset,
``log p(D) = sum_t log p(y_t | x_t, D_<t)``. The difference between two models is a
Bayes factor between their priors (plus each model's approximation error), with no
descriptors or kernels in between.

Rows are scored in blocks: every row in ``[a, b)`` is predicted from the first ``a``
rows, for block bounds ``BOUNDS`` (a valid, slightly lossy prequential code). Scores are
averaged over random row orderings and reported per row and per context-size block, so
a gap shows *where* (at which context size) one prior loses mass.

Every model scores the same rows on the same scale: ``y`` is standardized once per
dataset, so log densities are comparable across models. Quantile heads are read on the
coarse knot grid (``metrics.COARSE_NLL_STRIDE``) and reported raw as well.

A density is unbounded on a discrete target (counts, ratings), so each row is also scored
by the mass its model puts on the row's bin, ``log P(y in bin)``. Bins split the dataset's
distinct target values at their midpoints; on a continuous target that is the density
times a fixed width, so model differences match the density's.

The nanoprior control scores samples from the prior the base was trained on. There the
base should hold the most mass; if it does not, approximation error dominates and the
real-data gaps cannot be read as prior gaps.

Corpus datasets are cut to ``--rows`` rows (the corpus build's cap by default) and the
controls are drawn at that size too, so both reach the same largest context. Results go to
one directory per row cap.

Run from project root (TabPFN needs ``TABPFN_TOKEN``):
    pixi run python -m benchmarks.studies.prequential [--models base kumo ...] [--rows 2048]
"""

import argparse
import json
import os
import zlib
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

import numpy as np
import polars as pl
import scipy.stats
import torch
from benchmarks.base import released_base_checkpoint
from benchmarks.baselines import gp_fit_predict
from benchmarks.corpus import CHAR_DIR, iter_real_tasks
from benchmarks.data.regression_corpus import ROWS, load_corpus
from ebpfn.data import CharacterizationShape
from ebpfn.pfn.data import NanopriorTaskSource
from ebpfn.pfn.metrics import COARSE_NLL_STRIDE
from ebpfn.pfn.model import EBPFNModel
from ebpfn.pfn.quantile import QuantileDistribution, default_levels
from ebpfn.pfn.train import configure_device_memory, load_checkpoint, release_device_cache, select_device
from ebpfn.utils import RandomStreams

OUT_DIR = Path("benchmarks/results/prequential")
SEED = 0
BOUNDS = (8, 16, 32, 64, 128, 256, 512, 1024, 2048)
ORDERINGS = 5
PRIOR_DATASETS = 8  # nanoprior control datasets per real target shape
PRIOR_ORDERINGS = 1  # prior rows are exchangeable already
ESTIMATORS = 1  # single forward pass: the prior's predictive, not an ensemble over preprocessings
MAX_ATTEMPTS = 50
QUERY_CHUNK = 256  # query rows per forward pass: bounds memory at large blocks
ROW_CHUNK = 256  # rows per row-stage pass inside one forward: bounds rows x columns^2 attention
GP_MAX_CONTEXT = 1024  # exact GP is cubic in the context; larger ones are subsampled
MODELS = ("base", "tabiclv2", "kumo", "tabpfn", "gp")  # TabFM regresses a point, no density to score

MIN_MASS = 1e-30  # floors a bin mass a model rounds to zero: at most 69 nats per row

# (x_context, y_context, x_query, y_query, bin_low, bin_high)
#   -> (log density, coarse log density, log bin mass) per query row
Scorer = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]


def quantile_log_density(
    distribution: QuantileDistribution, quantiles: torch.Tensor, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    target = torch.as_tensor(y, dtype=torch.float32, device=quantiles.device)
    raw = distribution.log_prob(quantiles, target)
    coarse = -distribution.coarse_nll(quantiles, target, stride=COARSE_NLL_STRIDE)
    return raw.cpu().double().numpy(), coarse.cpu().double().numpy()


def log_bin_mass(cdf: Callable[[np.ndarray], np.ndarray], low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """``log(F(high) - F(low))`` per row, for a row-wise CDF evaluated in float64."""
    return np.log(np.maximum(cdf(high) - cdf(low), MIN_MASS))


def quantile_log_mass(levels: torch.Tensor, quantiles: torch.Tensor, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Bin mass of a quantile head, in float64 on the CPU (MPS has no float64)."""
    distribution = QuantileDistribution(levels.detach().cpu().double())
    quantiles = quantiles.detach().cpu().double()

    def cdf(y: np.ndarray) -> np.ndarray:
        return distribution.cdf(quantiles, torch.from_numpy(y.astype(np.float64))).numpy()

    return log_bin_mass(cdf, low, high)


def base_scorer(device: torch.device) -> Scorer:
    """The released base as studies use it: raw x, y standardized on the context."""
    model, _ = load_checkpoint(released_base_checkpoint(), map_location=device)
    return model_scorer(model.to(device), device)


def model_scorer(model: EBPFNModel, device: torch.device) -> Scorer:
    """Score with an in-memory model the way :func:`base_scorer` scores the base."""
    model.eval()
    distribution = model.distribution
    if not isinstance(distribution, QuantileDistribution):
        raise TypeError("prequential scoring needs a quantile head")

    @torch.no_grad()
    def score(
        x_ctx: np.ndarray, y_ctx: np.ndarray, x_q: np.ndarray, y_q: np.ndarray, low: np.ndarray, high: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean, std = y_ctx.mean(), y_ctx.std() or 1.0
        y_train = torch.from_numpy(((y_ctx - mean) / std)[None].astype(np.float32)).to(device)
        chunks = []
        for start in range(0, len(x_q), QUERY_CHUNK):  # query rows attend only to the context: chunking is exact
            x = np.concatenate([x_ctx, x_q[start : start + QUERY_CHUNK]])[None].astype(np.float32)
            chunks.append(model(torch.from_numpy(x).to(device), y_train, row_chunk=ROW_CHUNK)[0])
        quantiles = torch.cat(chunks)
        raw, coarse = quantile_log_density(distribution, quantiles, (y_q - mean) / std)
        mass = quantile_log_mass(distribution.levels, quantiles, (low - mean) / std, (high - mean) / std)
        return raw - np.log(std), coarse - np.log(std), mass  # Jacobian back to the shared scale; mass needs none

    return score


def sdm_scorer(name: str, device: torch.device) -> Scorer:
    """A ``structured-data-models`` model with its default preprocessing recipe."""
    from sdm.models import KumoTabular, TabICLv2

    model = {"tabiclv2": TabICLv2, "kumo": KumoTabular}[name](task="regression", device=device)
    distribution = QuantileDistribution(default_levels(999)).to(device)  # q001..q999

    def score(
        x_ctx: np.ndarray, y_ctx: np.ndarray, x_q: np.ndarray, y_q: np.ndarray, low: np.ndarray, high: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        def tensor(a: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(a.astype(np.float32)).to(device)

        model.fit(
            tensor(x_ctx),
            tensor(y_ctx[:, None]),
            num_estimators=ESTIMATORS,
            generator=torch.Generator(device).manual_seed(SEED),
        )
        chunks = [
            model.predict(tensor(x_q[start : start + QUERY_CHUNK])).numerical
            for start in range(0, len(x_q), QUERY_CHUNK)
        ]
        quantiles = torch.cat(chunks)
        model.clear()
        raw, coarse = quantile_log_density(distribution, quantiles, y_q)
        return raw, coarse, quantile_log_mass(distribution.levels, quantiles, low, high)

    return score


def tabpfn_scorer(device: torch.device) -> Scorer:
    from sklearn.base import clone
    from tabpfn import TabPFNRegressor

    template = TabPFNRegressor(n_estimators=ESTIMATORS, device=device, random_state=SEED)

    def score(
        x_ctx: np.ndarray, y_ctx: np.ndarray, x_q: np.ndarray, y_q: np.ndarray, low: np.ndarray, high: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Near-constant or heavily tied contexts (only seen in prior samples) break TabPFN's bar
        # construction: the block is scored undefined and counted, not dropped silently.
        # A refitted regressor carries state from its previous fit (a small context scored after
        # a large one moves by nats), so every block fits a fresh clone.
        try:
            regressor = clone(template).fit(x_ctx, y_ctx)
            outs = [
                regressor.predict(x_q[start : start + QUERY_CHUNK], output_type="full")
                for start in range(0, len(x_q), QUERY_CHUNK)
            ]
            logits = torch.cat([out["logits"] for out in outs])
            criterion = outs[0]["criterion"]
            with torch.no_grad():
                log_density = -criterion(logits, torch.as_tensor(y_q, dtype=logits.dtype, device=logits.device))
                edges = torch.as_tensor(np.stack([low, high], axis=1), dtype=logits.dtype, device=logits.device)
                cdf = criterion.cdf(logits, edges).cpu().double().numpy()
        except (AssertionError, ValueError):
            undefined = np.full(len(y_q), np.nan)
            return undefined, undefined, undefined
        log_density = log_density.cpu().double().numpy()
        mass = np.log(np.maximum(cdf[:, 1] - cdf[:, 0], MIN_MASS))  # float32 CDF: fine above ~1e-6 mass
        return log_density, log_density, mass

    return score


def gp_scorer() -> Scorer:
    """GP on context-standardized x and y; a Gaussian density has no knot grid to coarsen."""

    def score(
        x_ctx: np.ndarray, y_ctx: np.ndarray, x_q: np.ndarray, y_q: np.ndarray, low: np.ndarray, high: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_mean, x_std = x_ctx.mean(0), x_ctx.std(0)
        x_std[x_std == 0] = 1.0
        y_mean, y_std = y_ctx.mean(), y_ctx.std() or 1.0
        mean, std = gp_fit_predict(
            (x_ctx - x_mean) / x_std,
            (y_ctx - y_mean) / y_std,
            (x_q - x_mean) / x_std,
            max_samples=GP_MAX_CONTEXT,
            seed=SEED,
        )
        mean, std = mean * y_std + y_mean, std * y_std
        log_density = -0.5 * np.log(2 * np.pi * std**2) - (y_q - mean) ** 2 / (2 * std**2)
        mass = log_bin_mass(lambda y: scipy.stats.norm.cdf(y, mean, std), low, high)
        return log_density, log_density, mass

    return score


def build_scorer(name: str, device: torch.device) -> Scorer:
    if name == "base":
        return base_scorer(device)
    if name in ("tabiclv2", "kumo"):
        return sdm_scorer(name, device)
    if name == "tabpfn":
        return tabpfn_scorer(device)
    if name == "gp":
        return gp_scorer()
    raise ValueError(f"unknown model {name!r}")


def target_bins(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each row's bin: its distinct value's midpoints to the neighbouring values (half a gap past the ends)."""
    values = np.unique(y)
    if len(values) == 1:
        return y - 0.5, y + 0.5
    midpoints = (values[1:] + values[:-1]) / 2
    edges = np.concatenate([[2 * values[0] - midpoints[0]], midpoints, [2 * values[-1] - midpoints[-1]]])
    index = np.searchsorted(values, y)
    return edges[index], edges[index + 1]


def prequential(score: Scorer, x: np.ndarray, y: np.ndarray, orderings: int, key: str) -> pl.DataFrame:
    """Per-row log densities and bin masses over ``orderings`` random row orders, blocked at ``BOUNDS``."""
    n = len(y)
    low, high = target_bins(y)
    bounds = [b for b in BOUNDS if b < n] + [n]
    frames = []
    for ordering in range(orderings):
        order = np.random.default_rng([SEED, zlib.crc32(key.encode()), ordering]).permutation(n)
        xs, ys, lows, highs = x[order], y[order], low[order], high[order]
        for start, stop in pairwise(bounds):
            q = slice(start, stop)
            raw, coarse, mass = score(xs[:start], ys[:start], xs[q], ys[q], lows[q], highs[q])
            frames.append(
                pl.DataFrame(
                    {
                        "ordering": ordering,
                        "context": start,
                        "position": np.arange(start, stop),
                        "row": order[start:stop],
                        "log_density": raw,
                        "log_density_coarse": coarse,
                        "log_mass": mass,
                    }
                )
            )
    return pl.concat(frames)


def results_dir(rows: int) -> Path:
    return OUT_DIR / f"rows{rows}"


def load_datasets(prior_datasets: int, rows: int = ROWS) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    """``(source, name, x, y)`` for every real target and its nanoprior controls, then the extended corpus.

    Controls are drawn for the original targets only; they calibrate the models, not the datasets.
    They take each target's feature count and fit/score split at ``rows`` rows (at least the
    target's own), so the control covers every context the corpus reaches.
    """
    datasets = []
    source = NanopriorTaskSource(RandomStreams(SEED), max_attempts=MAX_ATTEMPTS)
    for name, repeat, task in iter_real_tasks(CHAR_DIR):
        if repeat != 0:
            continue
        x = np.concatenate([task.probe_fit.X.to_numpy(), task.probe_score.X.to_numpy()]).astype(np.float64)
        y = np.concatenate([task.probe_fit.y, task.probe_score.y]).astype(np.float64)
        datasets.append(("real", name, x, y))
        n = max(rows, len(y))
        n_fit = round(n * task.probe_fit.X.height / len(y))
        shape = CharacterizationShape(n_fit, n - n_fit, x.shape[1], 0, "regression")
        for member, prior in enumerate(source.sample_batch(prior_datasets, shape, "prequential", name)):
            px = np.concatenate([prior.probe_fit.X.to_numpy(), prior.probe_score.X.to_numpy()]).astype(np.float64)
            py = np.concatenate([prior.probe_fit.y, prior.probe_score.y]).astype(np.float64)
            datasets.append(("nanoprior", f"{name}/{member}", px, py))
    datasets += [("real", name, x, y) for name, x, y in load_corpus(rows)]
    return datasets


def summarize(frame: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Mean log density per row (nats) per dataset and model, overall and per context block, with the gap to the base."""
    per_dataset = frame.group_by("source", "dataset", "model").agg(
        log_density=pl.col("log_density").mean(),
        log_density_coarse=pl.col("log_density_coarse").mean(),
        log_mass=pl.col("log_mass").mean(),
        rows=pl.len(),
    )
    base = per_dataset.filter(pl.col("model") == "base").select("source", "dataset", base_coarse="log_density_coarse")
    per_dataset = (
        per_dataset.join(base, on=["source", "dataset"], how="left")
        .with_columns(
            vs_base=pl.col("log_density_coarse") - pl.col("base_coarse"),
        )
        .drop("base_coarse")
        .sort("source", "dataset", "model")
    )
    per_block = (
        frame.group_by("source", "dataset", "model", "context")
        .agg(
            log_density_coarse=pl.col("log_density_coarse").mean(),
        )
        .sort("source", "dataset", "model", "context")
    )
    return per_dataset, per_block


def is_scored(path: Path) -> bool:
    """A result file exists and carries every score column (older runs lack ``log_mass``)."""
    return path.exists() and "log_mass" in pl.read_parquet_schema(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--datasets", nargs="+", default=None, help="real target names; default all")
    parser.add_argument("--prior-datasets", type=int, default=PRIOR_DATASETS)
    parser.add_argument("--rows", type=int, default=ROWS, help="rows per corpus dataset and control")
    args = parser.parse_args()
    out_dir = results_dir(args.rows)

    device = select_device("auto")
    configure_device_memory(device)
    datasets = [
        d
        for d in load_datasets(args.prior_datasets, args.rows)
        if args.datasets is None or d[1].split("/")[0] in args.datasets
    ]
    print(
        f"device={device} | models={args.models} | {len(datasets)} datasets | bounds={BOUNDS} | estimators={ESTIMATORS}",
        flush=True,
    )

    for model_name in args.models:
        out = out_dir / model_name
        out.mkdir(parents=True, exist_ok=True)
        pending = [d for d in datasets if not is_scored(out / f"{d[0]}__{d[1].replace('/', '_')}.parquet")]
        if not pending:
            continue
        score = build_scorer(model_name, device)
        for source, name, x, y in pending:
            y_std = (y - y.mean()) / (y.std() or 1.0)  # one shared scale for every model
            orderings = ORDERINGS if source == "real" else PRIOR_ORDERINGS
            frame = prequential(score, x, y_std, orderings, f"{source}/{name}")
            frame = frame.with_columns(source=pl.lit(source), dataset=pl.lit(name), model=pl.lit(model_name))
            frame.write_parquet(out / f"{source}__{name.replace('/', '_')}.parquet")
            print(f"  {model_name} {source}/{name}: {frame['log_density_coarse'].mean():.3f} nats/row", flush=True)
            release_device_cache(device)

    frame = pl.concat([pl.read_parquet(p) for p in out_dir.glob("*/*.parquet")], how="diagonal_relaxed")
    per_dataset, per_block = summarize(frame)
    per_dataset.write_csv(out_dir / "summary.csv")
    per_block.write_csv(out_dir / "summary_by_context.csv")
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "bounds": list(BOUNDS),
                "orderings": ORDERINGS,
                "prior_datasets": args.prior_datasets,
                "rows": args.rows,
                "gp_max_context": GP_MAX_CONTEXT,
                "prior_orderings": PRIOR_ORDERINGS,
                "estimators": ESTIMATORS,
                "coarse_stride": COARSE_NLL_STRIDE,
                "tabpfn_token": bool(os.environ.get("TABPFN_TOKEN")),
            },
            indent=2,
        )
    )
    real = per_dataset.filter(pl.col("source") == "real").pivot(
        on="model", index="dataset", values="log_density_coarse"
    )
    prior = (
        per_dataset.filter(pl.col("source") == "nanoprior").group_by("model").agg(pl.col("log_density_coarse").mean())
    )
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=3, tbl_width_chars=250):
        print("real targets, mean log density per row (nats, higher = more prior mass):", real, sep="\n")
        print("nanoprior control, mean over datasets:", prior, sep="\n")


if __name__ == "__main__":
    main()
