"""Selection-prize gate: does a prior aimed at a target beat the origin-prior fine-tune?

Each tilt moves one eta block a moderate step from the origin and is fine-tuned with the
recipe of ``small_context``'s ``large_1e-5`` control (256-1024 context rows, lr 1e-5,
1000 steps) at the control's seed. Nanoprior seeds tasks from identity rather than eta,
so tilt and control see common random numbers and differ only in the mixture. Any
fine-tune drifts at these contexts (-0.03 to -0.08 nats per row, the same at every
seed), so tilts are read against the control, never the released base.

* **mechanism** -- half the mechanism-role mass moved onto one of the four groups;
* **graph** -- ``z.graph.n_nodes`` shifted toward small or large graphs;
* **dispersion** -- ``z.weights.dispersion`` shifted toward even or concentrated weights.

``analyze`` reads real datasets only, as bin mass minus the control. Rows scored at the
256 and 512 context blocks are VAL, rows of the 1024 block are TEST, so selection and
report never share a scored row. It reports each tilt's TEST effect, the best global tilt
chosen on VAL (Delta_global), per-target selection on VAL (Delta_spec, the control itself
being one of the options) and the TEST oracle (headroom). The control's other seeds can
stand in as tilts, which turns the same procedure into its own null. ``--replicate`` keeps
the seed-1 per-target picks and re-reads each one at every seed its tilt was trained at,
against the control at that seed: a pick that only won on seed-1 luck falls to zero there.

Run from project root (wrap in caffeinate):
    pixi run python -m benchmarks.studies.prior_tilts train [--tilts mechanism_smooth ...] [--seed 1]
    pixi run python -m benchmarks.studies.prior_tilts evaluate [--rows 2048] [--seed 1]
    pixi run python -m benchmarks.studies.prior_tilts analyze [--rows 2048] [--null | --replicate]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from benchmarks.base import released_base_checkpoint
from benchmarks.data.regression_corpus import ROWS
from benchmarks.studies import small_context
from benchmarks.studies.prequential import PRIOR_DATASETS, load_datasets
from ebpfn.pfn.train import configure_device_memory, load_checkpoint, select_device
from ebpfn.priors.nanoprior.eta import PriorEta
from ebpfn.priors.nanoprior.mechanisms import GROUP_ORDER, ORIGIN
from ebpfn.priors.nanoprior.scalars import ORIGIN_COORD

OUT_DIR = Path("benchmarks/results/prior_tilts")
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
CONTROL = "large_1e-5"
SEED = 1  # the control's seed, so tilts share its task stream
CONTROL_SEEDS = (1, 2, 3)  # seeds the control was trained at: the seeds a pick can replicate at
STEP = 1000
TILT_MIX = 0.5  # share of the mechanism-role mass moved onto the tilted group
SCALAR_STEP = 0.25  # coordinate distance from the origin: intervals slide by 3**0.5
VAL_CONTEXTS = (256, 512)
TEST_CONTEXTS = (1024,)
BOOTSTRAPS = 2000
KEYS = ["source", "dataset", "ordering", "row", "context"]


def mechanism_tilt(group: str) -> PriorEta:
    target = [float(name == group) for name in GROUP_ORDER]
    return PriorEta(mechanism=tuple((1 - TILT_MIX) * o + TILT_MIX * t for o, t in zip(ORIGIN, target, strict=True)))


def scalar_tilt(site: str, direction: int) -> PriorEta:
    return PriorEta(scalars={site: (ORIGIN_COORD + direction * SCALAR_STEP, ORIGIN_COORD)})


TILTS = {
    **{f"mechanism_{group}": mechanism_tilt(group) for group in GROUP_ORDER},
    "graph_small": scalar_tilt("z.graph.n_nodes", -1),
    "graph_large": scalar_tilt("z.graph.n_nodes", 1),
    "dispersion_low": scalar_tilt("z.weights.dispersion", -1),
    "dispersion_high": scalar_tilt("z.weights.dispersion", 1),
}


def run_name(tilt: str, seed: int) -> str:
    return f"{tilt}_seed{seed}_step{STEP}"


def train(tilt: str, seed: int, device: torch.device) -> None:
    fit, lr, distill_above = small_context.ARMS[CONTROL]
    checkpoint = CHECKPOINT_DIR / f"{run_name(tilt, seed)}.pt"
    losses = small_context.fine_tune(tilt, fit, lr, distill_above, seed, device, {STEP: checkpoint}, TILTS[tilt])
    (OUT_DIR / f"{tilt}_seed{seed}_losses.json").write_text(json.dumps(losses))
    (OUT_DIR / f"{tilt}_eta.json").write_text(json.dumps(TILTS[tilt].to_dict(), indent=2))


def arm_rows(directory: Path, name: str) -> pl.DataFrame:
    frame = pl.read_parquet(directory / "*.parquet").filter(pl.col("source") == "real")
    return frame.select(*KEYS, pl.col("log_mass").alias(name))


def per_dataset(control: pl.DataFrame, arm: pl.DataFrame, name: str) -> pl.DataFrame:
    """Mean arm-minus-control bin mass per dataset on the VAL and TEST rows."""
    split = (
        pl.when(pl.col("context").is_in(VAL_CONTEXTS))
        .then(pl.lit("val"))
        .when(pl.col("context").is_in(TEST_CONTEXTS))
        .then(pl.lit("test"))
    )
    joined = control.join(arm, on=KEYS).with_columns(split=split).drop_nulls("split")
    means = joined.group_by("dataset", "split").agg((pl.col(name) - pl.col("control")).mean().alias("delta"))
    wide = means.pivot(on="split", index="dataset", values="delta").drop_nulls()
    return wide.select("dataset", "val", "test", arm=pl.lit(name))


def bootstrap(values: np.ndarray, statistic, rng: np.random.Generator) -> tuple[float, float]:
    """95% interval of ``statistic`` over datasets resampled with replacement."""
    draws = statistic(rng.choice(values, size=(BOOTSTRAPS, len(values))), axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def control_dir(rows: int, seed: int) -> Path:
    return small_context.OUT_DIR / f"rows{rows}" / f"{CONTROL}_seed{seed}_step{STEP}"


def tilt_table(rows: int, seed: int) -> pl.DataFrame | None:
    """Per-dataset deltas of every tilt evaluated at ``seed``, against the control at that seed."""
    arms = {tilt: OUT_DIR / f"rows{rows}" / run_name(tilt, seed) for tilt in TILTS}
    arms = {tilt: path for tilt, path in arms.items() if path.exists()}
    if not arms:
        return None
    control = arm_rows(control_dir(rows, seed), "control")
    return pl.concat([per_dataset(control, arm_rows(path, tilt), tilt) for tilt, path in arms.items()])


def per_target(table: pl.DataFrame, by: str) -> pl.DataFrame:
    """Each dataset's best option on ``by``, the control (zero delta) being one of the options."""
    control = table.select("dataset").unique().with_columns(val=pl.lit(0.0), test=pl.lit(0.0), arm=pl.lit("control"))
    options = pl.concat([table, control], how="diagonal_relaxed")
    return options.sort(by, descending=True).group_by("dataset").first()


def column_mean(column: pl.Series) -> float:
    """Mean of a numeric or boolean column, 0 when it is empty."""
    return float(column.to_numpy().mean()) if len(column) else 0.0


def replicate(rows: int) -> None:
    """Re-read each seed-1 per-target pick at every seed its tilt was trained at."""
    table = tilt_table(rows, SEED)
    if table is None:
        raise FileNotFoundError(f"no tilt evaluated at seed {SEED}")
    picks = per_target(table, "val").filter(pl.col("arm") != "control")
    picks = picks.select("dataset", "arm", "val", pl.col("test").alias(f"test_seed{SEED}"))
    for seed in CONTROL_SEEDS:
        table = tilt_table(rows, seed) if seed != SEED else None
        if table is not None:
            picks = picks.join(
                table.select("dataset", "arm", pl.col("test").alias(f"test_seed{seed}")),
                on=["dataset", "arm"],
                how="left",
            )
    seeds = [int(name.removeprefix("test_seed")) for name in picks.columns if name.startswith("test_seed")]
    replicated = picks.drop_nulls().sort(f"test_seed{SEED}", descending=True)
    summary = {
        "picks": picks.height,
        "replicated": replicated.height,
        "seeds": seeds,
        "test_mean": {seed: column_mean(replicated[f"test_seed{seed}"]) for seed in seeds},
        "test_win": {seed: column_mean(replicated[f"test_seed{seed}"] > 0) for seed in seeds},
        "rows": replicated.to_dicts(),
    }
    with pl.Config(tbl_rows=-1, float_precision=3, fmt_str_lengths=50):
        print(replicated)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    (OUT_DIR / f"rows{rows}" / "replication.json").write_text(json.dumps(summary, indent=2))


def analyze(rows: int, null: bool) -> None:
    if null:  # the control's other seeds as tilts: what selection finds when no tilt differs
        arms = {
            path.name: path
            for path in sorted((small_context.OUT_DIR / f"rows{rows}").glob(f"{CONTROL}_seed*_step{STEP}"))
            if path != control_dir(rows, SEED)
        }
        control = arm_rows(control_dir(rows, SEED), "control")
        table = pl.concat([per_dataset(control, arm_rows(path, name), name) for name, path in arms.items()])
    else:
        table = tilt_table(rows, SEED)
        if table is None:
            raise FileNotFoundError(f"no tilt evaluated at seed {SEED}")
        arms = table["arm"].unique(maintain_order=True).to_list()
    rng = np.random.default_rng(0)

    by_tilt = (
        table.group_by("arm")
        .agg(
            val=pl.col("val").median(),
            test=pl.col("test").median(),
            test_win=(pl.col("test") > 0).mean(),
            val_test_rho=pl.corr("val", "test", method="spearman"),
            n=pl.len(),
        )
        .sort("test", descending=True)
    )
    best_global = by_tilt.sort("val", descending=True)["arm"][0]
    chosen = per_target(table, "val")
    oracle = per_target(table, "test")
    global_test = table.filter(pl.col("arm") == best_global)["test"].to_numpy()
    chosen_test = chosen["test"].to_numpy()

    summary = {
        "arms": list(arms),
        "datasets": table["dataset"].n_unique(),
        "global": {
            "arm": best_global,
            "test_median": float(np.median(global_test)),
            "ci": bootstrap(global_test, np.median, rng),
        },
        "spec": {
            "test_median": float(np.median(chosen_test)),
            "test_mean": float(np.mean(chosen_test)),
            "ci_mean": bootstrap(chosen_test, np.mean, rng),
            "chose_control": column_mean(chosen["arm"] == "control"),
            "choices": dict(chosen["arm"].value_counts().iter_rows()),
        },
        "oracle": {"test_median": float(np.median(oracle["test"].to_numpy()))},
    }
    with pl.Config(tbl_rows=-1, float_precision=3):
        print(by_tilt)
    print(json.dumps(summary, indent=2))
    out = OUT_DIR / f"rows{rows}" / ("analysis_null.json" if null else "analysis.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("train", "evaluate", "analyze"))
    parser.add_argument("--tilts", nargs="+", default=list(TILTS), choices=list(TILTS))
    parser.add_argument("--rows", type=int, default=ROWS, help="evaluate/analyze: rows per corpus dataset and control")
    parser.add_argument("--seed", type=int, default=SEED, help="train/evaluate: the control seed to share")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--null", action="store_true", help="analyze: the control's other seeds stand in as tilts")
    mode.add_argument("--replicate", action="store_true", help="analyze: re-read seed-1 picks at other seeds")
    args = parser.parse_args()

    if args.command == "analyze":
        if args.replicate:
            replicate(args.rows)
        else:
            analyze(args.rows, args.null)
        return
    if args.seed not in CONTROL_SEEDS:
        parser.error(f"--seed must be one the control was trained at: {CONTROL_SEEDS}")
    device = select_device("auto")
    configure_device_memory(device)
    if args.command == "train":
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        for tilt in args.tilts:
            if not (CHECKPOINT_DIR / f"{run_name(tilt, args.seed)}.pt").exists():
                train(tilt, args.seed, device)
        return

    datasets = load_datasets(PRIOR_DATASETS, args.rows)
    model, _ = load_checkpoint(released_base_checkpoint(), map_location=device)
    model.to(device)
    for tilt in args.tilts:
        checkpoint = CHECKPOINT_DIR / f"{run_name(tilt, args.seed)}.pt"
        if checkpoint.exists():
            model.load_state_dict(torch.load(checkpoint, map_location=device))
            small_context.evaluate(model, device, datasets, OUT_DIR / f"rows{args.rows}" / run_name(tilt, args.seed))
            print(f"  {run_name(tilt, args.seed)} evaluated at {args.rows} rows", flush=True)


if __name__ == "__main__":
    main()
