"""PFN feasibility study: profile training/inference cost and run a smoke train.

Produces the same artifact family as the other studies (parquet + json + summary.md +
environment.json + run.log) so the cost profile and a loss-decreases sanity check are
reproducible. The claim-bearing surrogate-validation study is separate and gated.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from benchmarks.studies.study_logging import configure_study_logging
from ebpfn.config.pfn import PfnStudyConfig
from ebpfn.pfn.feasibility import profile
from ebpfn.pfn.train import train_pfn
from ebpfn.utils import environment_provenance
from loguru import logger


def _loss_summary(losses: list[float]) -> dict[str, float]:
    arr = np.asarray(losses, dtype=float)
    window = max(1, len(arr) // 3)
    first = float(arr[:window].mean())
    last = float(arr[-window:].mean())
    return {"first": first, "last": last, "min": float(arr.min()), "decreased": last < first}


def run_study(config: PfnStudyConfig) -> dict[str, Any]:
    report = profile(config.arch, config.train, config.mode)
    smoke_train = config.train.model_copy(update={"steps": config.mode.smoke_steps})
    _, result = train_pfn(config.arch, smoke_train, log_every=max(1, config.mode.smoke_steps // 5))
    loss_summary = _loss_summary(result.losses)
    status = "pass" if loss_summary["decreased"] and report["in_regime"] else "check"
    return {"report": report, "losses": result.losses, "loss_summary": loss_summary, "status": status}


def _md_table(rows: list[dict[str, Any]], columns: tuple[tuple[str, str], ...]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"

    def fmt(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.2f}"
        return "" if value is None else str(value)

    body = ["| " + " | ".join(fmt(row.get(key)) for key, _ in columns) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def build_summary_markdown(config: PfnStudyConfig, result: dict[str, Any]) -> str:
    report = result["report"]
    loss = result["loss_summary"]
    realized = report["realized_shapes"]
    realized_rows = [
        {"quantity": "rows", **{q: realized["n_rows"][q] for q in ("0.1", "0.5", "0.9")}},
        {"quantity": "features", **{q: realized["n_features"][q] for q in ("0.1", "0.5", "0.9")}},
    ]
    headline = [
        f"Status: **{result['status']}**.",
        f"Mode: **{config.mode.name}** | device: **{report['device']}** | "
        f"parameters: **{report['n_parameters']:,}** | n_bins: {config.arch.n_bins}.",
        f"Anchor context: **{report['anchor']['rows']} rows x {report['anchor']['features']} features**, "
        f"max_context {report['max_context']} — in regime: **{report['in_regime']}**.",
        f"Smoke train ({config.mode.smoke_steps} steps): loss **{loss['first']:.3f} -> {loss['last']:.3f}** "
        f"(min {loss['min']:.3f}), decreased: **{loss['decreased']}**.",
    ]
    return "\n".join(
        [
            "# PFN Feasibility Study Summary",
            "",
            "## Headline",
            *[f"- {item}" for item in headline],
            "",
            "## Cost Profile",
            "_One context-agnostic model timed at each grid cell; peak memory is best-effort per device._",
            "",
            _md_table(
                report["cells"],
                (
                    ("rows", "Rows"),
                    ("features", "Features"),
                    ("n_train", "Train rows"),
                    ("n_test", "Test rows"),
                    ("train_ms", "Train ms/step"),
                    ("infer_ms", "Infer ms"),
                    ("peak_memory_mb", "Peak MB"),
                ),
            ),
            "",
            "## Realized Training Shapes",
            "_Quantiles of the jittered shapes drawn around the anchor._",
            "",
            _md_table(realized_rows, (("quantity", "Quantity"), ("0.1", "q10"), ("0.5", "q50"), ("0.9", "q90"))),
            "",
        ]
    )


def write_study_artifacts(config: PfnStudyConfig, project_root: Path, *, output: Path | None = None) -> dict[str, Any]:
    destination = output or (project_root / config.mode.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    configure_study_logging(destination, study="pfn")
    result = run_study(config)

    losses = pl.DataFrame({"step": range(len(result["losses"])), "loss": result["losses"]})
    losses.write_parquet(destination / "training.parquet")
    (destination / "feasibility.json").write_text(json.dumps(result["report"], indent=2, sort_keys=True))
    (destination / "loss_summary.json").write_text(json.dumps(result["loss_summary"], indent=2, sort_keys=True))
    (destination / "config.json").write_text(json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True))
    (destination / "summary.md").write_text(build_summary_markdown(config, result))
    (destination / "environment.json").write_text(
        json.dumps(environment_provenance(project_root), indent=2, sort_keys=True)
    )
    logger.success(f"✅ pfn feasibility complete | status={result['status']} | artifacts → {destination}")
    return {"status": result["status"], "steps": len(result["losses"])}
