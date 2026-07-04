"""Gate-1 primary figure (plans/gate1_revised.md §3): the per-task scatter of
prior-coverage distance vs calibration error, annotated with the n,d-partial
correlation and its bootstrap CI."""
from __future__ import annotations

import numpy as np


def make_gate_figure(coverage_rows: list[dict], calib_rows: list[dict], result: dict):
    import matplotlib.pyplot as plt

    cal_by_key = {(r["source_did"], r["target"]): r for r in calib_rows}
    cov, calib, d = [], [], []
    metric = result["metric"]
    for r in coverage_rows:
        key = (r["source_did"], r["target"])
        if key in cal_by_key:
            cov.append(r["coverage"]); calib.append(cal_by_key[key][metric]); d.append(r["d"])
    cov, calib, d = np.array(cov), np.array(calib), np.array(d)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    sc = ax.scatter(cov, calib, c=d, cmap="viridis", s=42, edgecolor="k", linewidth=0.4)
    fig.colorbar(sc, ax=ax, label="d (features)")
    ax.set_xlabel("prior coverage distance (k-NN-mean s-OTDD)")
    ax.set_ylabel(f"calibration error ({metric})")
    pass_str = "PASS" if result["passes"] else ("CI excl. 0" if result["ci_excludes_zero"] else "no")
    ax.set_title(
        f"Gate-1: partial ρ(n,d)={result['partial_spearman']:.3f} "
        f"[{result['ci_lo']:.3f}, {result['ci_hi']:.3f}]  "
        f"(raw {result['spearman_raw']:.3f})  → {pass_str}\n"
        f"{result['n_tasks']} tasks  thr={result['threshold']}",
        fontsize=10,
    )
    fig.tight_layout()
    return fig
