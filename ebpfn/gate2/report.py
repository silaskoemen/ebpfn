"""Decisive Gate-2 verdict (plans/gate2.md §4).

Combines the two pre-committed parts into one decision and one human-readable
verdict string. The whole point of Gate-2 is to come out *decisive* either way:
a variance FAIL kills coverage-gating (go direct); an ablation PASS promotes the
conditional-structure descriptor as the EB coverage surrogate; anything else is a
clean null. No tuning happens after this -- thresholds were frozen in config.
"""
from __future__ import annotations

# verdict codes
DEAD = "COVERAGE_GATING_DEAD"          # Part A failed: coverage can't discriminate real tasks
INCONCLUSIVE = "INCONCLUSIVE_PRIORS"   # priors too similar -> ablation has no coverage range
INCONCLUSIVE_POWER = "INCONCLUSIVE_POWER"  # ablation CI too wide / too few tasks to decide either way
LINK = "DESCRIPTOR_COVERAGE_PREDICTS_CALIBRATION"  # Part B passed (link shown)
NO_LINK = "NO_COVERAGE_CALIBRATION_LINK"           # Part B null: a usable effect is ruled OUT


def gate2_verdict(variance: dict, ablation: dict) -> dict:
    """One dict: the two parts, the verdict code, and a one-line explanation."""
    if not variance["passes"]:
        code = DEAD
        why = (f"Part A FAIL: only {variance['frac_outside']:.0%} of real tasks fall outside the "
               f"prior's descriptor null band (need {variance['min_frac_outside']:.0%}); "
               f"median distance ratio {variance['median_ratio']:.2f} (need {variance['min_median_ratio']:.2f}). "
               f"Descriptor coverage does not separate real tasks from the prior -- the Gate-1 failure "
               f"mode recurs one level up. Coverage-gating is dead; go direct.")
    elif not ablation["has_coverage_variation"]:
        code = INCONCLUSIVE
        why = (f"Part A passed, but the prior ladder produces near-zero within-task coverage spread "
               f"({ablation['coverage_spread']:.2e}); the ablation has no gradient to test. Widen the "
               f"prior ladder before re-running.")
    elif not ablation["enough_tasks"]:
        code = INCONCLUSIVE_POWER
        why = (f"Part A passed and coverage varies, but only {ablation['n_tasks']} tasks are shared "
               f"across all priors (need >= {ablation['min_ablation_tasks']}); the ablation is "
               f"underpowered to decide either way. Run more tasks before reading Part B.")
    elif ablation["passes"]:
        code = LINK
        why = (f"Part B PASS: across-prior fixed-effects correlation {ablation['fe_corr']:+.3f} "
               f"(95% CI [{ablation['ci_lo']:+.3f}, {ablation['ci_hi']:+.3f}], threshold "
               f"{ablation['effect_threshold']}). The CI lower bound clears the bar: tasks worse-covered "
               f"by a prior are worse-calibrated under that prior's PFN. Conditional-structure coverage "
               f"predicts calibration -- promote it as the EB surrogate.")
    elif ablation["ruled_out_effect"]:
        code = NO_LINK
        why = (f"Part A passed (coverage discriminates), but Part B is a true null: the fixed-effects "
               f"correlation CI [{ablation['ci_lo']:+.3f}, {ablation['ci_hi']:+.3f}] lies entirely "
               f"BELOW the {ablation['effect_threshold']} bar -- a usable positive effect is ruled out. "
               f"Coverage varies but does not drive calibration; do not build the nudging loop on it.")
    else:
        code = INCONCLUSIVE_POWER
        why = (f"Part A passed and coverage varies, but the Part B CI [{ablation['ci_lo']:+.3f}, "
               f"{ablation['ci_hi']:+.3f}] (half-width {ablation['ci_half_width']:.3f}) straddles the "
               f"{ablation['effect_threshold']} bar -- consistent with both a real link and none. This "
               f"is underpowered, NOT a null: add ladder rungs and/or tasks, do not conclude from it.")
    return {"verdict": code, "explanation": why, "variance": variance, "ablation": ablation}


def format_report(result: dict) -> str:
    """Pretty console block for the driver."""
    v, a = result["variance"], result["ablation"]
    lines = [
        "=" * 78,
        "GATE-2  -- conditional-structure coverage -> calibration",
        "=" * 78,
        f"  tasks (all priors): {a['n_tasks']}   priors: {', '.join(a['priors'])}",
        "",
        "  PART A  variance go/no-go (does coverage discriminate? -- checked pre-calibration)",
        f"    frac real tasks outside null band : {v['frac_outside']:.0%}  (need >= {v['min_frac_outside']:.0%})",
        f"    median(real)/median(null) ratio   : {v['median_ratio']:.2f}  (need >= {v['min_median_ratio']:.2f})",
        f"    -> {'PASS' if v['passes'] else 'FAIL'}",
        "",
        "  PART B  across-prior fixed-effects ablation (PRIMARY)",
        f"    tasks shared across all priors    : {a['n_tasks']}  (need >= {a['min_ablation_tasks']})",
        f"    within-task coverage spread       : {a['coverage_spread']:.3f}",
        f"    fixed-effects correlation         : {a['fe_corr']:+.3f}  95% CI [{a['ci_lo']:+.3f}, {a['ci_hi']:+.3f}]  (half-width {a['ci_half_width']:.3f})",
        f"    threshold                         : {a['effect_threshold']}",
        f"    -> {'LINK' if a['passes'] else 'NULL (effect ruled out)' if a['ruled_out_effect'] else 'INCONCLUSIVE (underpowered)'}",
        "",
        "  SECONDARY  per-prior cross-sectional partial Spearman (vs Gate-1 0.083)",
    ]
    for p, val in a["cross_sectional_partial_spearman"].items():
        lines.append(f"    {p:16s} : {val:+.3f}")
    lines += [
        "",
        f"  VERDICT: {result['verdict']}",
        f"  {result['explanation']}",
        "=" * 78,
    ]
    return "\n".join(lines)
