#!/usr/bin/env python
"""s99_analysis.py — frozen-rule analysis of s99_conditioning.jsonl (E2b designed-conditioning probe).

Rule source: lab/s99_prereg.md (committed before the run). No thresholds changed here.

Verdict branches, computed mechanically:
  K1 instrument repeat must be bit-identical on rho_ref and frac_over, else INSTRUMENT_FAILED.
  K2 engine set identical across levels, else ENGINE_CONFOUNDED.
  K3 construction check <=1e-3 rel, else CONSTRUCTION_FAILED.
  P1 PASS_CONDITIONING_PREDICTOR  |Spearman(kappa, rho_ref)| >= 0.40 AND span(rho_ref) >= 1.3
  P2 RETIRE_H1                    |Spearman| < 0.40 AND span >= 1.3
  P3 NO_LEVERAGE                  span < 1.3
Descriptive (no kill): within-level CV of rho_ref, extremum depth across levels, frac_over columns.
"""
import json
import os
import statistics

LAB = os.path.expanduser("~/kb-lab")
SRC = os.path.join(LAB, "results", "s99_conditioning.jsonl")
INST = os.path.join(LAB, "results", "s99_instrument.json")
SUM = os.path.join(LAB, "results", "s99_conditioning_summary.json")
OUT = os.path.join(LAB, "results", "s99_analysis.json")


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = (sum((v - mx) ** 2 for v in x)) ** 0.5
    sy = (sum((v - my) ** 2 for v in y)) ** 0.5
    if sx == 0 or sy == 0:
        return 0.0
    return sum((x[i] - mx) * (y[i] - my) for i in range(n)) / (sx * sy)


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def main():
    rows = [json.loads(l) for l in open(SRC)]
    summary = json.load(open(SUM))
    inst = json.load(open(INST))
    k1 = bool(inst.get("k1_bit_identical"))
    k2 = bool(summary.get("k2_engine_identical_across_levels"))
    k3 = bool(summary.get("k3_construction_ok"))

    kappa = [r["kappa_measured"] for r in rows]
    rho = [r["rho_ref"] for r in rows]
    span = max(rho) / min(rho) if min(rho) > 0 else None
    rho_kappa = spearman(kappa, rho)

    # descriptive arms
    desc = {}
    by_level = {}
    for r in rows:
        by_level.setdefault(r["level_kappa"], []).append(r)
    for lvl, rs in sorted(by_level.items()):
        vals = [x["rho_ref"] for x in rs]
        desc[str(lvl)] = {
            "n": len(rs),
            "rho_mean": statistics.mean(vals),
            "rho_min": min(vals),
            "rho_max": max(vals),
            "cv": (statistics.pstdev(vals) / statistics.mean(vals)) if statistics.mean(vals) else None,
            "frac_strict_mean": statistics.mean([x["frac_over_strict"] for x in rs]),
            "frac_ship_mean": statistics.mean([x["frac_over_shipped"] for x in rs]),
            "extremum_depth_median": statistics.median([x["extremum_depth"] for x in rs]),
            "extremum_abs_err_median": statistics.median([x["extremum_abs_err"] for x in rs]),
            "sigma_max_b_median": statistics.median([x["sigma_max_b_measured"] for x in rs]),
            "engine": rs[0].get("engine"),
        }
    pooled_cv = (statistics.pstdev(rho) / statistics.mean(rho)) if statistics.mean(rho) else None
    within = [desc[str(l)]["cv"] for l in sorted(by_level) if desc[str(l)]["cv"] is not None]
    desc["within_level_cv_median"] = statistics.median(within) if within else None
    desc["within_level_cv_max"] = max(within) if within else None
    desc["pooled_cv"] = pooled_cv
    desc["spearman_kappa_rho"] = rho_kappa
    desc["spearman_sigma_max_b_rho"] = spearman([r["sigma_max_b_measured"] for r in rows], rho)
    desc["spearman_median_absX_rho"] = spearman([r["median_absX"] for r in rows], rho)
    desc["spearman_depth_rho"] = spearman([r["extremum_depth"] for r in rows], rho)
    desc["rho_min"] = min(rho)
    desc["rho_max"] = max(rho)
    desc["span"] = span

    if not k1:
        verdict = "INSTRUMENT_FAILED"
    elif not k2:
        verdict = "ENGINE_CONFOUNDED"
    elif not k3:
        verdict = "CONSTRUCTION_FAILED"
    elif span is not None and span < 1.3:
        verdict = "P3_NO_LEVERAGE"
    elif abs(rho_kappa) >= 0.40:
        verdict = "P1_PASS_CONDITIONING_PREDICTOR"
    else:
        verdict = "P2_RETIRE_H1"

    out = {
        "script": "s99_analysis",
        "prereg": "lab/s99_prereg.md",
        "n_rows": len(rows),
        "kills": {"K1": k1, "K2": k2, "K3": k3},
        "verdict": verdict,
        "primary": {"spearman_kappa_rho": rho_kappa, "span": span,
                    "bar_abs_spearman": 0.40, "bar_span": 1.3},
        "descriptive": desc,
        "date": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
