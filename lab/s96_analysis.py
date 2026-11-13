#!/usr/bin/env python
"""s96_analysis.py — predeclared analysis of s96_amplitude_mechanism.jsonl (32 draws + instrument repeat).

Fires the coded kill/branch clauses verbatim and reports the Spearman arms, the CV ladder across tail
quantiles, and the engine-identity check. No new data, no post-hoc promotion.
"""
import json
import math
import os
import statistics as st

LAB = os.path.expanduser("~/rstation/kernelbench/lab")
IN = os.path.join(LAB, "results", "s96_amplitude_mechanism.jsonl")
OUT = os.path.join(LAB, "results", "s96_analysis.json")


def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = rank(x), rank(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def cv(v):
    return st.pstdev(v) / abs(st.mean(v)) if st.mean(v) else float("nan")


rows = [json.loads(l) for l in open(IN)]
main = rows[:-1]  # the last appended row is the instrument repeat of draw 0
print("rows", len(rows), "main", len(main), "repeat_present", len(rows) - len(main))

res = {"n_main": len(main), "rows_total": len(rows)}

# --- kill clauses
k1 = (rows[-1]["rho_ref"] == main[0]["rho_ref"]) and (rows[-1]["frac_over_ref_strict"] == main[0]["frac_over_ref_strict"])
k2 = any(r["rho_ref"] == 0 for r in main)
sig = [r["sigma_max_a"] for r in main]
k3 = cv(sig) < 1e-6
res["kills"] = {"K1_instrument_ok": bool(k1), "K2_vacuous_fired": bool(k2), "K3_degenerate_fired": bool(k3)}
res["instrument"] = {"repeat_rho": rows[-1]["rho_ref"], "base_rho": main[0]["rho_ref"],
                     "repeat_frac": rows[-1]["frac_over_ref_strict"], "base_frac": main[0]["frac_over_ref_strict"]}

rho = [r["rho_ref"] for r in main]
res["rho_ref"] = {"min": min(rho), "max": max(rho), "mean": st.mean(rho), "sd": st.pstdev(rho),
                  "cv": cv(rho), "spread_ratio": max(rho) / min(rho), "median": st.median(rho)}

# --- covariates
cov = {}
for key in ["sigma_max_a", "kappa_proxy", "sigma_max_b", "median_abs_x", "max_abs_x",
            "extremum_abs_err", "extremum_abs_x", "extremum_rel_to_median", "max_abs_ref_err"]:
    v = [r[key] for r in main]
    cov[key] = {"min": min(v), "max": max(v), "cv": cv(v), "spearman_vs_rho": spearman(v, rho)}
res["covariates"] = cov

# --- tail CV ladder
ladder = {}
for q in ["p99", "p99.9", "p99.99", "p99.999"]:
    v = [r["q"][q] for r in main if r["q"].get(q)]
    ladder[q] = {"n": len(v), "mean": st.mean(v), "min": min(v), "max": max(v), "cv": cv(v),
                 "cv_ratio_to_rho_cv": cv(v) / cv(rho)}
res["tail_ladder"] = ladder

# --- engine identity
sets = [tuple(r.get("engine_kernels") or []) for r in main if r.get("engine_kernels")]
res["engine"] = {"profiled": len(sets), "distinct_sets": len(set(sets)),
                 "sets": [list(s) for s in sorted(set(sets))]}

# --- verdicts, coded verbatim
if not k1 or k2:
    res["verdict"] = "INSTRUMENT_FAILED"
elif k3:
    res["verdict"] = "SPECTRAL_UNTESTABLE"
    res["branch"] = "AMPLITUDE_SCALE_CARRIED" if (abs(cov["median_abs_x"]["spearman_vs_rho"]) >= 0.4 or
                                                  abs(cov["max_abs_x"]["spearman_vs_rho"]) >= 0.4) else "AMPLITUDE_LOTTERY"
else:
    s_sig = cov["sigma_max_a"]["spearman_vs_rho"]
    s_kap = cov["kappa_proxy"]["spearman_vs_rho"]
    if abs(s_sig) >= 0.4 or abs(s_kap) >= 0.4:
        res["verdict"] = "AMPLITUDE_PREDICTABLE_SPECTRAL"
    elif abs(cov["median_abs_x"]["spearman_vs_rho"]) >= 0.4 or abs(cov["max_abs_x"]["spearman_vs_rho"]) >= 0.4:
        res["verdict"] = "AMPLITUDE_SCALE_CARRIED"
    else:
        res["verdict"] = "AMPLITUDE_LOTTERY"

c99 = ladder["p99.99"]["cv"]
if c99 < 0.03 and cv(rho) > 0.10:
    res["tail_clause"] = "TAIL_SAMPLING_SUPPORTED"
elif c99 > 0.10:
    res["tail_clause"] = "TAIL_MOVES"
else:
    res["tail_clause"] = "TAIL_INDETERMINATE"

res["engine_clause"] = "ENGINE_VARIES" if res["engine"]["distinct_sets"] > 1 else "ENGINE_STABLE"

with open(OUT, "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps({k: res[k] for k in ["kills", "rho_ref", "tail_ladder", "engine", "verdict", "tail_clause", "engine_clause"]}, indent=1))
print("--- spearman arms")
for k, v in cov.items():
    print("%-24s cv=%.4g rho_s=%.3f" % (k, v["cv"], v["spearman_vs_rho"]))
