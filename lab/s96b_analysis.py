#!/usr/bin/env python
"""s96b_analysis.py — descriptive analysis of the post-hoc ratio-argmax probe (no verdict issued).

Computes the per-draw ratio, the identity residual log rho = log e - log(atol*(1+|X_e|)), the spans of the
absolute error and of the cancellation depth at the ratio-argmax, and the Spearman of rho against e and
against depth. Artifact consumed by the prose audit (s51 C43).
"""
import json
import math
import os
import statistics as st

LAB = os.path.expanduser("~/rstation/kernelbench/lab")
IN = os.path.join(LAB, "results", "s96b_ratio_argmax.jsonl")
OUT = os.path.join(LAB, "results", "s96b_analysis.json")


def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0 + 1.0
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
ok = [r for r in rows if r.get("status") == "ok"]
res = {"n_rows": len(rows), "n_ok": len(ok),
       "resid_max": max(abs(r["log_rho"] - (r["log_e"] - r["log_thr"])) for r in ok)}

for key in ["rho_ref", "e_at_ratio_argmax", "abs_x_at_ratio_argmax", "depth_at_ratio_argmax",
            "e_max", "abs_x_at_e_argmax"]:
    v = [r[key] for r in ok]
    res[key] = {"min": min(v), "max": max(v), "mean": st.mean(v), "cv": cv(v), "spread": max(v) / min(v)}

res["spearman"] = {
    "rho_vs_e_at_ratio_argmax": spearman([r["rho_ref"] for r in ok], [r["e_at_ratio_argmax"] for r in ok]),
    "rho_vs_depth_at_ratio_argmax": spearman([r["rho_ref"] for r in ok], [r["depth_at_ratio_argmax"] for r in ok]),
    "rho_vs_e_max": spearman([r["rho_ref"] for r in ok], [r["e_max"] for r in ok]),
}
res["reading"] = ("POST_HOC_DESCRIPTIVE: the amplitude is carried by the absolute rounding error at the "
                  "worst-cancelled element, not by cancellation depth; no verdict, no new boundary.")
with open(OUT, "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps(res, indent=1)[:1800])
