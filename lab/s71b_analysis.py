#!/usr/bin/env python
"""s71b_analysis.py — post-run analysis of the s71 full-population audit. No new measurement: labels computed
from recorded rows only. Op-class labels are recomputed here under a corrected keyword order; s71's producer
labelled "1_Square_matrix_multiplication_" as elementwise because "mul" matched before "multiplication"
(defect recorded, descriptive labels only, no measurement affected).
"""
import json, math, os, sys

RES = os.path.expanduser("~/kb-lab/results")
rows = [json.loads(l) for l in open(os.path.join(RES, "s71_full_audit.jsonl"))]
S = json.load(open(os.path.join(RES, "s71_full_audit_summary.json")))["summary"]


def op_class(name):
    n = name.lower()
    if "convtranspose" in n or "conv_transpose" in n:
        return "conv_transpose"
    if "conv" in n:
        return "conv"
    if any(k in n for k in ("matmul", "gemm", "bmm", "multiplication", "matrix_multiplication")):
        return "gemm"
    if any(k in n for k in ("batchnorm", "layernorm", "groupnorm", "instancenorm", "rmsnorm")):
        return "norm"
    if "softmax" in n:
        return "softmax"
    if "pool" in n:
        return "pool"
    if any(k in n for k in ("lstm", "gru", "rnn")):
        return "rnn"
    if "attention" in n:
        return "attention"
    if any(k in n for k in ("sum", "mean", "_max", "max_", "min", "prod", "cumsum", "norm")):
        return "reduction"
    if any(k in n for k in ("relu", "gelu", "sigmoid", "tanh", "add", "mul", "sub", "div", "scale", "clamp")):
        return "elementwise"
    return "other"


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - h) / d, 4), round((c + h) / d, 4)]


for r in rows:
    r["op_class"] = op_class(r["name"])

adm = [r for r in rows if r.get("anchor") == "admissible"]
ran = [r for r in rows if r.get("status") is None and "A_vs_B" in r]
out = {"population": {"attempted": len(rows), "adm": len(adm), "ran": len(ran),
                      "admissibility": round(len(adm) / max(1, len(rows)), 4),
                      "excluded": S["excluded"]},
       "op_class_counts": {}, "exceedance": {}, "exceeding_tasks": [], "near_tasks": []}

for c in sorted({r["op_class"] for r in rows}):
    pop = [r for r in adm if r["op_class"] == c]
    k = sum(1 for r in pop if r.get("exceed_bench_default"))
    out["op_class_counts"][c] = {"adm": len(pop), "exceed_bench": k, "ci": wilson(k, len(pop)),
                                 "exceed_strict": sum(1 for r in pop if r.get("exceed_strict_default"))}

for key, pop in (("bench_default", adm), ("bench_tf32off", adm), ("strict_default", adm), ("strict_tf32off", adm)):
    k = sum(1 for r in pop if r.get(key))
    out["exceedance"][key] = {"count": k, "n": len(pop), "rate": round(k / max(1, len(pop)), 4), "ci": wilson(k, len(pop))}

ab = [r for r in rows if "A_vs_B" in r]
out["exceedance"]["ab_bench"] = {"count": sum(1 for r in ab if r["A_vs_B"]["ew_1e-2"]["count"] > 0), "n": len(ab)}
out["exceedance"]["ab_strict"] = {"count": sum(1 for r in ab if r["A_vs_B"]["ew_1e-4"]["count"] > 0), "n": len(ab)}

for r in adm:
    a = r["A_vs_anchor"]["ew_1e-2"]
    if a["count"] > 0:
        out["exceeding_tasks"].append({"level": r["level"], "id": r["problem_id"], "name": r["name"],
                                       "op": r["op_class"], "ew_frac": a["frac"], "ew_count": a["count"],
                                       "rho_global": a["rho_global"], "strict_count": r["A_vs_anchor"]["ew_1e-4"]["count"],
                                       "tf32off_bench": r["B_vs_anchor"]["ew_1e-2"]["count"]})
    elif a["frac"] > 0:
        out["near_tasks"].append({"level": r["level"], "id": r["problem_id"], "name": r["name"],
                                  "op": r["op_class"], "ew_frac": a["frac"], "rho_global": a["rho_global"]})

out["max_ew_frac_bench"] = max((r["A_vs_anchor"]["ew_1e-2"]["frac"] for r in adm), default=None)
out["median_ew_frac_bench"] = sorted(r["A_vs_anchor"]["ew_1e-2"]["frac"] for r in adm)[len(adm) // 2] if adm else None
out["zero_diff_tasks"] = sum(1 for r in adm if r["A_vs_anchor"]["ew_1e-2"]["maxabs"] == 0.0)
out["max_rho_global_bench_default"] = S["max_rho_global_bench_default"]
out["max_rho_global_bench_tf32off"] = S["max_rho_global_bench_tf32off"]

# prereg decision clauses, evaluated mechanically
p1 = out["exceedance"]["bench_default"]["rate"]
out["decisions"] = {
    "K1_anchor_admissibility_below_50pct": bool(out["population"]["admissibility"] < 0.5),
    "reconsider_ge_5pct": bool(p1 >= 0.05),
    "reconsider_lt_1pct": bool(p1 < 0.01),
    "verdict": ("MATERIAL: reference error exceeds benchmark tolerance on >=5% of admissible tasks"
                if p1 >= 0.05 else
                ("NEGATIVE: reference inside benchmark tolerance on >99% of admissible tasks" if p1 < 0.01 else
                 "BOUNDED: between 1% and 5%")),
}
json.dump(out, open(os.path.join(RES, "s71b_analysis.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("excluded",)}, indent=1)[:6000])
