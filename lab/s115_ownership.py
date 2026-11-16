#!/usr/bin/env python3
"""s115 — ownership of conv gate failures by arm precision. CPU only, committed rows.
Frozen rule verbatim from lab/s115_prereg.md. Writes lab/results/s115_ownership.json."""
import json, statistics, collections, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS = os.path.join(HERE, "results", "r1_conv_gated.jsonl")
OUT = os.path.join(HERE, "results", "s115_ownership.json")

rows = [json.loads(l) for l in open(ROWS)]
checks = []
for r in rows:
    for vn, v in r["variants"].items():
        checks.append({
            "task": f"{r['tag']}_N{r['N']}_H{r['H']}_W{r['W']}_C{r['C']}_CO{r['CO']}",
            "variant": vn,
            "precision": v.get("precision"),
            "hand": not vn.startswith("torch"),
            "pass_1e4": bool(v.get("pass_1e4")),
            "max_abs_err": v.get("max_abs_err"),
            "err_vs_fp32ref": v.get("err_vs_fp32ref"),
        })

hand = [c for c in checks if c["hand"]]
fail = [c for c in hand if not c["pass_1e4"]]

# K1 / K2 tie-outs
k1 = {
    "hand_pass": sum(c["pass_1e4"] for c in hand), "hand_total": len(hand),
    "ieee_pass": sum(c["pass_1e4"] for c in hand if c["precision"] == "ieee"),
    "ieee_total": sum(1 for c in hand if c["precision"] == "ieee"),
    "tf32_pass": sum(c["pass_1e4"] for c in hand if c["precision"] == "tf32"),
    "tf32_total": sum(1 for c in hand if c["precision"] == "tf32"),
}
k2 = {"arm_checks": len(checks), "hand": len(hand), "hand_fail": len(fail)}
k1_ok = (k1["hand_pass"] == 20 and k1["hand_total"] == 560 and k1["tf32_pass"] == 0
         and k1["tf32_total"] == 280 and k1["ieee_pass"] == 20 and k1["ieee_total"] == 280)
k2_ok = (k2["arm_checks"] == 672 and k2["hand"] == 560 and k2["hand_fail"] == 540)

# frozen statistic
qrows = []
for c in fail:
    if c["err_vs_fp32ref"] is None or not c["max_abs_err"]:
        continue
    q = c["err_vs_fp32ref"] / c["max_abs_err"]
    cls = "REFERENCE_OWNED" if q < 0.5 else ("CANDIDATE_OWNED" if q > 2.0 else "SHARED")
    qrows.append({**c, "q": q, "class": cls})

def med(xs):
    return statistics.median(xs) if xs else None

by_prec = {}
for p in ("ieee", "tf32"):
    qs = [x["q"] for x in qrows if x["precision"] == p]
    cls = collections.Counter(x["class"] for x in qrows if x["precision"] == p)
    by_prec[p] = {"n": len(qs), "med_q": med(qs), "q25": sorted(qs)[len(qs)//4] if qs else None,
                  "q75": sorted(qs)[3*len(qs)//4] if qs else None,
                  "classes": dict(cls)}

med_ieee = by_prec["ieee"]["med_q"]; med_tf32 = by_prec["tf32"]["med_q"]
if k1_ok and k2_ok:
    if med_ieee is not None and med_tf32 is not None and med_ieee < 0.5 and med_tf32 > 1.0:
        branch = "BRANCH_OWNERSHIP_ARM_KEYED"
    elif (med_ieee is not None and med_ieee < 0.5) or (med_tf32 is not None and med_tf32 > 1.0):
        branch = "BRANCH_OWNERSHIP_MIXED"
    else:
        branch = "BRANCH_OWNERSHIP_ARM_INVARIANT"
else:
    branch = "INVALID"

# descriptive extras (not the frozen branch)
all_cls = collections.Counter(x["class"] for x in qrows)
ref_owned_frac = all_cls.get("REFERENCE_OWNED", 0) / len(qrows) if qrows else None
# rows where at least one failing ieee arm is reference-owned
rows_any = sorted({x["task"] for x in qrows if x["class"] == "REFERENCE_OWNED"})
res = {
    "date": "2026-11-16", "artifact": "r1_conv_gated.jsonl", "n_rows": len(rows),
    "K1": k1, "K1_ok": k1_ok, "K2": k2, "K2_ok": k2_ok,
    "n_q_rows": len(qrows), "by_precision": by_prec, "class_counts": dict(all_cls),
    "reference_owned_frac": ref_owned_frac, "branch": branch,
    "rows_with_reference_owned_failure": len(rows_any),
    "non_claims": ["one draw per row", "conv rows only", "declared precision not named engine",
                   "q is a ratio of maxima"],
}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
