"""s66b -- relabel the s66 engine observations with s63's classifier and report both labelings.

Post-hoc, declared: s66 shipped a narrower regex set than s63 (it omitted `fprop|fp32|scudnn`), so one
row (w4 3x2x5) was labelled OTHER where s63 calls the same kernel FP32_SIMT. The kernel names are
recorded in results/s66_foreign_warm.jsonl, so the labels are recomputed here under s63's patterns
verbatim. Both labelings print; nothing else in the artifact is touched, and no GPU run is repeated.
"""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "results", "s66_foreign_warm.jsonl")
OUT = os.path.join(HERE, "results", "s66b_relabel.json")

CLASS_PATTERNS = [  # s63_engine_predicate.py, verbatim
    ("WINOGRAD", re.compile(r"winograd", re.I)),
    ("TF32_TENSOROP", re.compile(r"tf32|tensorop|1688|16816|h1688", re.I)),
    ("FP64", re.compile(r"dgrad|dgemm|dconv|fp64|double|_d_", re.I)),
    ("FP32_SIMT", re.compile(r"fprop|implicit_gemm|sgemm|s1688|fp32|scudnn|simt", re.I)),
    ("CUTLASS", re.compile(r"cutlass", re.I)),
]


def classify(names):
    for lab, pat in CLASS_PATTERNS:
        for n in names:
            if pat.search(n):
                return lab
    return "OTHER"


rows = [json.loads(l) for l in open(SRC)]
out = {"date": rows[0]["date"], "n_rows": len(rows), "source": "results/s66_foreign_warm.jsonl",
       "relabel_rule": "s63_engine_predicate.CLASS_PATTERNS verbatim, applied to the recorded kernel names",
       "label_changes": [], "engine_by_context": {}, "context_stability": {"stable": 0, "unstable": 0},
       "rho_max_by_context": {}, "rows": []}
for r in rows:
    labels = {}
    for ctx, rec in r["contexts"].items():
        if "kernels" not in rec:
            continue
        lab = classify([n for n, _ in rec["kernels"]])
        labels[ctx] = lab
        out["engine_by_context"].setdefault(ctx, {}).setdefault(lab, 0)
        out["engine_by_context"][ctx][lab] += 1
        old = rec.get("engine_class")
        if old and old != lab:
            out["label_changes"].append({"row": [r["H"], r["N"], r["C"]], "context": ctx,
                                         "s66_label": old, "relabelled": lab})
        if ctx != "A" and "rho_vs_A" in rec:
            out["rho_max_by_context"][ctx] = max(out["rho_max_by_context"].get(ctx, 0.0), rec["rho_vs_A"])
    core = [labels.get(c) for c in ("A", "B", "C", "D", "E")]
    stable = len(set(core)) == 1
    out["context_stability"]["stable" if stable else "unstable"] += 1
    out["rows"].append({"row": [r["H"], r["N"], r["C"]], "delivered_change": r["armA_delivered_change"],
                        "classes": labels, "context_stable": stable})
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps({k: out[k] for k in ("n_rows", "label_changes", "engine_by_context",
                                     "context_stability", "rho_max_by_context")}, indent=1))
