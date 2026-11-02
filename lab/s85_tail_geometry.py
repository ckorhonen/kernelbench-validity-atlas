#!/usr/bin/env python
"""s85_tail_geometry.py — does the admission-relevant tail compress in derived geometry where the median did not?

PREREG (frozen before the first computation; do not edit after).

WHY. s84 measured that the attributable column can carry the reference share at class level but not the
magnitude: inside one engine class the worst-element ratio spans 11.2x (conv n=23) and 19.9x
(conv_transpose n=24), while draw noise is only 1.08x, so the spread is genuine shape dependence. s74/s81
tested geometry against the MEDIAN bias (LOO R2 -0.99 / -0.19) and found no law. The tail statistic was never
tested against geometry. This is a bounded CPU test of that one question on committed artifacts.

POPULATION. The 54 exceed rows of lab/results/s73c_merged_56.jsonl, joined to lab/results/s75_engine_class_s73pop.jsonl
for input/output shapes and parameter count. Target y = log10(max_ratio_ref) of the C1 arm at 1e-4 (declared:
one arm for uniformity; C1 is the arm a compiler-generated candidate would present; 27/54 C1 rows are
bit-identical to the reference, so for those the target is the reference's own deviation under its own output
scale). Rows lacking a parsed shape are excluded and counted, never imputed.

FEATURES (derived coordinates, all from declared shapes): log10(output elements), log10(input elements),
log10(parameter count), log10(spatial extent = product of output dims from axis 2 on), log10(input channels).
All standardised on the training fold only.

METHOD. Within-op-class leave-one-out OLS on log10 target, 5 features; report LOO R2. Also report Spearman
of the target against log10(output elements) as a rank check, and an in-sample R2 for context (labelled).
The gemm class has n=1 and is skipped.

VERDICT (frozen):
  TAIL_GEOMETRY_COMPRESSES  iff LOO R2 >= 0.50 for BOTH conv and conv_transpose.
  TAIL_GEOMETRY_PARTIAL     iff exactly one family clears 0.50.
  TAIL_GEOMETRY_FAILS       otherwise.
KILL (declared): FAILS means the per-shape tail column stands as final for this population and no geometry
surface is claimed for the tail. COMPRESSES/PARTIAL yields a surface label for the clearing family only, and
it is labelled a hypothesis until it replicates on fresh shapes — no coordinate is promoted to a law in the
tick that discovers it.
"""
import json, os, statistics
import numpy as np

LAB = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(LAB, "results")
S73 = os.path.join(RES, "s73c_merged_56.jsonl")
S75 = os.path.join(RES, "s75_engine_class_s73pop.jsonl")
OUT = os.path.join(RES, "s85_tail_geometry.json")
BAR = 0.50


def logl(x):
    return np.log10(max(float(x), 1.0))


rows = [json.loads(l) for l in open(S73) if l.strip()]
meta = {}
for l in open(S75):
    r = json.loads(l)
    meta[(r["level"], r["problem_id"])] = r

data, excluded = [], []
for r in rows:
    if r.get("kind") != "exceed":
        continue
    d = (r.get("decomp_c1_compile") or {}).get("1e-4") or {}
    m = meta.get((r["level"], r["problem_id"]))
    if not m or not m.get("output_shape") or not m.get("input_shapes") or d.get("max_ratio_ref") is None:
        excluded.append({"level": r["level"], "problem_id": r["problem_id"], "reason": "shape/target missing"})
        continue
    out = m["output_shape"]
    inp = m["input_shapes"][0]
    spatial = 1
    for v in out[2:]:
        spatial *= v
    data.append({
        "level": r["level"], "problem_id": r["problem_id"], "op_class": r["op_class"],
        "engine_class": m.get("engine_class_refined"),
        "y": np.log10(max(float(d["max_ratio_ref"]), 1e-12)),
        "features": [logl(np.prod(out)), logl(np.prod(inp)), logl(m.get("n_params") or 1),
                     logl(spatial), logl(inp[1] if len(inp) > 1 else 1)],
    })

FEATS = ["log10_out_elems", "log10_in_elems", "log10_params", "log10_spatial", "log10_in_channels"]


def loo_r2(X, y):
    n = len(y)
    preds = []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        Xt, yt = X[tr], y[tr]
        mu, sd = Xt.mean(0), Xt.std(0)
        sd[sd == 0] = 1.0
        Xs = np.column_stack([np.ones(len(tr)), (Xt - mu) / sd])
        beta, *_ = np.linalg.lstsq(Xs, yt, rcond=None)
        xv = np.concatenate([[1.0], (X[i] - mu) / sd])
        preds.append(float(xv @ beta))
    preds = np.array(preds)
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else None


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else None


per_family = {}
for fam in ("conv", "conv_transpose"):
    sub = [d for d in data if d["op_class"] == fam]
    if len(sub) < 8:
        per_family[fam] = {"n": len(sub), "skipped": True}
        continue
    X = np.array([d["features"] for d in sub], dtype=float)
    y = np.array([d["y"] for d in sub], dtype=float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xs = np.column_stack([np.ones(len(sub)), (X - mu) / sd])
    beta, *_ = np.linalg.lstsq(Xs, y, rcond=None)
    in_r2 = 1.0 - float(((y - Xs @ beta) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())
    per_family[fam] = {
        "n": len(sub),
        "loo_r2": loo_r2(X, y),
        "in_sample_r2": in_r2,
        "spearman_log_out_elems": spearman(X[:, 0], y),
        "spearman_log_spatial": spearman(X[:, 3], y),
        "target_log10_min": float(y.min()), "target_log10_max": float(y.max()),
        "target_log10_median": float(np.median(y)),
        "coef_standardised": {FEATS[i]: float(beta[i + 1]) for i in range(len(FEATS))},
    }

clearing = [f for f, v in per_family.items() if v.get("loo_r2") is not None and v["loo_r2"] >= BAR]
if len(clearing) == 2:
    verdict = "TAIL_GEOMETRY_COMPRESSES"
elif len(clearing) == 1:
    verdict = "TAIL_GEOMETRY_PARTIAL"
else:
    verdict = "TAIL_GEOMETRY_FAILS"

out = {
    "artifact": "s85_tail_geometry",
    "date": "2026-11-02T15:52:00Z",
    "prereg": "docstring of lab/s85_tail_geometry.py, frozen pre-run",
    "population": {"n_used": len(data), "excluded": excluded,
                   "by_op": {f: sum(1 for d in data if d["op_class"] == f) for f in ("conv", "conv_transpose", "gemm")}},
    "features": FEATS,
    "bar_loo_r2": BAR,
    "per_family": per_family,
    "verdict": verdict,
    "clearing_families": clearing,
    "kill_note": ("per-shape tail column stands as final for this population" if verdict == "TAIL_GEOMETRY_FAILS"
                  else "surface label for clearing families only, hypothesis until fresh-shape replication"),
}
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps(out, indent=1)[:4000])
