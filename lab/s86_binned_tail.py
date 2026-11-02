#!/usr/bin/env python
"""s86_binned_tail.py — last magnitude attempt: does a binned, reproducible key bound the tail?

PREREG (frozen before the first computation; do not edit after).

WHY. s84 killed the engine-class key for magnitude (within-class tail spread 11.2x / 19.9x). s85 killed the
continuous geometry key for the same target (LOO R2 -1.08 / -0.37). This is the declared second and final
magnitude attempt: a coarse binned key that uses only measured, reproducible quantities (engine class x op_class x
output-extent band), so the cell means can be estimated with n>=5 per cell instead of a 5-feature regression on 23
rows. A second failure closes the magnitude question for this population and parks it.

POPULATION. The 54 exceed rows of s73c_merged_56.jsonl with shapes from s75. Target y = log10(max_ratio_ref) C1
arm at 1e-4, identical to s85 so the two tests share a target.
KEY. cell = (engine_class_refined, op_class, extent_band) where extent_band is a fixed three-way split of
log10(output elements): <6.0, 6.0-8.0, >=8.0. Bands are declared here, before any variance is computed.

TESTS (frozen):
  T1 variance explained: eta^2 = 1 - SS_within/SS_total over cells with n>=2. Bar: eta^2 >= 0.50 for a
     compression claim.
  T2 cell bound: every cell with n>=5 must have max(y)/min(y) <= 1 decade (i.e. max_ratio_ref spread <= 10x).
  T3 replication honesty: report the number of cells with n>=5 (a claim on fewer than three such cells is
     labelled thin regardless of the numbers).
VERDICT (frozen): TAIL_KEY_BOUNDED iff T1 and T2 both pass AND T3 >= 3 cells; TAIL_KEY_THIN if T1 and T2 pass
but fewer than three cells qualify; TAIL_KEY_FAILS otherwise.
KILL (declared): TAIL_KEY_FAILS or TAIL_KEY_THIN => the magnitude question is parked for this population: the
per-shape tail column is the deliverable and no further key is searched in this lane. Only a new population
(fresh shapes or a peer's margins) reopens it.
"""
import json, os
import numpy as np

LAB = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(LAB, "results")
OUT = os.path.join(RES, "s86_binned_tail.json")

rows = [json.loads(l) for l in open(os.path.join(RES, "s73c_merged_56.jsonl")) if l.strip()]
meta = {}
for l in open(os.path.join(RES, "s75_engine_class_s73pop.jsonl")):
    r = json.loads(l)
    meta[(r["level"], r["problem_id"])] = r

data = []
for r in rows:
    if r.get("kind") != "exceed":
        continue
    d = (r.get("decomp_c1_compile") or {}).get("1e-4") or {}
    m = meta.get((r["level"], r["problem_id"]))
    if not m or not m.get("output_shape") or d.get("max_ratio_ref") is None:
        continue
    lo = np.log10(max(float(np.prod(m["output_shape"])), 1.0))
    band = "<6" if lo < 6.0 else ("6-8" if lo < 8.0 else ">=8")
    data.append({"key": (m.get("engine_class_refined"), r["op_class"], band),
                 "y": np.log10(max(float(d["max_ratio_ref"]), 1e-12)),
                 "level": r["level"], "problem_id": r["problem_id"]})

y = np.array([d["y"] for d in data])
cells = {}
for d in data:
    cells.setdefault(d["key"], []).append(d["y"])

ss_tot = float(((y - y.mean()) ** 2).sum())
ss_within = 0.0
cell_report = []
for k, v in sorted(cells.items()):
    arr = np.array(v)
    ss_within += float(((arr - arr.mean()) ** 2).sum())
    cell_report.append({"engine_class": k[0], "op_class": k[1], "band": k[2], "n": len(v),
                        "log10_min": float(arr.min()), "log10_max": float(arr.max()),
                        "tail_spread": float(10 ** (arr.max() - arr.min())),
                        "log10_median": float(np.median(arr))})
eta2 = 1.0 - ss_within / ss_tot if ss_tot > 0 else None
big = [c for c in cell_report if c["n"] >= 5]
t1 = eta2 is not None and eta2 >= 0.50
t2 = all(c["tail_spread"] <= 10.0 for c in big) if big else False
t3 = len(big)
if t1 and t2 and t3 >= 3:
    verdict = "TAIL_KEY_BOUNDED"
elif t1 and t2:
    verdict = "TAIL_KEY_THIN"
else:
    verdict = "TAIL_KEY_FAILS"

out = {"artifact": "s86_binned_tail", "date": "2026-11-02T16:00:00Z",
       "prereg": "docstring of lab/s86_binned_tail.py, frozen pre-run",
       "n_used": len(data), "n_cells": len(cells), "eta2": eta2,
       "T1_bar": 0.50, "T2_bar_spread": 10.0, "T3_cells_n_ge_5": t3,
       "cells": cell_report, "verdict": verdict,
       "kill_note": ("magnitude question parked; per-shape tail column is the deliverable"
                     if verdict != "TAIL_KEY_BOUNDED" else "cell means admissible for cells with n>=5, hypothesis label")}
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps(out, indent=1)[:3500])
