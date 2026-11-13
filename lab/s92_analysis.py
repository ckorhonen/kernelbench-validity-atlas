#!/usr/bin/env python
"""s92_analysis.py — post-run analysis of s92_frac_boundary.jsonl (declared post-hoc where noted).

The frozen summary returned FRAC_BOUNDARY_PARTIAL because the 0.5 level was never reached: frac_over saturates
below 0.02 across the whole swept range. This script computes, from the committed rows only:
  1. cross-tuple agreement of the frac=0.01 crossing (g, |A|max, u) — the BRANCH_FRAC_COLLAPSE clause the frozen
     code skipped when it took the PARTIAL branch;
  2. the plateau: frac_over at the top of the grid, the fraction of the total rise contributed by the last 4x in g,
     and an asymptotic ceiling estimate (post-hoc, labelled);
  3. the extremum growth exponent (log-log slope of rho vs g) so the divergence/saturation split is quantified;
  4. the gap between the frac=0.01 crossing and s91's rho=1 crossing, per tuple.
No thresholds are introduced beyond the frozen ones.
"""
import json
import os

LAB = os.path.expanduser("~/rstation/kernelbench/lab")
ROWS = os.path.join(LAB, "results", "s92_frac_boundary.jsonl")
OUT = os.path.join(LAB, "results", "s92_analysis.json")
S91_RHO1_G = {"T1": 8.354, "T2": 7.336}

rows = [json.loads(l) for l in open(ROWS)]
ok = [r for r in rows if r.get("status") == "ok"]
out = dict(n_rows=len(rows), n_ok=len(ok))

cross = {}
for tid in ("T1", "T2"):
    cand = [r for r in ok if r["tid"] == tid and r["kind"] == "bisect"
            and r.get("A_1e-2", {}).get("frac_over", 0) >= 0.01]
    cand.sort(key=lambda r: r["g"])
    if not cand:
        continue
    r = cand[0]
    cross[tid] = dict(g=r["g"], maxabs=r["maxabs_out"], rho=r["A_1e-2"]["rho"],
                      frac=r["A_1e-2"]["frac_over"], u=2.2e-4 * r["maxabs_out"] / 1e-2,
                      B_frac=r["B_1e-2"]["frac_over"], B_rho=r["B_1e-2"]["rho"],
                      engine_class=r.get("engine_class"))
out["frac001_crossing"] = cross
if len(cross) == 2:
    gs = [cross[t]["g"] for t in cross]
    ms = [cross[t]["maxabs"] for t in cross]
    us = [cross[t]["u"] for t in cross]
    out["cross_tuple"] = dict(g_ratio=max(gs) / min(gs), maxabs_ratio=max(ms) / min(ms),
                              u_ratio=max(us) / min(us), within_25=bool(max(us) / min(us) <= 1.25))

plateau = {}
for tid in ("T1", "T2"):
    seq = sorted([(r["g"], r["A_1e-2"]["frac_over"], r["A_1e-2"]["rho"]) for r in ok
                  if r["tid"] == tid and r["kind"] == "grid"], key=lambda t: t[0])
    if not seq:
        continue
    f_lo, f_hi = seq[0][1], seq[-1][1]
    last4 = [s for s in seq if s[0] >= 1024.0]
    rise_last = (last4[-1][1] - last4[0][1]) if len(last4) > 1 else 0.0
    plateau[tid] = dict(frac_at_g64=f_lo, frac_at_g4096=f_hi, total_rise=f_hi - f_lo,
                        rise_last_4x=rise_last, share_last_4x=(rise_last / (f_hi - f_lo)) if f_hi > f_lo else None,
                        rho_at_g64=seq[0][2], rho_at_g4096=seq[-1][2],
                        rho_loglog_slope=(__import__("math").log(seq[-1][2] / seq[0][2]) /
                                          __import__("math").log(seq[-1][0] / seq[0][0])))
out["plateau"] = plateau
out["gap_over_s91_rho1"] = {t: round(cross[t]["g"] / S91_RHO1_G[t], 3) for t in cross}
out["posthoc_note"] = ("cross_tuple and plateau are post-hoc computations from committed rows; the frozen branch "
                       "clauses only evaluated the frac=0.01 level because the 0.5 level does not exist in range")
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps(out, indent=1))
