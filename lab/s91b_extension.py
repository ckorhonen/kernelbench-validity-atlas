#!/usr/bin/env python
"""s91b_extension.py — downward grid extension for the crossing the frozen s91 grid missed.

AMENDMENT v2 to s91 (declared post-hoc, BEFORE this run, and labelled as such). The frozen s91 grid floor
(g=0.0625) sits ABOVE T2's 1e-4 crossing: T2 at g=0.0625 already shows rho_1e-4 = 1.0441, so the frozen rule
correctly reported NO_CROSSING_IN_RANGE. That is a prereg GRID-FLOOR DEFECT, not an outcome-driven choice: the
grid was laid out around s90's norm point without checking that the strict-gate crossing of the larger tuple
falls inside it. This extension adds three downward points for T2 (g in {0.015625, 0.03125, 0.05}) and bisects
the T2 1e-4 crossing geometrically (4 steps, 2% stop), using the identical run_point code, seed, predicate and
tolerance from s91. No threshold, branch rule or kill clause changes; the added rows are tagged kind="ext" and
carry this amendment's sha so they cannot be confused with frozen-grid rows.
"""
import importlib.util, json, os, hashlib, time

spec = importlib.util.spec_from_file_location("s91", os.path.expanduser("~/kb-lab/s91_scale_boundary.py"))
s91 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s91)

OUT = os.path.join(s91.LAB, "results", "s91b_extension.jsonl")
SUMOUT = os.path.join(s91.LAB, "results", "s91b_extension_summary.json")
open(OUT, "w").close()
sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
print("amendment sha", sha[:12], flush=True)

td = [t for t in s91.TUPLE_DEFS if t["tid"] == "T2"][0]
rows = []
for g in (0.015625, 0.03125, 0.05):
    r = s91.run_point(td, g, "ext")
    rows.append(r)
    with open(OUT, "a") as f:
        f.write(json.dumps(r) + "\n")
    print("ext g", g, r["status"], "maxabs", round(r.get("maxabs_out", 0), 4),
          "rho1e4", round(s91.rho_of(r, "1e-4") or 0, 4), "rho1e2", round(s91.rho_of(r, "1e-2") or 0, 4),
          "rep", r.get("a_repeat_bit_identical"), flush=True)

seq = [(r["g"], s91.rho_of(r, "1e-4")) for r in rows if r["status"] == "ok"]
br = None
for i in range(len(seq) - 1):
    if seq[i][1] < 1.0 <= seq[i + 1][1]:
        br = (seq[i][0], seq[i + 1][0])
if br is None:
    print("no bracket found in extension", seq, flush=True)
    json.dump(dict(amendment_sha=sha, bracket=None, seq=seq), open(SUMOUT, "w"), indent=1)
else:
    lo, hi = br
    for step in range(4):
        mid = (lo * hi) ** 0.5
        r = s91.run_point(td, mid, "ext-bisect")
        rows.append(r)
        with open(OUT, "a") as f:
            f.write(json.dumps(r) + "\n")
        rho = s91.rho_of(r, "1e-4")
        print("  bisect g", round(mid, 6), r["status"], "rho", rho, flush=True)
        if r["status"] != "ok":
            break
        if rho >= 1.0:
            hi = mid
        else:
            lo = mid
        if hi / lo < 1.02:
            break
    pts = [r for r in rows if r["kind"] == "ext-bisect" and r["status"] == "ok"]
    inside = [p for p in pts if lo <= p["g"] <= hi]
    near = min(inside, key=lambda p: abs(p["g"] - (lo * hi) ** 0.5)) if inside else None
    summary = dict(amendment_sha=sha, tid="T2", gate="1e-4", bracket=[lo, hi], seq=seq)
    if near:
        summary.update(g_measured=near["g"], maxabs_out=near["maxabs_out"], rho=s91.rho_of(near, "1e-4"),
                       u=s91.REF_REL * near["maxabs_out"] / s91.STRICT,
                       frac_over=near["A_1e-4"]["frac_over"], argmax=near["A_1e-4"]["argmax"],
                       B_rho=near.get("B_1e-4", {}).get("rho"), engine_class=near.get("engine_class"))
    json.dump(summary, open(SUMOUT, "w"), indent=1)
    print(json.dumps(summary, indent=1)[:2000], flush=True)
