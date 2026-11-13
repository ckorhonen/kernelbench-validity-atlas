#!/usr/bin/env python
"""s91c_t2_cross.py — bisect T2's strict-gate crossing inside the bracket the frozen grid straddles.

The frozen s91 grid has T2 at g=0.05 -> rho_1e-4 0.8056 and g=0.0625 -> 1.0441; s91b added 0.015625/0.03125/0.05.
Bracket = [0.05, 0.0625]. Identical run_point code, seed, predicate, tolerance. Rows tagged kind="ext-bisect".
No threshold or branch change; this only supplies the measured g* the frozen grid floor could not bracket.
"""
import importlib.util, json, os, hashlib

spec = importlib.util.spec_from_file_location("s91", os.path.expanduser("~/kb-lab/s91_scale_boundary.py"))
s91 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s91)

OUT = os.path.join(s91.LAB, "results", "s91b_extension.jsonl")
SUMOUT = os.path.join(s91.LAB, "results", "s91c_t2_cross_summary.json")
sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
print("script sha", sha[:12], flush=True)
td = [t for t in s91.TUPLE_DEFS if t["tid"] == "T2"][0]

lo, hi = 0.05, 0.0625
rows = []
for step in range(4):
    mid = (lo * hi) ** 0.5
    r = s91.run_point(td, mid, "ext-bisect")
    rows.append(r)
    with open(OUT, "a") as f:
        f.write(json.dumps(r) + "\n")
    rho = s91.rho_of(r, "1e-4")
    print("g", round(mid, 6), r["status"], "maxabs", round(r.get("maxabs_out", 0), 4), "rho", rho,
          "rep", r.get("a_repeat_bit_identical"), flush=True)
    if r["status"] != "ok":
        break
    if rho >= 1.0:
        hi = mid
    else:
        lo = mid
    if hi / lo < 1.02:
        break

pts = [r for r in rows if r["status"] == "ok"]
near = min(pts, key=lambda p: abs(p["g"] - (lo * hi) ** 0.5)) if pts else None
summary = dict(script_sha=sha, tid="T2", gate="1e-4", bracket=[lo, hi],
               g_star_geom=(lo * hi) ** 0.5)
if near:
    summary.update(g_measured=near["g"], maxabs_out=near["maxabs_out"], rho=s91.rho_of(near, "1e-4"),
                   u=s91.REF_REL * near["maxabs_out"] / s91.STRICT,
                   frac_over=near["A_1e-4"]["frac_over"], argmax=near["A_1e-4"]["argmax"],
                   B_rho=near.get("B_1e-4", {}).get("rho"), engine_class=near.get("engine_class"))
json.dump(summary, open(SUMOUT, "w"), indent=1)
print(json.dumps(summary, indent=1)[:1500], flush=True)
