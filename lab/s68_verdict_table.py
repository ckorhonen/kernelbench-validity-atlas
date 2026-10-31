#!/usr/bin/env python3
"""s68 — reference-conditional verdict table (stated policy), CSV + markdown.

PREREG (docstring frozen before computation): table rows are the 56 gate_stability rows, keyed
(H,N,C); policy = fastest admitted arm over the 12-arm pool with baseline fallback only when the
pool is empty; times from lab/r1_times.json; labels from the recorded pass maps. Columns are the
delivered arm and admissible-pool size under each of the three reference realizations (default,
tf32-off, fp64 anchor), plus change flags. No filtering, no thresholds.
"""
import json, os, csv
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
gs = [json.loads(l) for l in open(os.path.join(RES, "gate_stability.jsonl")) if l.strip()]
times = json.load(open(os.path.join(HERE, "r1_times.json")))

def delivered(passmap, tmap, arms):
    pool = [a for a in arms if a in tmap]
    adm = sorted([a for a in pool if passmap.get(a)])
    if not adm:
        return "torch", 0
    best = min(tmap[a] for a in adm)
    tie = sorted([a for a in adm if abs(tmap[a] - best) <= 1e-12 * max(1.0, best)])
    return tie[0], len(adm)

rows = []
for r in gs:
    key = "%s|%s|%s|%s" % (r["tag"], r["H"], r["N"], r["C"])
    tmap = times.get(key)
    if tmap is None:
        continue
    arms = r["arms"]
    d_def, n_def = delivered(r["pass_def"], tmap, arms)
    d_ieee, n_ieee = delivered(r["pass_ieee"], tmap, arms)
    d_64, n_64 = delivered(r["pass_64"], tmap, arms)
    rows.append(dict(key=key, tag=r["tag"], H=r["H"], N=r["N"], C=r["C"],
                     delivered_default=d_def, admitted_default=n_def,
                     delivered_tf32off=d_ieee, admitted_tf32off=n_ieee,
                     delivered_fp64=d_64, admitted_fp64=n_64,
                     arm_change_tf32off=int(d_def != d_ieee),
                     arm_change_fp64=int(d_def != d_64),
                     pool_change_tf32off=int(n_def != n_ieee),
                     pool_change_fp64=int(n_def != n_64)))

rows.sort(key=lambda x: (x["tag"], x["H"], x["N"], x["C"]))
out = os.path.join(RES, "rfc_verdict_table.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for x in rows:
        w.writerow(x)

summary = {
    "rows": len(rows),
    "arm_change_tf32off": sum(x["arm_change_tf32off"] for x in rows),
    "arm_change_fp64": sum(x["arm_change_fp64"] for x in rows),
    "pool_change_tf32off": sum(x["pool_change_tf32off"] for x in rows),
    "pool_change_fp64": sum(x["pool_change_fp64"] for x in rows),
    "empty_pool_default": sum(1 for x in rows if x["admitted_default"] == 0),
    "empty_pool_tf32off": sum(1 for x in rows if x["admitted_tf32off"] == 0),
    "empty_pool_fp64": sum(1 for x in rows if x["admitted_fp64"] == 0),
}
with open(os.path.join(RES, "rfc_verdict_table_summary.json"), "w") as f:
    json.dump(summary, f, indent=1)
print(json.dumps(summary, indent=1))
print("csv:", out, os.path.getsize(out), "bytes")
# illustrative rows for the post
for k in ("m3m2|4|1|256", "m3m2|14|4|256", "w4|5|2|96"):
    for x in rows:
        if x["key"] == k:
            print(x)
