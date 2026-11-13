#!/usr/bin/env python
"""s95_analysis.py — coded analysis of results/s95_draw_repeat.jsonl under the s95 prereg (80a91f6).

Declared post-hoc additions are labelled: the per-draw slope stability statistic and the amplitude
spread are computed here because the prereg's K2 clause (CV>10%) fires, so no power-law claim may be
made; the exponent's draw-stability is reported as a descriptive statistic, not as a law.

Prereg rules implemented verbatim:
  K1 instrument  : repeat of draw 0 at K=1024 bit-identical (rho_ref, rho_cand, frac_over).
  K2 stability   : pooled CV of rho_ref over the 8 independent draws at any K > 10% => DRAWS_UNSTABLE.
  dip clause     : FLOOR_REAL iff rho(1024) > rho(2048) in >=7/8 draws AND |mean diff| > 2*pooled sd;
                   FLOOR_NOISE iff that count <=4 OR |diff| <= 2*pooled sd; FLOOR_MIXED otherwise.
  share clause   : SHARE_STABLE iff |mean(share) - 0.560| <= 0.05 else SHARE_UNSTABLE.
"""
import json
import math
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "..", "results", "s95_draw_repeat.jsonl")
OUT = os.path.join(HERE, "..", "results", "s95_analysis.json")

rows = [json.loads(l) for l in open(IN)]
ok = [r for r in rows if r["status"] == "ok"]
d0 = [r for r in ok if r["draw"] == 0 and r["K"] == 1024]
rep = d0[1] if len(d0) > 1 else None
# de-dup: keep first occurrence per (draw,K); the second D0_K1024 record is the instrument repeat
seen = set()
uniq = []
for r in ok:
    key = (r["draw"], r["K"])
    if key in seen:
        continue
    seen.add(key)
    uniq.append(r)

inst = dict(
    repeat_rho_identical=bool(rep and rep["rho_ref"] == d0[0]["rho_ref"]),
    repeat_rho_cand_identical=bool(rep and rep["rho_cand"] == d0[0]["rho_cand"]),
    repeat_frac_identical=bool(rep and rep["frac_over_ref_strict"] == d0[0]["frac_over_ref_strict"]),
)
k1 = all(inst.values())

by_k = {}
for r in uniq:
    by_k.setdefault(r["K"], []).append(r)

per_k = {}
cv_max = 0.0
for K in sorted(by_k):
    v = [r["rho_ref"] for r in by_k[K]]
    m = st.mean(v)
    sd = st.stdev(v)
    cv = sd / m
    cv_max = max(cv_max, cv)
    per_k[K] = dict(n=len(v), mean=m, sd=sd, cv=cv, min=min(v), max=max(v),
                    spread=max(v) / min(v),
                    frac_over_ref_strict_mean=st.mean([r["frac_over_ref_strict"] for r in by_k[K]]),
                    frac_over_ref_ship_mean=st.mean([r["frac_over_ref_ship"] for r in by_k[K]]))

draws = {}
for r in uniq:
    draws.setdefault(r["draw"], {})[r["K"]] = r["rho_ref"]

ndip = sum(1 for d, v in draws.items() if v.get(1024, 0) > v.get(2048, 0))
m1 = st.mean([v[1024] for v in draws.values()])
m2 = st.mean([v[2048] for v in draws.values()])
s1 = st.stdev([v[1024] for v in draws.values()])
s2 = st.stdev([v[2048] for v in draws.values()])
pooled_sd = math.sqrt((s1 ** 2 + s2 ** 2) / 2)
diff = m1 - m2
if ndip >= 7 and abs(diff) > 2 * pooled_sd:
    dip_branch = "FLOOR_REAL"
elif ndip <= 4 or abs(diff) <= 2 * pooled_sd:
    dip_branch = "FLOOR_NOISE"
else:
    dip_branch = "FLOOR_MIXED"

slopes = {}
for d, v in sorted(draws.items()):
    ks = sorted(v)
    xs = [math.log(k) for k in ks]
    ys = [math.log(v[k]) for k in ks]
    mx, my = st.mean(xs), st.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    slopes[d] = b

shares = [r["share"] for r in uniq if r.get("share") is not None]
share_stats = dict(n=len(shares), mean=st.mean(shares), sd=st.stdev(shares),
                   min=min(shares), max=max(shares)) if shares else dict(n=0)
share_branch = ("SHARE_STABLE" if shares and abs(st.mean(shares) - 0.560) <= 0.05 else "SHARE_UNSTABLE")

verdict = dict(
    instrument_pass=k1,
    instrument=inst,
    cv_max=cv_max,
    k2_fired=bool(cv_max > 0.10),
    per_k={str(k): v for k, v in per_k.items()},
    dip=dict(n_draws_with_dip=ndip, n_draws=len(draws), mean_1024=m1, mean_2048=m2,
             pooled_sd=pooled_sd, mean_diff=diff, abs_diff_over_pooled_sd=abs(diff) / pooled_sd,
             branch=dip_branch),
    slope_descriptive=dict(mean=st.mean(slopes.values()), sd=st.stdev(slopes.values()),
                           per_draw={str(d): b for d, b in slopes.items()}),
    share=share_stats,
    share_branch=share_branch,
)
# the law claim is only issued if neither K2 nor K3 fired
verdict["law_claim_permitted"] = not verdict["k2_fired"] and k1

with open(OUT, "w") as f:
    json.dump(verdict, f, indent=1)

print(json.dumps(verdict, indent=1)[:2600])
