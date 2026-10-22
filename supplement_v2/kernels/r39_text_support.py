"""r39 text-pass support computations (no GPU; CPU only).

1) R3 matched-n rider: replicate atlas_population's exact log2-rounded binning,
   verify against stored summaries, then compare on the COMMON subset retained
   by the 2D n>=3 bins: 2D vs 1D bytes/thread vs total-bytes binning (the
   'size-and-launch-geometry lookup' reading), plus the degeneracy algebra
   bytes == bt * threads, threads == waves * 48 * 2048.
2) Stability rider (deferred from #38): day-1 conv backfill hand-variant error
   distribution vs R1 rerun per-arm errors (median level, abs + rel).

Outputs lab/results/r39_text_support.json; prints a summary.
"""
import json, math, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUP = ROOT / "supplement_v2" / "results"
WS = ROOT / "lab" / "results"
SM_CAP = 48 * 2048

# ---------- 1) matched-n, exact replication of atlas_population ----------
ap = json.loads((SUP / "atlas_population.json").read_text())
pts = ap["points"]
for p in pts:
    p["bt"] = float(p["bt"]); p["threads"] = float(p["threads"])
    p["GBps"] = float(p["GBps"])
    p["waves"] = p["threads"] / SM_CAP
    p["bytes"] = p["bt"] * p["threads"]
    p["k1"] = round(math.log2(p["bt"]), 1)
    p["k2"] = (p["k1"], round(math.log2(max(p["waves"], 1e-9)), 1))
    p["kb"] = round(math.log2(p["bytes"]), 1)


def stats(keys, nmin):
    bins = {}
    for p in pts:
        bins.setdefault(keys(p), []).append(p["GBps"])
    keep = {k: v for k, v in bins.items() if len(v) >= nmin}
    res, cvs, n = [], [], 0
    for v in keep.values():
        n += len(v)
        med = st.median(v)
        res += [abs(x - med) / med * 100 for x in v]
        m = st.fmean(v)
        if len(v) > 1:
            cvs.append(st.stdev(v) / m * 100 if m else 0.0)
    res.sort(); cvs.sort()
    pk = lambda a, q: a[min(len(a) - 1, int(q * len(a)))] if a else None
    return {"bins": len(keep), "points": n,
            "median_resid_pct": pk(res, .5), "p90_resid_pct": pk(res, .9),
            "median_cv_pct": pk(cvs, .5)}


rep2 = stats(lambda p: p["k2"], 3)
rep1 = stats(lambda p: ("k1", p["k1"]), 1)
c2 = {}
for p in pts:
    c2.setdefault(p["k2"], []).append(p)
retained = [p for v in c2.values() if len(v) >= 3 for p in v]
keep_ids = {(p["src"], p.get("n"), p.get("grid")): True for p in retained}


def stats_sub(sub, keys, nmin):
    bins = {}
    for p in sub:
        bins.setdefault(keys(p), []).append(p["GBps"])
    keep = {k: v for k, v in bins.items() if len(v) >= nmin}
    res = []
    for v in keep.values():
        med = st.median(v)
        res += [abs(x - med) / med * 100 for x in v]
    res.sort()
    pk = lambda a, q: a[min(len(a) - 1, int(q * len(a)))] if a else None
    return {"bins": len(keep), "points": sum(len(v) for v in keep.values()),
            "median_resid_pct": pk(res, .5), "p90_resid_pct": pk(res, .9)}


m2 = stats_sub(retained, lambda p: p["k2"], 3)
m1 = stats_sub(retained, lambda p: ("k1", p["k1"]), 3)
m1n1 = stats_sub(retained, lambda p: ("k1", p["k1"]), 1)
mb = stats_sub(retained, lambda p: ("kb", p["kb"]), 3)
mbn1 = stats_sub(retained, lambda p: ("kb", p["kb"]), 1)

alg = {"bytes_eq_bt_times_threads_dev_pct": max(
        abs(p["bytes"] - p["bt"] * p["threads"]) / p["bytes"] for p in retained) * 100,
       "threads_eq_waves_sm_cap_dev_pct": max(
        abs(p["threads"] - p["waves"] * SM_CAP) / p["threads"] for p in retained) * 100,
       "distinct_bytes_bins_log2": len({p["kb"] for p in retained}),
       "distinct_1d_keys": len({p["k1"] for p in retained}),
       "distinct_2d_keys": len({p["k2"] for p in retained}),
       "n_retained": len(retained),
       "n_points_total": len(pts)}

# ---------- 2) stability: day1 vs R1 arms ----------
bf = [json.loads(l) for l in (SUP / "conv_err_backfill.jsonl").read_text().splitlines() if l.strip()]
day1 = {}
for r in bf:
    for vk, vd in r.get("delivered", {}).items():
        if vk.startswith("imp_") and isinstance(vd, dict) and vd.get("max_abs_err") is not None:
            day1.setdefault(r["src"], []).append(float(vd["max_abs_err"]))

r1 = [json.loads(l) for l in (WS / "r1_conv_gated.jsonl").read_text().splitlines() if l.strip()]
arms = {"ieee": {"abs": [], "rel": [], "fp32": []}, "tf32": {"abs": [], "rel": [], "fp32": []}}
for r in r1:
    for k, v in r["variants"].items():
        if not isinstance(v, dict) or k == "torch" or k == "torch_cmp":
            continue
        a = "ieee" if k.endswith("[ieee]") else "tf32"
        arms[a]["abs"].append(float(v["max_abs_err"]))
        arms[a]["rel"].append(float(v["rel_err"]))
        if v.get("err_vs_fp32ref") is not None:
            arms[a]["fp32"].append(float(v["err_vs_fp32ref"]))

stab = {"day1_median_abs_by_src": {k: round(st.median(v), 6) for k, v in sorted(day1.items())},
        "day1_median_abs_all": st.median([x for v in day1.values() for x in v]),
        "day1_n": sum(len(v) for v in day1.values()),
        "r1_ieee_median_abs": st.median(arms["ieee"]["abs"]),
        "r1_tf32_median_abs": st.median(arms["tf32"]["abs"]),
        "r1_ieee_median_rel": st.median(arms["ieee"]["rel"]),
        "r1_tf32_median_rel": st.median(arms["tf32"]["rel"]),
        "r1_ieee_median_rel_vs_fp32ref": st.median(arms["ieee"]["fp32"]),
        "r1_tf32_median_rel_vs_fp32ref": st.median(arms["tf32"]["fp32"]),
        "r1_n_arms": {k: len(v["abs"]) for k, v in arms.items()}}

out = {"replication_2d": rep2, "replication_1d_full": rep1,
       "stored_2d": {"bins_n3": 204, "points": 1136, "median_cv_pct": 1.2359, "median_resid": 0.55198},
       "stored_1d": {"median_resid": 35.849},
       "matched_common": {"n_retained": len(retained), "two_d": m2, "one_d": m1,
                          "one_d_n1": m1n1, "bytes_log2bin": mb, "bytes_log2bin_n1": mbn1},
       "degeneracy_algebra": alg, "stability_day1_vs_r1": stab}
(WS / "r39_text_support.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
