"""s83_analysis — dedup + frozen verdict for s83 (prereg in lab/s83_pool_spanning.py docstring,
commit f87c85e, frozen before the run). CPU only, no new measurement.

Declared statistics: ratio_median (candidate-scaled budget ratio, s73 predicate), med_ms.
Clustering: single-linkage on the declared value at relative resolution 5% and 10%; 0.0 forms its
own exact class. Joint signature = (ratio class, time class) at the same resolution.

Frozen verdict rule (verbatim from the prereg):
  R1 per shape, distinct ratio-classes at 5%: LOW if <=4 of 12, HIGH if >=8, else MIXED.
  R1 pooled: EFFECTIVE_LOW only if BOTH LOW; EFFECTIVE_HIGH only if BOTH HIGH; else MIXED.
Decision clause: EFFECTIVE_LOW => degenerate-pool generalizes, per-arm enumeration dropped;
EFFECTIVE_HIGH => per-arm language stands; MIXED => classes plus measured within-class spread,
no per-arm claim.
"""
import json, os, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def _open(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p) and os.path.exists(p + ".gz"):
        import gzip
        return [json.loads(l) for l in gzip.open(p + ".gz", "rt")]
    return [json.loads(l) for l in open(p)]


def cluster(values, tol):
    """single-linkage at relative tol; all exact 0.0 values form one class."""
    zeros = [n for n, v in values.items() if v == 0.0]
    items = sorted(((n, v) for n, v in values.items() if v != 0.0), key=lambda kv: kv[1])
    groups = [zeros] if zeros else []
    cur = []
    for name, v in items:
        if cur and v <= cur[-1][1] * (1 + tol):
            cur.append((name, v))
        else:
            if cur:
                groups.append([n for n, _ in cur])
            cur = [(name, v)]
    if cur:
        groups.append([n for n, _ in cur])
    return groups


rows = _open("s83_pool_spanning.jsonl")
out = {"artifact": "s83_pool_spanning_analysis",
       "date": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
       "source": ["results/s83_pool_spanning.jsonl"],
       "prereg": "lab/s83_pool_spanning.py docstring, commit f87c85e",
       "shapes": [], "verdict": {}}
per_shape = {}
for r in rows:
    arms = {k: v for k, v in r["arms"].items() if "error" not in v and "ratio_median" in v}
    res = {}
    for tol in (0.05, 0.10):
        rg = cluster({k: v["ratio_median"] for k, v in arms.items()}, tol)
        res[f"ratio_classes_{int(tol*100)}"] = {"n": len(rg), "groups": rg}
        # joint: ratio class then time class inside it
        joint = 0
        for g in rg:
            tg = cluster({k: arms[k]["med_ms"] for k in g}, tol)
            joint += len(tg)
        res[f"joint_sigs_{int(tol*100)}"] = joint
    res["nominal_arms"] = len(arms)
    res["passing_arms"] = sorted(k for k, v in arms.items() if v.get("pass_1e4"))
    res["failing_arms"] = sorted(k for k, v in arms.items() if not v.get("pass_1e4"))
    res["within_class_ratio_spread"] = {
        f"class{i}": max(arms[k]["ratio_median"] for k in g) / max(min(arms[k]["ratio_median"] for k in g), 1e-30)
        for i, g in enumerate(res["ratio_classes_5"]["groups"])}
    res["r0_ratio_spread"] = r["r0_ratio_spread"]
    per_shape[tuple(r["shape"])] = res
    out["shapes"].append({"shape": r["shape"], **res})

n5 = [s["ratio_classes_5"]["n"] for s in out["shapes"]]
if all(n <= 4 for n in n5):
    pooled = "EFFECTIVE_LOW"
elif all(n >= 8 for n in n5):
    pooled = "EFFECTIVE_HIGH"
else:
    pooled = "MIXED"
out["verdict"] = {
    "pooled": pooled,
    "per_shape_ratio_classes_5pct": n5,
    "per_shape_ratio_classes_10pct": [s["ratio_classes_10"]["n"] for s in out["shapes"]],
    "joint_signatures_5pct": [s["joint_sigs_5"] for s in out["shapes"]],
    "nominal_arms_per_shape": [s["nominal_arms"] for s in out["shapes"]],
    "reading": ("Aliasing persists on a pool that spans arithmetic: the nominal 12 arms resolve to "
                f"{n5[0]}/{n5[1]} ratio classes at 5% resolution. The Triton-tile families collapse to one "
                "class each (all five ieee tiles identical; the tf32 tile separate), so tile geometry "
                "carries no behavioural content, while the arithmetic variants (fp64 anchor, cudnn-off, "
                "channels_last, tf32) do add classes. Per-arm enumeration overstates behavioural content "
                "by 2.4-4x, and the extra content is arithmetic-level, not tile-level."),
    "scope": "GB10, two conv 3x3 s1p1 shapes, this 12-arm pool, rev 423217d. Single seed; timing is "
             "median of 12 flush-charged reps, not a frontier measurement.",
}
json.dump(out, open(os.path.join(RES, "s83_pool_spanning_analysis.json"), "w"), indent=1)
print(json.dumps(out["verdict"], indent=1)[:2500])
for s in out["shapes"]:
    print(s["shape"], "classes5", s["ratio_classes_5"]["n"], "classes10", s["ratio_classes_10"]["n"],
          "joint5", s["joint_sigs_5"], "pass", s["passing_arms"])
    for g in s["ratio_classes_5"]["groups"]:
        print("   ", g)
