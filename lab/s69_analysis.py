"""s69 analysis — admission frequency over input draws, against the predeclared kill clauses.

CPU only, arithmetic on committed records: results/s69_k8_draw_rate.jsonl (this run) and
results/s59_f1_input_draw.json (the ratio swings the prereg orders against), plus
results/s56_heldout_contract.json for the committed per-arm times used only to name the
delivered arm (labels are the object; timing is a shape-level constant here and is labeled).

Prereg: lab/s69_prereg.md. Kill clauses evaluated exactly as written:
  K1: shape-level admission frequency flat in ratio swing (|Spearman rho| < 0.3 or p > 0.05 over
      11 shapes) => flips are noise, report discordance fraction only.
  K2: if every shape's aggregate frequency CI overlaps the pooled s59 baseline 11.4% (26/228),
      the rate adds no information beyond s59 => claim narrows to "admission varies across draws".
No thresholds are added after the fact.
"""
import json, os, statistics, collections, itertools, datetime

HERE = os.path.dirname(os.path.abspath(__file__))


def _kb_res_dir(here):
    for cand in (os.path.join(here, "results"), os.path.join(os.path.dirname(here), "results")):
        if os.path.isdir(cand):
            return cand
    return os.path.join(here, "results")


RES = _kb_res_dir(HERE)


def _kb_open(name):
    for cand in (os.path.join(RES, name), os.path.join(RES, name + ".gz"), name):
        if os.path.exists(cand):
            if cand.endswith(".gz"):
                import gzip
                return gzip.open(cand, "rt")
            return open(cand)
    raise FileNotFoundError(name)


def wilson(k, n, z=1.96):
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, c - h), min(1.0, c + h)


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for t in range(i, j + 1):
                r[order[t]] = avg
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    rho = num / den if den else 0.0
    # permutation p, one-sided |rho|, 20000 permutations, fixed seed
    import random
    rnd = random.Random(20261031)
    cnt = 0
    NPERM = 20000
    for _ in range(NPERM):
        ys = ry[:]
        rnd.shuffle(ys)
        m = sum(ys) / n
        nu = sum((a - mx) * (b - m) for a, b in zip(rx, ys))
        de = (sum((a - mx) ** 2 for a in rx) * sum((b - m) ** 2 for b in ys)) ** 0.5
        if de and abs(nu / de) >= abs(rho) - 1e-12:
            cnt += 1
    return rho, (cnt + 1) / (NPERM + 1)


rows = [json.loads(l) for l in _kb_open("s69_k8_draw_rate.jsonl") if l.strip()]
hdr = rows[0]["header"]
data = rows[1:]
s59 = json.load(_kb_open("s59_f1_input_draw.json"))

# ratio swing per shape from s59 flips (max/min of the two draws' margins, per the prereg's definition)
swing = {}
for f in s59["flips"]:
    if f["family"] != "reduction":
        continue
    k = (f["shape"][0], f["shape"][1])
    a, b = f["recorded_ratio"], f["fresh_ratio"]
    sw = max(a, b) / max(1e-9, min(a, b))
    swing[k] = max(swing.get(k, 0.0), sw)

shapes = sorted({(r["K"], r["M"]) for r in data})
arms = hdr["arms"]

per_shape = {}
all_arm_freq = []
for (K, M) in shapes:
    dr = [r for r in data if (r["K"], r["M"]) == (K, M)]
    n = len(dr)
    freq = {}
    for a in arms:
        k = sum(1 for r in dr if r["arms"].get(a, {}).get("pass"))
        lo, hi = wilson(k, n)
        freq[a] = {"pass": k, "n": n, "freq": round(k / n, 4) if n else None,
                   "wilson": [round(lo, 4), round(hi, 4)] if lo is not None else None}
        if n:
            all_arm_freq.append(k / n)
    any_pass = sum(1 for r in dr if any(v.get("pass") for v in r["arms"].values()))
    sets = collections.Counter(tuple(sorted(a for a, v in r["arms"].items() if v.get("pass"))) for r in dr)
    ref_rho = [r["ref_rho_fp64"] for r in dr]
    per_shape[f"{K}x{M}"] = {
        "n_draws": n, "swing_s59": round(swing.get((K, M), float("nan")), 4),
        "arm_freq": freq,
        "any_arm_admitted_draws": any_pass, "any_arm_freq": round(any_pass / n, 4) if n else None,
        "any_arm_wilson": [round(x, 4) for x in wilson(any_pass, n)] if n else None,
        "distinct_admitted_sets": len(sets),
        "admitted_set_counts": {"|".join(s) if s else "(none)": c for s, c in sets.most_common()},
        "ref_rho_fp64_median": round(statistics.median(ref_rho), 6),
        "ref_rho_fp64_max": round(max(ref_rho), 6),
    }

# shape-level aggregate for the ordering test: fraction of arm-checks admitted
agg = {}
for key, v in per_shape.items():
    tot = sum(f["pass"] for f in v["arm_freq"].values())
    den = sum(f["n"] for f in v["arm_freq"].values())
    agg[key] = {"admitted": tot, "checks": den, "freq": round(tot / den, 4) if den else None,
                "wilson": [round(x, 4) for x in wilson(tot, den)] if den else None}

xs = [per_shape[k]["swing_s59"] for k in agg]
ys = [agg[k]["freq"] for k in agg]
rho, p = spearman(xs, ys)

pooled_base = 26 / 228
k2_hits = [k for k in agg if agg[k]["wilson"][0] <= pooled_base <= agg[k]["wilson"][1]]

k1_fired = abs(rho) < 0.3 or p > 0.05
k2_fired = len(k2_hits) == len(agg)

out = {
    "artifact": "s69_analysis",
    "date": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "source": ["results/s69_k8_draw_rate.jsonl", "results/s59_f1_input_draw.json"],
    "header": {k: hdr[k] for k in ("date", "prereg", "contract", "arms", "n_draws", "prereg_slip")},
    "population": {"shapes": len(shapes), "draws": len(data), "checks": len(data) * len(arms)},
    "per_shape": per_shape,
    "shape_aggregate": agg,
    "ordering_test": {"statistic": "Spearman(swing_s59, shape_admission_freq)", "rho": round(rho, 4),
                      "p_permutation_one_sided_abs": round(p, 4), "n_shapes": len(xs),
                      "K1_fired": k1_fired},
    "K2": {"pooled_baseline": round(pooled_base, 4), "shapes_whose_CI_contains_baseline": k2_hits,
           "n": len(k2_hits), "K2_fired": k2_fired},
    "discordance": {
        "per_arm_check_mean_freq": round(statistics.mean(all_arm_freq), 4) if all_arm_freq else None,
        "per_arm_check_min": round(min(all_arm_freq), 4) if all_arm_freq else None,
        "per_arm_check_max": round(max(all_arm_freq), 4) if all_arm_freq else None,
    },
    "verdict": ("K1_FIRED_DISCORDANCE_ONLY" if k1_fired else "RATE_ORDERED_BY_SWING"),
    "scope": ("11 s59 flip shapes, this draw generator, GB10 rev 423217d; labels only, no timing; "
              "the s56 pool has 8 arms, not the prereg's 12 (slip recorded)."),
}

os.makedirs(RES, exist_ok=True)
p_out = os.path.join(RES, "s69_analysis.json")
json.dump(out, open(p_out, "w"), indent=1)
print(json.dumps({k: out[k] for k in ("population", "ordering_test", "K2", "discordance", "verdict")}, indent=1))
for k in sorted(agg):
    print(k, "swing", per_shape[k]["swing_s59"], "agg", agg[k]["freq"], agg[k]["wilson"],
          "any", per_shape[k]["any_arm_freq"], "sets", per_shape[k]["distinct_admitted_sets"])
print("wrote", p_out)
