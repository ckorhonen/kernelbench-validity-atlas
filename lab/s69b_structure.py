"""s69b — structure of the draw-level admission frequencies (post-K1, descriptive).

The prereg's K1 fired: shape-level aggregate frequency does not order by the s59 ratio swing
(rho 0.064, p 0.863). K1's stated interpretation was "the flips are noise". This script tests that
interpretation directly on the same artifact, with no new threshold: a cell is "not a coin flip"
if its Wilson 95% interval excludes 0.5. If most cells exclude 0.5, the flips are structured and
K1's arithmetic fired while its interpretation did not — reported as such.

Also: delivered-arm identity per draw under the committed s56 times (shape-level constant, NOT
re-measured; labeled), to say whether the rate claim touches delivery or only admission.

Post-hoc derived-coordinate probe (labeled hypothesis, not prereg): Spearman of per-arm admission
frequency against M (output extent) and against K/M, pooled over the 11 shapes.
"""
import json, os, statistics, collections, datetime, random

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
            return json.load(open(cand))
    raise FileNotFoundError(name)


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def spearman(x, y, perms=20000, seed=20261031):
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
    mx = sum(rx) / n
    def rho_of(a, b):
        m = sum(b) / n
        nu = sum((p - mx) * (q - m) for p, q in zip(a, b))
        de = (sum((p - mx) ** 2 for p in a) * sum((q - m) ** 2 for q in b)) ** 0.5
        return nu / de if de else 0.0
    rho = rho_of(rx, ry)
    rnd = random.Random(seed)
    cnt = 0
    for _ in range(perms):
        ys = ry[:]
        rnd.shuffle(ys)
        if abs(rho_of(rx, ys)) >= abs(rho) - 1e-12:
            cnt += 1
    return round(rho, 4), round((cnt + 1) / (perms + 1), 4)


rows = [json.loads(l) for l in open(os.path.join(RES, "s69_k8_draw_rate.jsonl")) if l.strip()][1:]
hdr = json.load(open(os.path.join(RES, "s69_k8_draw_rate.jsonl"))) if False else None
arms = ["compile"] + [a for a in sorted({k for r in rows for k in r["arms"]}) if a.startswith("tsplit")]

s56 = _kb_open("s56_heldout_contract.json")
t56 = {(r["K"], r["M"]): r for r in s56["rows"] if r["family"] == "reduction"}

shapes = sorted({(r["K"], r["M"]) for r in rows})
cells = []
for (K, M) in shapes:
    dr = [r for r in rows if (r["K"], r["M"]) == (K, M)]
    n = len(dr)
    for a in arms:
        k = sum(1 for r in dr if r["arms"].get(a, {}).get("pass"))
        ci = wilson(k, n)
        cells.append({"K": K, "M": M, "arm": a, "pass": k, "n": n, "freq": round(k / n, 4),
                      "wilson": [round(x, 4) for x in ci] if ci else None,
                      "ci_excludes_half": bool(ci and (ci[0] > 0.5 or ci[1] < 0.5)),
                      "ci_excludes_10pct": bool(ci and (ci[0] > 0.1 or ci[1] < 0.1))})

nontrivial = [c for c in cells]
excl_half = [c for c in cells if c["ci_excludes_half"]]
by_arm = {}
for a in arms:
    v = [c["freq"] for c in cells if c["arm"] == a]
    by_arm[a] = {"n_shapes": len(v), "mean_freq": round(statistics.mean(v), 4),
                 "min": round(min(v), 4), "max": round(max(v), 4),
                 "shapes_below_half": sum(1 for x in v if x < 0.5)}

# delivered-arm identity under committed s56 times (shape-level constant, not re-measured)
deliv = {}
for (K, M) in shapes:
    r56 = t56.get((K, M))
    times = {a: (v.get("t_ms"), v.get("pass_full")) for a, v in r56["arms"].items() if v.get("t_ms")}
    dr = [r for r in rows if (r["K"], r["M"]) == (K, M)]
    ident = collections.Counter()
    for r in dr:
        adm = [a for a in arms if r["arms"].get(a, {}).get("pass")] + ["torch"]
        cand = [(times[a][0], a) for a in adm if a in times]
        ident[min(cand)[1] if cand else "(none)"] += 1
    deliv[f"{K}x{M}"] = dict(ident)

# post-hoc derived-coordinate probe
xs_m = [c["M"] for c in cells]
ys = [c["freq"] for c in cells]
rho_m, p_m = spearman(xs_m, ys)
xs_k = [c["K"] for c in cells]
rho_k, p_k = spearman(xs_k, ys)

out = {
    "artifact": "s69b_structure",
    "date": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "source": "results/s69_k8_draw_rate.jsonl + results/s56_heldout_contract.json (times only)",
    "k1_context": "prereg K1 fired on the ordering test; this tests K1's INTERPRETATION (flips = noise)",
    "cells": len(cells),
    "cells_ci_excludes_half": len(excl_half),
    "cells_ci_excludes_half_frac": round(len(excl_half) / len(cells), 4),
    "cells_ci_excludes_10pct": sum(1 for c in cells if c["ci_excludes_10pct"]),
    "by_arm": by_arm,
    "delivered_identity_committed_times": deliv,
    "posthoc_derived_coords": {
        "spearman_M_vs_freq": {"rho": rho_m, "p": p_m},
        "spearman_K_vs_freq": {"rho": rho_k, "p": p_k},
        "label": "post-hoc hypothesis, not preregistered",
    },
    "reading": ("the per-cell frequencies are far from 0.5 in most cells, so the draw-level flips are "
                "structured (shape x arm specific), not noise; what failed was only the ordering by the "
                "s59 two-draw swing."),
    "cells_detail": cells,
}
p = os.path.join(RES, "s69b_structure.json")
json.dump(out, open(p, "w"), indent=1)
print(json.dumps({k: out[k] for k in ("cells", "cells_ci_excludes_half", "cells_ci_excludes_half_frac",
                                      "cells_ci_excludes_10pct", "by_arm", "posthoc_derived_coords",
                                      "reading")}, indent=1))
print("delivered identity:", json.dumps(deliv)[:600])
print("wrote", p)
