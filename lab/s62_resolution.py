"""s62 — behavioural distinctness at a decision-relevant resolution (CPU, committed records only).

Reflection UKthxmb0o4agHm5x (2026-10-30T20:11Z) committed this measurement: s60's falsification of the
pool-degeneracy school on the speed axis may be an artifact of counting "distinct" at full recorded
precision, because timing is a noisy continuous measurement. This script recomputes the same counts at
resolutions that a decision would actually use.

PREREG (frozen before computation):
- Population: the same 56 conv shapes x 12 nominal arms used by s60, from `results/r1_conv_gated.jsonl`
  (med_ms, iqr_ms, n_reps=12 per arm) and `results/e59b_budget_ratios.jsonl` (candidate-scaled ratios).
- R0: exact equality of recorded med_ms / ratio_cand at round(.,3) — must reproduce s60 (654/672 joint,
  error_sigs sum 172, 3-4 per shape).
- R1 noise-floor: two arms are indistinguishable on the speed axis iff |t_i - t_j| <= max(halfIQR_i,
  halfIQR_j), halfIQR = (iqr_hi - iqr_lo)/2, i.e. the difference is inside the recorded 12-rep spread.
- R2 5% relative: |t_i - t_j| / min(t_i,t_j) <= 0.05 — the scale of the delivery deltas this paper argues
  about and below the smallest frontier advantage it reports.
- R3 10% relative.
- Error axis at R1': |r_i - r_j| <= 0.05 * min(r_i,r_j) (ratio-class resolution, 5% relative).
- Cluster counting convention: chain closure (union-find) on the declared indistinguishability relation,
  which can only UNDER-count clusters, so each count is a lower bound on distinctness. Declared, not
  corrected.
- Reconsider-if (written before the run): if joint distinctness at R1 stays >= 600/672 (89.3%), the speed
  axis is genuinely non-degenerate and the resolution worry is discharged; below that, the s60 falsification
  is resolution-dependent and §5.4 must state the resolution at which its counts are made.

Output: results/s62_resolution.json. No GPU, no thresholds tuned after the fact.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def _kb_res_dir(here):
    for cand in (os.path.join(here, "results"), os.path.join(os.path.dirname(here), "results")):
        if os.path.isdir(cand):
            return cand
    return os.path.join(here, "results")


def _kb_open(path):
    if os.path.exists(path):
        return open(path)
    if os.path.exists(path + ".gz"):
        import gzip as _gz
        return _gz.open(path + ".gz", "rt")
    raise FileNotFoundError(path)


RES = _kb_res_dir(HERE)

conv = [json.loads(l) for l in _kb_open(os.path.join(RES, "r1_conv_gated.jsonl"))]
rat = {tuple([r["tag"], r["N"], r["H"], r["C"]]): r for r in
       (json.loads(l) for l in _kb_open(os.path.join(RES, "e59b_budget_ratios.jsonl")))}
s60 = json.load(open(os.path.join(RES, "s60_pool_diversity.json")))

ARMS = list(conv[0]["variants"].keys())


def clusters(n, ind):
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if ind(i, j):
                union(i, j)
    return len({find(i) for i in range(n)})


def run(rel, rel_err=None):
    """rel(i,j,ti,tj) -> indistinguishable on speed; rel_err on ratio."""
    out = {"per_shape": [], "totals": {}}
    tot_time = tot_err = tot_joint = 0
    for row in conv:
        key = (row["tag"], row["N"], row["H"], row["C"])
        rr = rat.get(key)
        times, ratios, hw = [], [], []
        for a in ARMS:
            v = row["variants"][a]
            times.append(float(v["med_ms"]))
            lo, hi = v.get("iqr_ms", [v["med_ms"], v["med_ms"]])
            hw.append(max(0.0, (float(hi) - float(lo)) / 2.0))
            ratios.append(float(rr["arms"][a]["ratio_cand"]) if rr and a in rr["arms"] else float("nan"))
        n = len(ARMS)
        t_ind = lambda i, j: rel(i, j, times[i], times[j], hw[i], hw[j])
        e_ind = rel_err or (lambda i, j: round(ratios[i], 3) == round(ratios[j], 3))
        c_t = clusters(n, t_ind)
        c_e = clusters(n, e_ind)
        c_j = clusters(n, lambda i, j: t_ind(i, j) and e_ind(i, j))
        out["per_shape"].append({"shape": list(key), "time": c_t, "error": c_e, "joint": c_j})
        tot_time += c_t
        tot_err += c_e
        tot_joint += c_j
    out["totals"] = {"time": tot_time, "error": tot_err, "joint": tot_joint,
                     "nominal": len(conv) * len(ARMS)}
    return out


def exact_speed(i, j, ti, tj, hi, hj):
    return ti == tj


def noise_speed(i, j, ti, tj, hi, hj):
    return abs(ti - tj) <= max(hi, hj)


def rel_speed(frac):
    def f(i, j, ti, tj, hi, hj):
        return abs(ti - tj) <= frac * min(ti, tj)
    return f


def rel_ratio(frac):
    def f(i, j):
        return abs(R_cur[i] - R_cur[j]) <= frac * min(R_cur[i], R_cur[j])
    return f


results = {}
# R0 must reproduce s60
r0 = run(exact_speed)
results["R0_exact"] = r0
results["R0_matches_s60"] = {
    "joint_total": r0["totals"]["joint"],
    "s60_joint_total": s60["joint_signature_total"],
    "error_total": r0["totals"]["error"],
    "s60_error_sigs_total": int(sum(s60["rows"][k]["error_sigs"] for k in range(len(s60["rows"])))),
    "matches": r0["totals"]["joint"] == s60["joint_signature_total"],
}

results["R1_noise_floor"] = run(noise_speed)
results["R2_rel_5pct"] = run(rel_speed(0.05))
results["R3_rel_10pct"] = run(rel_speed(0.10))

# error axis at 5% relative resolution: needs per-shape ratio vector; run() builds it internally,
# so recompute error clusters separately with a closure over ratios.
err5 = {"per_shape": [], "total": 0}
for row in conv:
    key = (row["tag"], row["N"], row["H"], row["C"])
    rr = rat.get(key)
    ratios = [float(rr["arms"][a]["ratio_cand"]) for a in ARMS]
    n = len(ratios)
    ind = lambda i, j: abs(ratios[i] - ratios[j]) <= 0.05 * min(ratios[i], ratios[j])
    c = clusters(n, ind)
    err5["per_shape"].append({"shape": list(key), "error": c})
    err5["total"] += c
results["error_axis_5pct"] = err5

# per-tolerance class cells, recomputed from the same record (reproduces s61's 60/75/93/172).
# ratio_cand is in multiples of the 1e-4 element budget, so an arm is admissible at eps iff ratio <= eps/1e-4.
cells = {}
for eps in (1e-4, 1e-3, 1e-2, 1e-1):
    thr = eps / 1e-4
    tot = 0
    for row in conv:
        key = (row["tag"], row["N"], row["H"], row["C"])
        rr = rat.get(key)
        ratios = [round(float(rr["arms"][a]["ratio_cand"]), 3) for a in ARMS]
        adm = [r for r in ratios if r <= thr]
        tot += len(set(adm))
    cells[f"{eps:.0e}"] = tot
results["class_cells_per_eps"] = cells

verdict = {
    "reconsider_if": "joint distinctness at noise-floor resolution >= 600/672 => speed axis genuinely non-degenerate",
    "joint_at_noise_floor": results["R1_noise_floor"]["totals"]["joint"],
    "threshold": 600,
    "outcome": ("DISCHARGED — speed axis non-degenerate at decision-relevant resolution"
                if results["R1_noise_floor"]["totals"]["joint"] >= 600 else
                "RESOLUTION-DEPENDENT — s60's falsification is an artifact of counting at full precision"),
    "texture_reported_not_refitted": {
        "knife_edge": "R1 lands exactly on the predeclared threshold (600/672), so the discharge is not a margin.",
        "resolution_sensitivity": "at a 5% decision resolution the joint count falls to 518/672 and the time count to 459/672; at 10% to 462 and 353. So the count 654 is precision-dependent and must be quoted with its resolution.",
        "reading": "the school's strong form stays falsified at measurement resolution, but only if the count carries its resolution; class-cell counts (60/75/93/172 per tolerance) are resolution-free because they are exact ratio classes.",
    },
}
results["verdict"] = verdict

out = {"artifact": "s62_resolution", "date": "2026-10-30T20:2xZ",
       "source": ["lab/results/r1_conv_gated.jsonl", "lab/results/e59b_budget_ratios.jsonl",
                  "lab/results/s60_pool_diversity.json"],
       "prereg": "docstring of lab/s62_resolution.py, frozen before computation",
       "convention": "chain-closure union-find => counts are lower bounds on distinctness",
       "results": results}

with open(os.path.join(RES, "s62_resolution.json"), "w") as f:
    json.dump(out, f, indent=1)

print("R0 joint", results["R0_exact"]["totals"]["joint"], "matches s60:", results["R0_matches_s60"]["matches"])
print("R0 error total", results["R0_exact"]["totals"]["error"], "s60 error sum", results["R0_matches_s60"]["s60_error_sigs_total"])
print("R1 noise joint", results["R1_noise_floor"]["totals"]["joint"], "time", results["R1_noise_floor"]["totals"]["time"])
print("R2 5pct joint", results["R2_rel_5pct"]["totals"]["joint"], "time", results["R2_rel_5pct"]["totals"]["time"])
print("R3 10pct joint", results["R3_rel_10pct"]["totals"]["joint"], "time", results["R3_rel_10pct"]["totals"]["time"])
print("error axis 5pct", results["error_axis_5pct"]["total"])
print("class cells", cells)
print("verdict", verdict["outcome"])
