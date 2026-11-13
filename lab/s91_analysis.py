#!/usr/bin/env python
"""s91_analysis.py — CPU analysis of the s91 scale-boundary sweep (committed rows only; no GPU, no new thresholds).

Reads lab/results/s91_scale_boundary.jsonl (frozen grid + frozen bisections), s91b_extension.jsonl (declared
grid-floor amendment) and s91c_t2_cross_summary.json, and emits the per-tuple/per-gate crossing table plus the
branch arithmetic the prereg asks for.

Declared readings:
  (1) monotonicity: log-log slope of rho vs g over the frozen grid, per tuple and gate; violations per the
      frozen 0.90 rule.
  (2) crossing: measured point inside the final bisection bracket (not an interpolation), with maxabs_out, u,
      rho, frac_over, argmax diagnostics, B-arm rho and engine class.
  (3) collapse arithmetic: cross-tuple ratio at fixed gate, cross-gate ratio at fixed tuple. The prereg's
      BRANCH_COLLAPSE_ATOL first clause is read per tuple (u*(1e-2) vs u*(1e-4) for the SAME tuple), which is the
      natural reading; the frozen summary computed it as a min/max across gates, which is a coding slip recorded
      here rather than a threshold change.
  (4) population-vs-extremum: the g at which frac_over first exceeds 0.01 and 0.5, per tuple and gate, from the
      grid only (descriptive).
"""
import json, math, os

LAB = os.path.expanduser("~/rstation/kernelbench/lab/results")
OUT = os.path.join(LAB, "s91_analysis.json")
REF_REL = 2.2e-4

rows = []
for fn in ("s91_scale_boundary.jsonl", "s91b_extension.jsonl"):
    p = os.path.join(LAB, fn)
    if not os.path.exists(p):
        continue
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        r["_src"] = fn
        rows.append(r)
ok = [r for r in rows if r.get("status") == "ok"]
print("rows", len(rows), "ok", len(ok))

grid = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
tuples = ["T1", "T2"]
gates = [("1e-4", 1e-4), ("1e-2", 1e-2)]


def rho(r, gate):
    return r.get("A_" + gate, {}).get("rho")


def frac(r, gate):
    return r.get("A_" + gate, {}).get("frac_over")


# --- (1) monotonicity and slopes on the frozen grid only
mono = {}
for tid in tuples:
    for gate, _ in gates:
        pts = []
        for g in grid:
            cand = [r for r in ok if r["tid"] == tid and r["kind"] == "grid" and r["g"] == g]
            if cand:
                pts.append((g, rho(cand[0], gate), cand[0].get("maxabs_out"), frac(cand[0], gate)))
        viol = []
        for i in range(len(pts) - 1):
            if pts[i + 1][1] is not None and pts[i][1] is not None and pts[i + 1][1] < 0.90 * pts[i][1]:
                viol.append([pts[i][0], pts[i + 1][0], pts[i][1], pts[i + 1][1]])
        xs = [math.log(p[0]) for p in pts if p[1] and p[1] > 0]
        ys = [math.log(p[1]) for p in pts if p[1] and p[1] > 0]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        resid = max(abs(y - (my + slope * (x - mx))) for x, y in zip(xs, ys))
        mono[tid + "_" + gate] = dict(n_points=len(pts), violations=viol,
                                      loglog_slope=round(slope, 4), max_abs_log_resid=round(resid, 4),
                                      rho_series=[[p[0], None if p[1] is None else round(p[1], 5),
                                                   None if p[3] is None else round(p[3], 6)] for p in pts])

# --- (2) crossings: measured point inside the final bracket
crossings = {}
# frozen summary carries T1 both gates and T2 1e-2
frozen = json.load(open(os.path.join(LAB, "s91_scale_boundary_summary.json")))
for key, v in frozen.get("crossings", {}).items():
    if v.get("status") == "CROSSING":
        crossings[key] = dict(v)
# amendment: T2 1e-4
p = os.path.join(LAB, "s91c_t2_cross_summary.json")
if os.path.exists(p):
    c = json.load(open(p))
    if c.get("g_measured"):
        crossings["T2_1e-4"] = dict(status="CROSSING", bracket=c["bracket"], gate="1e-4", atol=1e-4,
                                    g_measured=c["g_measured"], maxabs_out=c["maxabs_out"], rho=c["rho"],
                                    frac_over=c["frac_over"], u=c["u"], argmax=c["argmax"],
                                    B_rho=c["B_rho"], engine_class=c["engine_class"], source="s91c")

for k, v in crossings.items():
    atol = v["atol"]
    am = v.get("argmax", {})
    x = am.get("x")
    v["argmax_abs_x_over_atol"] = None if x is None else round(abs(x) / atol, 3)
    v["argmax_rel_error"] = None if (x is None or x == 0) else round(abs(am.get("d", 0)) / abs(x), 6)
    v["atol_floor_predictor_u_eq_1"] = bool(v["u"] < 1.0)

# --- (3) collapse arithmetic (per-tuple cross-gate, cross-tuple same-gate)
arith = {}
for tid in tuples:
    a = crossings.get(tid + "_1e-4", {}).get("u")
    b = crossings.get(tid + "_1e-2", {}).get("u")
    if a and b:
        arith[tid + "_cross_gate"] = dict(u_strict=a, u_shipped=b, ratio=round(max(a, b) / min(a, b), 4))
for gate in ("1e-4", "1e-2"):
    us = [crossings.get(t + "_" + gate, {}).get("u") for t in tuples]
    if all(us):
        arith[gate + "_cross_tuple"] = dict(u=us, ratio=round(max(us) / min(us), 4))
arith["element_count_ratio"] = 14246720 / 2918528

# --- (4) population-vs-extremum thresholds (descriptive, grid only)
pop = {}
for tid in tuples:
    for gate, _ in gates:
        pts = []
        for g in grid:
            cand = [r for r in ok if r["tid"] == tid and r["kind"] == "grid" and r["g"] == g]
            if cand:
                pts.append((g, frac(cand[0], gate)))
        pts.sort()
        out = {}
        for bar in (0.01, 0.5):
            hit = [p for p in pts if p[1] is not None and p[1] >= bar]
            out["first_g_frac_ge_%s" % bar] = hit[0][0] if hit else None
        pop[tid + "_" + gate] = out

# --- verdict, honest
same_gate_ok = all(arith.get(g + "_cross_tuple", {}).get("ratio", 99) <= 1.25 for g in ("1e-4", "1e-2"))
cross_gate_ratios = [arith[t + "_cross_gate"]["ratio"] for t in tuples if t + "_cross_gate" in arith]
cross_gate_ok = bool(cross_gate_ratios) and max(cross_gate_ratios) <= 1.25
if same_gate_ok and cross_gate_ok:
    verdict = "BRANCH_COLLAPSE_ATOL"
elif same_gate_ok:
    verdict = "BRANCH_GATE_ONLY"
else:
    verdict = "BRANCH_PER_TUPLE"

summary = dict(
    n_rows=len(rows), n_ok=len(ok),
    instrument=dict(tieout=frozen.get("tieout"), repeat_bit_identical_frac=frozen.get("repeat_bit_identical_frac"),
                    repeat_fail=frozen.get("n_repeat_fail")),
    monotonicity=mono,
    crossings=crossings,
    collapse_arithmetic=arith,
    population_thresholds=pop,
    frozen_summary_verdict=frozen.get("verdict"),
    verdict=verdict,
    verdict_note=(
        "Same-gate cross-tuple agreement: " + ", ".join(
            "%s ratio %.4f" % (k, v["ratio"]) for k, v in arith.items() if k.endswith("_cross_tuple")) +
        ". Per-tuple cross-gate ratio: " + ", ".join(
            "%s %.4f" % (k, v["ratio"]) for k, v in arith.items() if k.endswith("_cross_gate")) +
        ". The frozen summary printed %s from a min/max pairing across gates (coding slip); the prereg's "
        "BRANCH_COLLAPSE_ATOL clause read per tuple is not met (max %.2f > 1.25), so the governing branch is %s: "
        "the crossing collapses across tuples at a FIXED gate and shifts with the gate." % (
            frozen.get("verdict"), max(cross_gate_ratios) if cross_gate_ratios else float("nan"), verdict)),
)
json.dump(summary, open(OUT, "w"), indent=1)
print(json.dumps({k: summary[k] for k in ("verdict", "collapse_arithmetic", "population_thresholds",
                                          "frozen_summary_verdict")}, indent=1)[:2500])
print("\nCROSSINGS")
for k, v in crossings.items():
    print(k, json.dumps({x: v[x] for x in ("g_measured", "maxabs_out", "rho", "frac_over", "u", "B_rho",
                                           "engine_class", "argmax_abs_x_over_atol", "argmax_rel_error") if x in v}))
print("\nMONO")
for k, v in mono.items():
    print(k, "slope", v["loglog_slope"], "resid", v["max_abs_log_resid"], "viol", v["violations"])
