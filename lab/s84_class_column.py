#!/usr/bin/env python
"""s84_class_column.py — the attributable column, at class granularity, on the 54 exceed rows.

PREREG (frozen before the first computation; do not edit after).

WHY THIS EXISTS. s83 (commit f87c85e, frozen pre-run) measured behavioural content on an
arithmetic-spanning pool: 12 nominal arms resolve to 3-5 ratio classes at 5% resolution (4 at 10%),
five Triton ieee tiles are ONE arm, only the tf32 tile is distinct, and channels_last/compile are
bit-identical to the default on one shape and fail at 78% of elements on another. Per that prereg's
decision, the deliverable carries CLASS-level entries plus a measured within-class spread and drops
per-arm enumeration. This script builds that column on the 54 exceed rows of s73c_merged_56.jsonl,
where the per-arm decomposition already exists, and tests whether the class key can carry the
magnitude or only the share.

POPULATION (rule, not outcome). All rows of lab/results/s73c_merged_56.jsonl with kind == "exceed"
(54). The 8 negative controls are reported as a gate check, never pooled into class statistics.
Join keys: (level, problem_id) to lab/results/s75_engine_class_s73pop.jsonl for the profiler-measured
engine class; the s73 decomp blocks decomp_c1_compile and decomp_c2_tf32off supply the per-arm
counts, shares, and tail ratios. No GPU, no new measurement.

COLUMN SCHEMA (one row per (task, arm)); arms are C1 = torch.compile fp32 default flags,
C2 = same model with cudnn/matmul tf32 off. Fields:
  level, problem_id, op_class, engine_class_refined, arm, arm_relation,
  verdict_default_1e4, verdict_anchor_1e4, verdict_default_1e2,
  fail_default_1e4, fail_anchor_1e4, total, ref_induced, cand_induced,
  ref_share, cand_share, frac_over_ref, max_ratio_ref, max_ratio_cand,
  bit_identical_ca, draw_probe (bool), draw_spread_max_over_min, draw_frac_over_range.
arm_relation is BIT_IDENTICAL when decomp.bit_identical_ca is true (measured zero |C-A|), else
NUMERICALLY_DISTINCT. A row with no measured share (fail_default == 0) carries ref_share = null and
is excluded from share aggregation; it is NOT imputed.

Q1 — SHARE AGGREGATION (the attributable part). Pooled ref_share over rows with a measured share,
plus per-cell (engine_class_refined x op_class) median and min/max for n >= 3 cells.
  Verdict SHARE_REFERENCE_DOMINATED iff pooled median >= 0.95 AND every n>=3 cell median >= 0.90.
  Verdict SHARE_MIXED otherwise (printed with the offending cells).
  No kill clause: this is a description of committed numbers, and either branch is reportable.

Q2 — TAIL BOUND (the design question). For every cell (engine_class_refined x op_class) with n >= 5
rows carrying a measured max_ratio_ref for the C2 arm, compute
  cell_tail_spread = max(max_ratio_ref) / min(max_ratio_ref).
  Verdict TAIL_CELL_BOUNDED iff every such cell has spread <= 10.0.
  Verdict TAIL_PER_SHAPE otherwise, with the offending cells named.
KILL (declared): if TAIL_PER_SHAPE, no cell-level magnitude claim is made — the column carries
class labels for the share only and the per-shape tail column stands unchanged. If TAIL_CELL_BOUNDED,
cells with n>=5 become the magnitude unit for those cells and only those cells.

Q3 — DRAW INTERVAL vs CELL SPREAD. For the tasks that have draw probes (s79d summary, s80b summary,
keyed "L{level}/{problem_id}"), record rho draw spread and frac_over_1e-4 range, and compare the
median probed draw spread against the smallest cell_tail_spread from Q2.
  Verdict DRAW_NOISE_BELOW_CELL_SPREAD iff median draw spread < smallest cell spread; else
  DRAW_NOISE_COMPARABLE, which means a cell bound at that width cannot be distinguished from input
  noise and is labelled descriptive only.

Q4 — VERDICT CLASS-CONDITIONALITY (headline count, no threshold). Number of tasks where C1 passes and
C2 fails at the 1e-4 default-reference gate, and the number of tasks where C1 is BIT_IDENTICAL (pass
inherited by construction). Reported as counts over 54, never as a rate claim.

Q5 — CONTROL GATE. Every control row must show ref_over_budget / total <= 0.10 at 1e-4; a violation
marks the harness defective and the class statistics are reported as provisional.

INSTRUMENT NOTES CARRIED. The bias/median coordinate is degenerate on near-exact tensors (s81/s82);
the tail (max_ratio_ref, frac_over_ref) is the admission-relevant statistic and is what Q2 uses.
The kernel-instance key (top-1 profiler name) is not reproducible across passes (s80c), so no
instance key enters any verdict here; engine class does (s72 fresh-process 26/26). Every count that
depends on resolution states its resolution.

OUTPUTS: results/s84_class_column.json, results/s84_class_column.csv, results/s84_class_column_rows.csv
"""
import csv, json, os, statistics

LAB = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(LAB, "results")
S73 = os.path.join(RES, "s73c_merged_56.jsonl")
S75 = os.path.join(RES, "s75_engine_class_s73pop.jsonl")
S79D = os.path.join(RES, "s79d_dgrad_draw_verified_summary.json")
S80B = os.path.join(RES, "s80b_dgrad_family_draw_summary.json")
OUTJ = os.path.join(RES, "s84_class_column.json")
OUTC = os.path.join(RES, "s84_class_column.csv")
OUTR = os.path.join(RES, "s84_class_column_rows.csv")

TAIL_BOUND = 10.0
SHARE_POOLED_BAR = 0.95
SHARE_CELL_BAR = 0.90
MIN_CELL_N = 5
CONTROL_BAR = 0.10


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


rows = load_jsonl(S73)
eng = {}
for r in load_jsonl(S75):
    eng[(r["level"], r["problem_id"])] = r

draw = {}
for path in (S79D, S80B):
    d = json.load(open(path))
    for k, v in d.get("per_tuple", {}).items():
        lv, pid = k.split("/")
        draw[(int(lv[1:]), int(pid))] = {
            "source": os.path.basename(path),
            "draw_spread": v.get("draw_spread_max_over_min"),
            "frac_over_min": v.get("frac_over_1e-4_min"),
            "frac_over_max": v.get("frac_over_1e-4_max"),
        }

col_rows = []
for r in rows:
    if r.get("kind") != "exceed":
        continue
    key = (r["level"], r["problem_id"])
    e = eng.get(key, {})
    engine_class = e.get("engine_class_refined")
    for arm, tag in (("C1_compile", "decomp_c1_compile"), ("C2_tf32off", "decomp_c2_tf32off")):
        d = (r.get(tag) or {}).get("1e-4")
        if not d:
            continue
        dr = draw.get(key)
        col_rows.append({
            "level": r["level"], "problem_id": r["problem_id"], "op_class": r.get("op_class"),
            "engine_class_refined": engine_class, "arm": arm,
            "arm_relation": "BIT_IDENTICAL" if d.get("bit_identical_ca") else "NUMERICALLY_DISTINCT",
            "verdict_default_1e4": d.get("verdict_default"),
            "verdict_anchor_1e4": d.get("verdict_anchor"),
            "verdict_default_1e2": ((r.get(tag) or {}).get("1e-2") or {}).get("verdict_default"),
            "fail_default_1e4": d.get("fail_default"), "fail_anchor_1e4": d.get("fail_anchor"),
            "total": d.get("total"), "ref_induced": d.get("ref_induced"), "cand_induced": d.get("cand_induced"),
            "ref_share": d.get("ref_share_of_fails"), "cand_share": d.get("cand_share_of_fails"),
            "frac_over_ref": (d.get("ref_over_budget") / d["total"]) if d.get("total") else None,
            "max_ratio_ref": d.get("max_ratio_ref"), "max_ratio_cand": d.get("max_ratio_cand"),
            "bit_identical_ca": bool(d.get("bit_identical_ca")),
            "draw_probe": bool(dr),
            "draw_spread_max_over_min": dr["draw_spread"] if dr else None,
            "draw_frac_over_min": dr["frac_over_min"] if dr else None,
            "draw_frac_over_max": dr["frac_over_max"] if dr else None,
        })

# Q1 share aggregation
shares = [c for c in col_rows if c["ref_share"] is not None]
pooled = {
    "n_rows_with_share": len(shares),
    "arms": sorted({c["arm"] for c in shares}),
    "ref_share_median": statistics.median([c["ref_share"] for c in shares]) if shares else None,
    "ref_share_min": min(c["ref_share"] for c in shares) if shares else None,
    "ref_share_max": max(c["ref_share"] for c in shares) if shares else None,
    "cand_share_median": statistics.median([c["cand_share"] for c in shares]) if shares else None,
    "cand_share_max": max(c["cand_share"] for c in shares) if shares else None,
    "fail_default_total": sum(c["fail_default_1e4"] for c in shares),
    "ref_induced_total": sum(c["ref_induced"] for c in shares),
    "cand_induced_total": sum(c["cand_induced"] for c in shares),
}
cells_share = {}
for c in shares:
    k = (c["engine_class_refined"], c["op_class"], c["arm"])
    cells_share.setdefault(k, []).append(c["ref_share"])
q1_cells = []
for k, v in sorted(cells_share.items()):
    if len(v) >= 3:
        q1_cells.append({"engine_class": k[0], "op_class": k[1], "arm": k[2], "n": len(v),
                         "ref_share_median": statistics.median(v), "min": min(v), "max": max(v)})
q1_ok = (pooled["ref_share_median"] is not None and pooled["ref_share_median"] >= SHARE_POOLED_BAR
         and all(x["ref_share_median"] >= SHARE_CELL_BAR for x in q1_cells))
Q1 = "SHARE_REFERENCE_DOMINATED" if q1_ok else "SHARE_MIXED"

# Q2 tail bound
cells_tail = {}
for c in col_rows:
    if c["arm"] != "C2_tf32off" or c["max_ratio_ref"] is None or c["engine_class_refined"] is None:
        continue
    cells_tail.setdefault((c["engine_class_refined"], c["op_class"]), []).append(c)
q2_cells = []
for k, v in sorted(cells_tail.items()):
    if len(v) < MIN_CELL_N:
        continue
    vals = [x["max_ratio_ref"] for x in v]
    q2_cells.append({"engine_class": k[0], "op_class": k[1], "n": len(v),
                     "tail_min": min(vals), "tail_max": max(vals),
                     "cell_tail_spread": (max(vals) / min(vals)) if min(vals) > 0 else None,
                     "frac_over_ref_median": statistics.median([x["frac_over_ref"] for x in v])})
bad = [x for x in q2_cells if x["cell_tail_spread"] is None or x["cell_tail_spread"] > TAIL_BOUND]
Q2 = "TAIL_CELL_BOUNDED" if q2_cells and not bad else "TAIL_PER_SHAPE"

# Q3 draw interval vs cell spread
probed = [c for c in col_rows if c["draw_probe"] and c["draw_spread_max_over_min"]]
med_draw = statistics.median([c["draw_spread_max_over_min"] for c in probed]) if probed else None
smallest_cell = min((x["cell_tail_spread"] for x in q2_cells if x["cell_tail_spread"]), default=None)
Q3 = ("DRAW_NOISE_BELOW_CELL_SPREAD"
      if (med_draw is not None and smallest_cell is not None and med_draw < smallest_cell)
      else "DRAW_NOISE_COMPARABLE")

# Q4 verdict class-conditionality
by_task = {}
for c in col_rows:
    by_task.setdefault((c["level"], c["problem_id"]), {})[c["arm"]] = c
q4 = {"n_tasks": len(by_task), "c1_pass_c2_fail_1e4": 0, "c1_bit_identical": 0,
      "both_pass_1e4": 0, "both_fail_1e4": 0, "c1_pass_1e2": 0, "c2_pass_1e2": 0}
for k, v in by_task.items():
    c1, c2 = v.get("C1_compile"), v.get("C2_tf32off")
    if not c1 or not c2:
        continue
    if c1["verdict_default_1e4"] == "pass" and c2["verdict_default_1e4"] == "fail":
        q4["c1_pass_c2_fail_1e4"] += 1
    if c1["bit_identical_ca"]:
        q4["c1_bit_identical"] += 1
    if c1["verdict_default_1e4"] == "pass" and c2["verdict_default_1e4"] == "pass":
        q4["both_pass_1e4"] += 1
    if c1["verdict_default_1e4"] == "fail" and c2["verdict_default_1e4"] == "fail":
        q4["both_fail_1e4"] += 1
    if c1["verdict_default_1e2"] == "pass":
        q4["c1_pass_1e2"] += 1
    if c2["verdict_default_1e2"] == "pass":
        q4["c2_pass_1e2"] += 1

# Q5 control gate
controls = [r for r in rows if r.get("kind") != "exceed"]
ctl = []
for r in controls:
    d = (r.get("decomp_c1_compile") or {}).get("1e-4") or {}
    frac = (d.get("ref_over_budget") / d["total"]) if d.get("total") else None
    ctl.append({"level": r["level"], "problem_id": r["problem_id"], "ref_over_budget_frac": frac,
                "ok": frac is not None and frac <= CONTROL_BAR})
Q5 = "CONTROL_GATE_PASS" if all(c["ok"] for c in ctl) else "CONTROL_GATE_VIOLATION"

summary = {
    "artifact": "s84_class_column",
    "date": "2026-11-02T15:50:00Z",
    "prereg": "docstring of lab/s84_class_column.py, frozen pre-run",
    "population": {"exceed_rows": 54, "controls": len(controls),
                   "op_class": {k: sum(1 for c in col_rows if c["op_class"] == k and c["arm"] == "C1_compile")
                                for k in sorted({c["op_class"] for c in col_rows})}},
    "Q1_share": {"verdict": Q1, "pooled": pooled, "cells": q1_cells,
                 "bars": {"pooled_median": SHARE_POOLED_BAR, "cell_median": SHARE_CELL_BAR}},
    "Q2_tail": {"verdict": Q2, "cells": q2_cells, "offending": bad, "bar_spread": TAIL_BOUND,
                "min_cell_n": MIN_CELL_N},
    "Q3_draw": {"verdict": Q3, "n_probed_rows": len(probed), "median_draw_spread": med_draw,
                "smallest_cell_tail_spread": smallest_cell,
                "probed_tasks": sorted({"%d/%d" % (c["level"], c["problem_id"]) for c in probed})},
    "Q4_verdict_class": q4,
    "Q5_control": {"verdict": Q5, "rows": ctl},
    "schema": ["level", "problem_id", "op_class", "engine_class_refined", "arm", "arm_relation",
               "verdict_default_1e4", "verdict_anchor_1e4", "verdict_default_1e2",
               "fail_default_1e4", "fail_anchor_1e4", "total", "ref_induced", "cand_induced",
               "ref_share", "cand_share", "frac_over_ref", "max_ratio_ref", "max_ratio_cand",
               "bit_identical_ca", "draw_probe", "draw_spread_max_over_min",
               "draw_frac_over_min", "draw_frac_over_max"],
    "resolution_note": "class counts in s83 were 5/3 at 5% and 4/3 at 10%; any behavioural count is quoted with its resolution",
    "n_column_rows": len(col_rows),
}
json.dump(summary, open(OUTJ, "w"), indent=1, sort_keys=False)
with open(OUTC, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=summary["schema"])
    w.writeheader()
    for c in col_rows:
        w.writerow({k: c.get(k) for k in summary["schema"]})
with open(OUTR, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["engine_class", "op_class", "arm", "n", "ref_share_median", "max_ratio_ref_min",
                "max_ratio_ref_max", "tail_spread", "frac_over_ref_median"])
    for k, v in sorted(cells_tail.items()):
        vals = [x["max_ratio_ref"] for x in v]
        w.writerow([k[0], k[1], "C2_tf32off", len(v),
                    statistics.median([x["ref_share"] for x in v if x["ref_share"] is not None]) if any(x["ref_share"] is not None for x in v) else "",
                    min(vals), max(vals), (max(vals) / min(vals)) if min(vals) > 0 else "",
                    statistics.median([x["frac_over_ref"] for x in v])])

print(json.dumps({k: summary[k] for k in ("Q1_share", "Q2_tail", "Q3_draw", "Q4_verdict_class", "Q5_control")}, indent=1)[:6000])
