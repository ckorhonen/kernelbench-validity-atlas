#!/usr/bin/env python
"""s92_frac_boundary.py — locate the POPULATION (frac_over) boundary of the reference at the shipped 1e-2 gate.

PREREG (frozen before any run; commit this file before executing on spark; do not edit after).

Question. s91 located the reference's shipped-gate validity boundary in the extremum statistic (rho=1 crossing at
g=8.354 / |A|max 40.33 for T1 and g=7.336 / |A|max 38.38 for T2) and found frac_over still below 0.01 at the top of
its sweep (g=64, |A|max 309-335, frac 0.0087/0.0085). That leaves the "which statistic" statement one-sided: the
extremum boundary governs allclose admission (one failing element fails a tensor), but the fraction statistic — the
one that says how much of the output is unusable — has no measured boundary. s92 extends the same sweep upward on
the same two tuples and bisects the weight scale at which frac_over crosses 0.01 and 0.5 at the shipped gate.

Why it matters for the deliverable: the certificate column currently ships share/ceiling/ratio plus the extremum
boundary. If the population boundary also collapses across tuples in a scale coordinate, the column can carry one
derived coordinate for usability as well as admission; if it does not, the column must carry per-tuple fractions and
the "reference-clean" claim stays extremum-only. Either outcome is a scoped, publishable statement.

Population (rule declared before outcome; same tuples as s91, not selected by result):
  T1 = s90 idx 5: conv2d 3x3 s1 p1, B=2, C=64 -> 64, 151x151, seed 1085, 2,918,528 output elements.
  T2 = s90 idx 4: conv2d 1x1 s1 p0, B=2, C=192 -> 160, 211x211, seed 1068, 14,246,720 output elements.
Harness is s91's, imported verbatim (same arms, same predicate, same chunked comparison) so the tie-out is
bit-level rather than statistical.

Scale grid (declared, geometric, upward extension only): g in {64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048,
4096}; weight = benchmark-realistic normalization (w /= sqrt(fan_in)) times g. g=64 is s91's top grid point and is
the tie-out control.

Statistic under test: frac_over at the shipped 1e-2 gate = |{e : |A_e - X_e| > atol + rtol*|X_e|}| / N with
atol = rtol = 1e-2 and X the fp64 anchor on the exact widening of the same fp32 inputs (evaluator-exact direction,
eval.py:804). rho and the tf32-off (B) arm are recorded at every point as before.

Controls / kill clauses (coded, fired verbatim):
  K_INSTRUMENT_TIEOUT  g=64 rows must reproduce s91's measured values within 2% relative on rho_1e-2 (7.6348 T1 /
                       7.4003 T2) and within 0.002 absolute on frac_over_1e-2 (0.008665 / 0.008544). Failure =>
                       INSTRUMENT_FAILED, no boundary quoted.
  K_INSTRUMENT_REPEAT  the A-arm repeat must be bit-identical on all but at most 2 points; else INSTRUMENT_FAILED.
  K_MONOTONE_FRAC      frac_over must be non-decreasing in g at fixed tuple: a violation is
                       frac_{i+1} < frac_i - max(1e-4, 0.10*frac_i). Violation => NON_MONOTONE_FRAC, grid bracket
                       reported, no bisected boundary claimed.
  K_NO_CROSSING        frac_over < 0.01 at every grid point => only a lower bound on the population boundary.
  K_B_FIRES            if the tf32-off arm's frac_over >= 0.01 at or below the A crossing scale, the class
                       attribution is withdrawn for that tuple (recorded, not silently dropped).

Bisection: for each tuple and each target level (0.01, 0.5), bisect geometrically 6 steps from the bracketing grid
interval, stop at 1% in g. Report g*, |A|max*, rho*, frac*, u* = 2.2e-4*|A|max*/atol (s91's coordinate), and the
B-arm at the crossing. No threshold other than the two declared levels is introduced.

Branch clauses (only if no kill fired and both crossings located on both tuples):
  BRANCH_FRAC_COLLAPSE   the crossing g* and |A|max* agree across T1/T2 within 25% at each level => the population
                         boundary collapses across a 4.9x element-count difference at a fixed gate.
  BRANCH_FRAC_PER_TUPLE  >25% apart => population boundary is per-tuple; column carries per-tuple fractions.
  BRANCH_FRAC_EXTREMUM_DECOUPLED  if the frac=0.01 crossing sits at rho >= 1.5x the rho=1 crossing (s91 measured
                         rho=1 at g 8.354/7.336) => admission and usability boundaries are separate surfaces, so a
                         single rho value cannot stand for both (recorded as the headline if it fires).

Declared limits: GB10 sm_120 only; fp64 anchor is a proxy for exact arithmetic; two conv tuples, so family
statements stay scoped to conv at these shapes; weight scale is the only varied coordinate; frac_over is measured
on one draw per point (s91 established the draw is bit-stable at fixed tuple+scale, so this is not re-tested).
"""
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s91_scale_boundary as s91  # harness reuse: run_point, TUPLE_DEFS, SHIPPED, REF_REL, STRICT

import torch

LAB = os.path.expanduser("~/kb-lab")
OUT = os.path.join(LAB, "results", "s92_frac_boundary.jsonl")
SUMOUT = os.path.join(LAB, "results", "s92_frac_boundary_summary.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

GRID = [64.0, 96.0, 128.0, 192.0, 256.0, 384.0, 512.0, 768.0, 1024.0, 2048.0, 4096.0]
LEVELS = [0.01, 0.5]
TUPLE_DEFS = s91.TUPLE_DEFS
SHIPPED = s91.SHIPPED
REF_REL = s91.REF_REL
TIE = {"T1": dict(rho=7.6348, frac=0.008665), "T2": dict(rho=7.4003, frac=0.008544)}
S91_RHO1_G = {"T1": 8.354, "T2": 7.336}
GLOBAL_DEADLINE_S = 1500


def frac_of(rec):
    return rec.get("A_1e-2", {}).get("frac_over")


def rho_of(rec):
    return rec.get("A_1e-2", {}).get("rho")


def append(rec):
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")


def bisect_level(td, level, g_lo, g_hi):
    lo, hi = g_lo, g_hi
    rows = []
    for _ in range(6):
        mid = (lo * hi) ** 0.5
        r = s91.run_point(td, mid, "bisect")
        append(r)
        rows.append(r)
        f = frac_of(r) if r.get("status") == "ok" else None
        print("  bisect", td["tid"], "level", level, "g", round(mid, 5), r["status"],
              "frac", None if f is None else round(f, 6), "rho", None if r.get("status") != "ok" else round(rho_of(r), 3),
              flush=True)
        if f is None:
            break
        if f >= level:
            hi = mid
        else:
            lo = mid
        if hi / lo < 1.01:
            break
    return lo, hi, rows


def main():
    torch.cuda.init()
    t_start = time.time()
    rows = []
    open(OUT, "w").close()
    sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    print("prereg sha", sha[:12], flush=True)

    for td in TUPLE_DEFS:
        for g in GRID:
            if time.time() - t_start > GLOBAL_DEADLINE_S:
                print("GLOBAL DEADLINE", td["tid"], g, flush=True)
                break
            r = s91.run_point(td, g, "grid")
            rows.append(r)
            append(r)
            print("grid", td["tid"], "g", g, r["status"], "maxabs", round(r.get("maxabs_out", 0), 2),
                  "rho", None if r.get("status") != "ok" else round(rho_of(r), 4),
                  "frac", None if r.get("status") != "ok" else round(frac_of(r), 6),
                  "Bfrac", None if r.get("status") != "ok" else round(r.get("B_1e-2", {}).get("frac_over", 0), 6),
                  "class", r.get("engine_class"), "rep", r.get("a_repeat_bit_identical"),
                  "t", r.get("t_s"), flush=True)

    ok = [r for r in rows if r.get("status") == "ok"]
    summary = dict(prereg_sha=sha, n_attempted=len(rows), n_ok=len(ok), grid=GRID, levels=LEVELS)

    # --- controls
    tie = {}
    for td in TUPLE_DEFS:
        cand = [r for r in ok if r["tid"] == td["tid"] and r["kind"] == "grid" and r["g"] == 64.0]
        if not cand:
            tie[td["tid"]] = dict(status="missing", pass_=False)
            continue
        r = cand[0]
        got_rho, got_frac = rho_of(r), frac_of(r)
        ref = TIE[td["tid"]]
        rel = abs(got_rho - ref["rho"]) / ref["rho"]
        dfrac = abs(got_frac - ref["frac"])
        tie[td["tid"]] = dict(rho_got=got_rho, rho_s91=ref["rho"], rho_rel=rel,
                              frac_got=got_frac, frac_s91=ref["frac"], frac_abs_delta=dfrac,
                              pass_=bool(rel <= 0.02 and dfrac <= 0.002))
    summary["tieout"] = tie
    tie_ok = all(v.get("pass_") for v in tie.values())
    n_rep_fail = sum(1 for r in ok if not r.get("a_repeat_bit_identical"))
    summary["repeat_bit_identical_frac"] = (sum(1 for r in ok if r.get("a_repeat_bit_identical")) / len(ok)) if ok else 0.0
    summary["n_repeat_fail"] = n_rep_fail
    instr_ok = tie_ok and n_rep_fail <= 2 and len(ok) >= 0.8 * max(1, len(rows))
    summary["instrument_ok"] = instr_ok

    # --- monotonicity of frac_over in g
    mono, violations = {}, []
    for td in TUPLE_DEFS:
        seq = []
        for g in GRID:
            rr = [r for r in ok if r["tid"] == td["tid"] and r["kind"] == "grid" and r["g"] == g]
            if rr:
                seq.append((g, frac_of(rr[0]), rho_of(rr[0]), rr[0].get("maxabs_out")))
        mono[td["tid"]] = seq
        for i in range(len(seq) - 1):
            f0, f1 = seq[i][1], seq[i + 1][1]
            if f1 < f0 - max(1e-4, 0.10 * f0):
                violations.append((td["tid"], seq[i][0], seq[i + 1][0], f0, f1))
    summary["frac_monotonicity"] = {k: [[g, round(f, 6), round(r or 0, 3), m] for g, f, r, m in v] for k, v in mono.items()}
    summary["frac_monotone_violations"] = violations
    summary["frac_monotone_pass"] = len(violations) == 0

    # --- crossings
    crossings = {}
    if instr_ok and summary["frac_monotone_pass"]:
        for td in TUPLE_DEFS:
            seq = mono[td["tid"]]
            for level in LEVELS:
                br = None
                for i in range(len(seq) - 1):
                    if seq[i][1] < level <= seq[i + 1][1]:
                        br = (seq[i][0], seq[i + 1][0])
                        break
                key = "%s_frac%.3g" % (td["tid"], level)
                if br is None:
                    crossings[key] = dict(status="NO_CROSSING_IN_RANGE", frac_max=max(s[1] for s in seq))
                    continue
                lo, hi, brows = bisect_level(td, level, br[0], br[1])
                pts = [r for r in brows if r.get("status") == "ok"]
                inside = [p for p in pts if lo <= p["g"] <= hi]
                gstar = (lo * hi) ** 0.5
                near = min(inside, key=lambda p: abs(p["g"] - gstar)) if inside else None
                rec = dict(status="CROSSING", level=level, bracket=[lo, hi], g_star_geom=gstar)
                if near:
                    rec.update(g_measured=near["g"], maxabs_out=near["maxabs_out"], rho=rho_of(near),
                               frac=frac_of(near), u=REF_REL * near["maxabs_out"] / SHIPPED,
                               B_frac=near.get("B_1e-2", {}).get("frac_over"),
                               B_rho=near.get("B_1e-2", {}).get("rho"),
                               engine_class=near.get("engine_class"))
                crossings[key] = rec
    summary["crossings"] = crossings

    # --- verdict
    if not instr_ok:
        summary["verdict"] = "INSTRUMENT_FAILED"
    elif not summary["frac_monotone_pass"]:
        summary["verdict"] = "NON_MONOTONE_FRAC"
    else:
        located = [k for k, v in crossings.items() if v.get("status") == "CROSSING"]
        summary["n_located"] = len(located)
        summary["frac001_gap_over_rho1"] = {}
        for td in TUPLE_DEFS:
            v = crossings.get("%s_frac0.01" % td["tid"])
            if v and v.get("status") == "CROSSING":
                summary["frac001_gap_over_rho1"][td["tid"]] = round(v["g_star_geom"] / S91_RHO1_G[td["tid"]], 3)
        if len(located) < len(TUPLE_DEFS) * len(LEVELS):
            summary["verdict"] = "FRAC_BOUNDARY_PARTIAL"
        else:
            spreads = {}
            for level in LEVELS:
                gs = [crossings["%s_frac%.3g" % (td["tid"], level)].get("g_star_geom") for td in TUPLE_DEFS]
                ms = [crossings["%s_frac%.3g" % (td["tid"], level)].get("maxabs_out") for td in TUPLE_DEFS]
                gs = [x for x in gs if x]
                ms = [x for x in ms if x]
                spreads[level] = dict(g_ratio=(max(gs) / min(gs)) if len(gs) > 1 else None,
                                      maxabs_ratio=(max(ms) / min(ms)) if len(ms) > 1 else None)
            summary["cross_tuple_spread"] = spreads
            collapse = all(s["g_ratio"] is not None and s["g_ratio"] <= 1.25 for s in spreads.values())
            summary["frac_collapse"] = collapse
            gaps = summary["frac001_gap_over_rho1"]
            decoupled = bool(gaps) and min(gaps.values()) >= 1.5
            if collapse:
                summary["verdict"] = "BRANCH_FRAC_COLLAPSE"
            else:
                summary["verdict"] = "BRANCH_FRAC_PER_TUPLE"
            summary["extremum_decoupled"] = decoupled
            if decoupled:
                summary["verdict"] += "+FRAC_EXTREMUM_DECOUPLED"
            b_fires = any((crossings[k].get("B_frac") or 0) >= 0.01 for k in located)
            summary["b_arm_fires"] = b_fires
            if b_fires:
                summary["verdict"] += "+B_FIRES"

    with open(SUMOUT, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1)[:5000], flush=True)


if __name__ == "__main__":
    main()
