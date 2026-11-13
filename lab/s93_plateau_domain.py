#!/usr/bin/env python
"""s93_plateau_domain.py — does the 1.83% population plateau of the reference's shipped-gate budget hold two decades above its measurement?

PREREG (frozen before any run; commit this file before executing on spark; do not edit after).

Question. s92 located the frac_over=0.01 crossing at the shipped 1e-2 gate (T1 g=76.91 |A|max 371.3; T2 g=79.38
|A|max 415.3) and found that frac then saturates at 1.84%/1.83% by g=4096 (|A|max 19,775/21,431) while rho keeps
growing 7.6 -> 327. s92b attributed the plateau to the reference's relative-error tail (evaluator predicate and an
atol=0 predicate agree to 2e-4 at that scale). That attribution rests on ONE scale point per tuple, so it cannot
distinguish "the predicate is relative here" from "the relative tail is scale-invariant". This run measures the
plateau two decades above (g=1024 -> g=1e5) and one decade below (g=16), and measures the relative-error
distribution directly, so the plateau becomes a property with a domain rather than a point.

Why it matters. If the tail constant is scale-invariant and shape-independent, the column can ship one arithmetic
constant for the population statistic (usability) beside the gate-specific extremum coordinate (admission), and
"reference-clean" becomes a statement about a measured relative-error distribution rather than about one sweep.
If the plateau tracks scale, s92b's mechanism sentence is falsified and the column carries a per-scale fraction.

Population (declared before outcome; same two tuples as s91/s92, not selected by result):
  T1 = s90 idx 5: conv2d 3x3 s1 p1, B=2, C=64 -> 64, 151x151, seed 1085, 2,918,528 output elements.
  T2 = s90 idx 4: conv2d 1x1 s1 p0, B=2, C=192 -> 160, 211x211, seed 1068, 14,246,720 output elements.
  T3 = FRESH SHAPE (not from s90): conv2d 3x3 s1 p1, B=1, C=64 -> 64, 128x128, seed 2093, 1,048,576 output
       elements. Included to test whether the tail constant is shape-independent; its engine class is recorded and,
       if it is not TF32_TENSOROP, the T3 comparison is labelled cross-class rather than merged.

Harness: s91's arms imported verbatim (build, conv_fwd, set_cfg, profile_call, classify), same predicate direction
(evaluator-exact, candidate = anchor), same chunk-free element comparison. The distribution statistics below are
added diagnostics on the same tensors; no arm or predicate is changed.

Scale grid (declared): g in {16, 1024, 4096, 1e5} for T1/T2; g in {1024, 1e5} for T3. Weight =
benchmark-realistic normalization (w /= sqrt(fan_in)) times g. g=4096 is s92's top measured point and is the
tie-out control.

Statistics per point (all recorded): frac_over at the shipped predicate (atol=rtol=1e-2), frac_over at an atol=0
predicate (|d| > rtol*|X|), the share of failing elements with |X| <= atol, rho at both predicates, the relative
error distribution |d|/|X| over elements with |X| > 0 (p50/p90/p99/p99.9/p99.99 and P(rel > rtol)), engine class,
the tf32-off (B) arm, and the A-arm repeat bit-identity.

Controls / kill clauses (coded, fired verbatim):
  K_INSTRUMENT_TIEOUT  g=4096 rows must reproduce s92's measured frac_over_1e-2 (0.018420 T1 / 0.018308 T2) within
                       2e-4 absolute and rho_1e-2 (327.441 / 345.845) within 2% relative. Failure => INSTRUMENT_FAILED.
  K_INSTRUMENT_REPEAT  the A-arm repeat must be bit-identical on all points; one failure is tolerated, two or more
                       => INSTRUMENT_FAILED.
  K_FRAC_DECADE        primary kill, per the reflection that commissioned this run: frac_over(1e5) vs frac_over(1024)
                       differing by more than 20% relative at fixed tuple => PLATEAU_POINT (the plateau is a
                       single-scale coincidence; s92b's tail attribution is withdrawn).
  K_TAIL_INVARIANT     relative-error p99 at 1e5 vs 1024 differing by more than 20% relative => the relative tail is
                       not scale-invariant, so the plateau cannot be explained by it.
  K_B_FIRES            tf32-off frac_over >= 0.01 at any point => class attribution withdrawn for that tuple.

Branch clauses (only if no kill fired):
  PLATEAU_TAIL_SCALE_INVARIANT   frac_over ratio <= 1.20 over the two decades AND p99 ratio <= 1.20 AND
                                 |frac_atol0 - frac_over| / frac_over <= 0.05 at g=1e5 on both tuples => the plateau
                                 is the relative-error tail, scale-invariant, and the predicate is relative there.
  MIXED_TAIL                     frac_over stable but p99 moves > 20% => the bulk fraction is stable for a different
                                 reason than the tail; reported as such, no constant claimed.
  PLATEAU_POINT                  frac_over moves > 20% => point, not property.
Cross-tuple clause: if frac_over at g=1e5 agrees within 20% across T1/T2 and T3's class is TF32_TENSOROP and its
frac agrees within 20%, the constant is reported as shape-independent within the TF32 class; otherwise scoped.

Declared limits: GB10 sm_120 only; fp64 anchor is a proxy for exact arithmetic; conv only; the weight scale is the
only varied coordinate; one draw per point (s79a DRAW_STABLE at fixed tuple+scale, not re-tested); quantiles are
computed on the fp32 relative error of the reference against the fp64 anchor, so they inherit the anchor's own
rounding (bounded by s90's measured 2.2e-4 reference-side constant).
"""
import hashlib
import json
import os
import signal
import time

import torch

import s91

LAB = os.path.expanduser("~/kb-lab")
OUT = os.path.join(LAB, "results", "s93_plateau_domain.jsonl")
SUMOUT = os.path.join(LAB, "results", "s93_plateau_domain_summary.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

SHIPPED = s91.SHIPPED
STRICT = s91.STRICT
CFG_ALARM_S = 240
GLOBAL_DEADLINE_S = 1200
QS = [0.5, 0.9, 0.99, 0.999, 0.9999]

TUPLE_DEFS = list(s91.TUPLE_DEFS)
T3 = dict(tid="T3", s90_idx=-1, desc="conv 3x3 s1 128x128 C64->64 B1 (fresh shape)", seed=2093, op="conv",
          args=dict(n=1, cin=64, h=128, w=128, cout=64, k=3, s=1, p=1))
GRID_T12 = [16.0, 1024.0, 4096.0, 100000.0]
GRID_T3 = [1024.0, 100000.0]
TIE = {"T1": dict(frac=0.018420, rho=327.441), "T2": dict(frac=0.018308, rho=345.845)}


def append(rec):
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")


def run_point_ext(td, g, kind):
    rec = dict(tid=td["tid"], s90_idx=td["s90_idx"], desc=td["desc"], seed=td["seed"], g=g, kind=kind,
               status="ok")
    t0 = time.time()
    signal.alarm(CFG_ALARM_S)
    try:
        s91.set_cfg(True, False)
        torch.manual_seed(td["seed"])
        x, w = s91.build(td, torch.float32)
        fan_in = w.numel() // w.shape[0]
        w.div_(fan_in ** 0.5)
        w.mul_(g)
        rec["fan_in"] = fan_in
        a1 = s91.conv_fwd(x, w, td["args"])
        prof = s91.profile_call(lambda: s91.conv_fwd(x, w, td["args"]))
        a2 = s91.conv_fwd(x, w, td["args"])
        rec["a_repeat_bit_identical"] = bool(torch.equal(a1, a2))
        rec["engine_top5"] = [(n, round(t, 2)) for n, t in prof]
        cls, top1 = s91.classify([n for n, _ in prof])
        rec["engine_class"], rec["engine_top1"] = cls, top1
        a = a1
        del a2

        s91.set_cfg(False, False)
        b = s91.conv_fwd(x, w, td["args"])
        xa = s91.conv_fwd(x.double(), w.double(), td["args"])
        del x, w
        if a.shape != xa.shape:
            raise RuntimeError("shape mismatch")

        af = a.detach().reshape(-1)
        xf = xa.detach().reshape(-1)
        bf = b.detach().reshape(-1)
        n = int(af.numel())
        d = (af - xf).abs()
        ax = xf.abs()
        bud = SHIPPED + SHIPPED * ax
        ratio = d / bud
        fail = ratio > 1
        nf = int(fail.sum().item())
        rec["n_elems"] = n
        rec["maxabs_out"] = float(af.abs().max().item())
        rec["A_1e-2"] = dict(n_fail=nf, frac_over=nf / n, rho=float(ratio.max().item()),
                             argmax=dict(ratio=float(ratio.max().item())))
        fail0 = d > SHIPPED * ax
        rec["frac_atol0"] = float(int(fail0.sum().item())) / n
        rec["rho_atol0"] = float((d / (SHIPPED * ax)).max().item())
        small = ax <= SHIPPED
        rec["atol_floor_share"] = float(int((fail & small).sum().item())) / max(1, nf)
        mask = ax > 0
        rel = d[mask] / ax[mask]
        rec["rel_quantiles"] = {str(q): float(torch.quantile(rel, q).item()) for q in QS}
        rec["rel_frac_gt_rtol"] = float(int((rel > SHIPPED).sum().item())) / max(1, int(rel.numel()))
        rec["rel_n"] = int(rel.numel())

        db = (bf - xf).abs()
        rb = db / bud
        nfb = int((rb > 1).sum().item())
        rec["B_1e-2"] = dict(n_fail=nfb, frac_over=nfb / n, rho=float(rb.max().item()))

        # argmax element diagnostics (s91's cancellation statement, cheap to carry)
        k = int(ratio.argmax().item())
        rec["A_1e-2"]["argmax"].update(a=float(af[k].item()), x=float(xf[k].item()), d=float(d[k].item()))

        rec["u_1e-2"] = s91.REF_REL * rec["maxabs_out"] / SHIPPED
        rec["t_s"] = round(time.time() - t0, 2)
        del a, b, xa, d, ax, bud, ratio, fail, fail0, small, rel, mask, db, rb
        torch.cuda.empty_cache()
    except s91.Timeout:
        rec["status"] = "timeout"
    except torch.cuda.OutOfMemoryError as e:
        rec["status"] = "oom"
        rec["err"] = str(e)[:200]
        torch.cuda.empty_cache()
    except Exception as e:
        rec["status"] = "error"
        rec["err"] = repr(e)[:300]
        torch.cuda.empty_cache()
    finally:
        signal.alarm(0)
        rec.setdefault("t_s", round(time.time() - t0, 2))
    return rec


def main():
    torch.cuda.init()
    t_start = time.time()
    rows = []
    open(OUT, "w").close()
    sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    print("prereg sha", sha[:12], flush=True)

    plan = []
    for td in TUPLE_DEFS:
        for g in GRID_T12:
            plan.append((td, g))
    for g in GRID_T3:
        plan.append((T3, g))

    for td, g in plan:
        if time.time() - t_start > GLOBAL_DEADLINE_S:
            print("GLOBAL DEADLINE at", td["tid"], g, flush=True)
            break
        r = run_point_ext(td, g, "grid")
        rows.append(r)
        append(r)
        print("grid", td["tid"], "g", g, r["status"], "class", r.get("engine_class"),
              "maxabs", round(r.get("maxabs_out", 0), 2),
              "frac", round(r.get("A_1e-2", {}).get("frac_over", 0), 6),
              "frac0", round(r.get("frac_atol0", 0), 6),
              "rho", round(r.get("A_1e-2", {}).get("rho", 0), 2),
              "p99", r.get("rel_quantiles", {}).get("0.99"),
              "Bfrac", round(r.get("B_1e-2", {}).get("frac_over", 0), 6),
              "rep", r.get("a_repeat_bit_identical"), "t", r.get("t_s"), flush=True)

    ok = [r for r in rows if r.get("status") == "ok"]
    summary = dict(prereg_sha=sha, n_attempted=len(rows), n_ok=len(ok), grid_t12=GRID_T12, grid_t3=GRID_T3,
                   tuples=[td["tid"] for td in TUPLE_DEFS] + ["T3"])

    # controls
    tie = {}
    for tid, ref in TIE.items():
        cand = [r for r in ok if r["tid"] == tid and r["g"] == 4096.0]
        if not cand:
            tie[tid] = dict(status="missing", pass_=False)
            continue
        r = cand[0]
        df = abs(r["A_1e-2"]["frac_over"] - ref["frac"])
        dr = abs(r["A_1e-2"]["rho"] - ref["rho"]) / ref["rho"]
        tie[tid] = dict(frac_got=r["A_1e-2"]["frac_over"], frac_s92=ref["frac"], frac_abs_delta=df,
                        rho_got=r["A_1e-2"]["rho"], rho_s92=ref["rho"], rho_rel=dr,
                        pass_=bool(df <= 2e-4 and dr <= 0.02))
    summary["tieout"] = tie
    tie_ok = all(v.get("pass_") for v in tie.values())
    n_rep_fail = sum(1 for r in ok if not r.get("a_repeat_bit_identical"))
    summary["n_repeat_fail"] = n_rep_fail
    summary["repeat_bit_identical_frac"] = (len(ok) - n_rep_fail) / len(ok) if ok else 0.0
    instr_ok = tie_ok and n_rep_fail <= 1 and len(ok) >= 0.9 * len(rows)
    summary["instrument_ok"] = instr_ok

    def pick(tid, g):
        c = [r for r in ok if r["tid"] == tid and r["g"] == g]
        return c[0] if c else None

    # per-tuple decade test
    decade = {}
    for tid in ["T1", "T2", "T3"]:
        lo = pick(tid, 1024.0)
        hi = pick(tid, 100000.0)
        if not (lo and hi):
            decade[tid] = dict(status="missing")
            continue
        f_lo, f_hi = lo["A_1e-2"]["frac_over"], hi["A_1e-2"]["frac_over"]
        q_lo = lo["rel_quantiles"]["0.99"]
        q_hi = hi["rel_quantiles"]["0.99"]
        decade[tid] = dict(frac_1024=f_lo, frac_1e5=f_hi, frac_rel_change=(f_hi - f_lo) / max(f_lo, 1e-12),
                           p99_1024=q_lo, p99_1e5=q_hi, p99_rel_change=(q_hi - q_lo) / max(q_lo, 1e-12),
                           frac_atol0_1e5=hi["frac_atol0"],
                           atol0_vs_eval_rel=abs(hi["frac_atol0"] - f_hi) / max(f_hi, 1e-12),
                           atol_floor_share_1e5=hi["atol_floor_share"],
                           engine_class_1024=lo["engine_class"], engine_class_1e5=hi["engine_class"],
                           u_1e5=hi.get("u_1e-2"),
                           B_frac_1e5=hi["B_1e-2"]["frac_over"], B_rho_1e5=hi["B_1e-2"]["rho"])
    summary["decade"] = decade

    # kill clauses
    kills = []
    for tid, d in decade.items():
        if d.get("status") == "missing":
            continue
        if abs(d["frac_rel_change"]) > 0.20:
            kills.append("K_FRAC_DECADE:" + tid)
        if abs(d["p99_rel_change"]) > 0.20:
            kills.append("K_TAIL_INVARIANT:" + tid)
    b_fires = [tid for tid, d in decade.items() if d.get("status") != "missing" and (d["B_frac_1e5"] or 0) >= 0.01]
    summary["b_arm_fires"] = b_fires
    summary["kill_fired"] = kills

    if not instr_ok:
        verdict = "INSTRUMENT_FAILED"
    elif any(k.startswith("K_FRAC_DECADE") for k in kills):
        verdict = "PLATEAU_POINT"
    elif any(k.startswith("K_TAIL_INVARIANT") for k in kills):
        verdict = "MIXED_TAIL"
    else:
        tail_ok = all(abs(d.get("atol0_vs_eval_rel", 1)) <= 0.05 for d in decade.values() if d.get("status") != "missing")
        verdict = "PLATEAU_TAIL_SCALE_INVARIANT" if tail_ok else "MIXED_TAIL"

    # shape-independence clause
    hi_fracs = {tid: d.get("frac_1e5") for tid, d in decade.items() if d.get("status") != "missing"}
    if hi_fracs:
        vals = [v for v in hi_fracs.values() if v]
        ratio = max(vals) / min(vals) if len(vals) > 1 else None
        classes = {tid: d.get("engine_class_1e5") for tid, d in decade.items() if d.get("status") != "missing"}
        tf32_all = all(c == "TF32_TENSOROP" for c in classes.values())
        summary["shape_independence"] = dict(frac_1e5=hi_fracs, cross_tuple_ratio=ratio, classes=classes,
                                             tf32_all=tf32_all)
        if ratio is not None and ratio <= 1.20 and tf32_all:
            verdict += "+SHAPE_INDEPENDENT_TF32"
        else:
            verdict += "+SHAPE_SCOPED"
    if b_fires:
        verdict += "+B_FIRES"
    summary["verdict"] = verdict

    with open(SUMOUT, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1)[:6000], flush=True)


if __name__ == "__main__":
    main()
