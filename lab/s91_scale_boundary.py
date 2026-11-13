#!/usr/bin/env python
"""s91_scale_boundary.py — bisect the reference's shipped-gate validity boundary in the |A|max coordinate.

PREREG (frozen before any run; commit this file before executing on spark; do not edit after).

Question. s90 measured the reference-configuration certificate on 24 fresh configs and found that the shipped
1e-2 gate is not exceeded at benchmark-realistic weight scaling (max rho_1e-2 0.174) but IS exceeded on 11/24
configs when the same shapes/seeds use raw randn weights (max rho 7.49). That is a two-point comparison, so the
scale coordinate is an observation, not a boundary. This run turns it into a boundary with a validity domain:
for one fixed conv tuple, sweep the weight scale and bisect the |A|max at which rho = max_e |A_e - X_e| /
(atol + rtol*|X_e|) crosses 1.0, at BOTH gates the benchmark can be run at (shipped 1e-2 and strict 1e-4).

Monotonicity first (IDENTITY working rule: bisection only after a monotonicity check). If rho is not monotone in
the sweep, no bisection is claimed and the boundary is reported as a grid bracket.

Population (rule, declared before outcome; tuples are members of s90's population, not selected by result):
  T1 = s90 idx 5: conv2d 3x3 s1 p1, B=2, C=64 -> 64, 151x151, seed 1085, 2,918,528 output elements.
  T2 = s90 idx 4: conv2d 1x1 s1 p0, B=2, C=192 -> 160, 211x211, seed 1068, 14,246,720 output elements.
Both are TF32_TENSOROP-labelled at the benchmark-realistic scale (s90 recorded class for both). T2 is included
with 4.9x T1's element count because rho is an extremum statistic and a boundary that moves with element count
would not be a shape-independent derived coordinate.

Scale sweep (declared grid, geometric): g in {0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64}, where the
weight tensor is the benchmark-realistic normalization (w /= sqrt(fan_in)) multiplied by g, so output magnitude
scales with g. g=1 is s90's norm-arm point and is the tie-out control.

Arms per point, same instantiation and same inputs:
  A  default fp32 reference realization (cudnn allow_tf32 True, matmul allow_tf32 False) — run twice for the
     repeat control.
  B  tf32 fully off — the class control: if the crossing is TF32-class, B must stay clean at the crossing scale.
  X  fp64 anchor on the EXACT WIDENING of the same fp32 inputs (t.double()), element-chunked comparison
     (CHUNK_ELEMS = 2^22, sequential release).

Predicate: evaluator-exact direction, budget b_e = atol + rtol*|X_e| (candidate = anchor here, eval.py:804
direction). rho_gate = max_e ratio, frac_over_gate = |{ratio>1}|/N. Both gates are computed from the same rows.
Additional diagnostic recorded per point: at the argmax element, |X_e|, |A_e|, |delta_e|, and whether
|X_e| <= atol (the atol-floor mechanism claim, declared descriptive, not a branch).

Bisection: for each tuple and each gate, if the grid brackets a crossing (rho<1 at g_lo, rho>1 at g_hi), bisect
geometrically for 5 steps with stop tolerance 2% in g. Report g*, maxabs_out*, u* = 2.2e-4*maxabs_out*/atol
(the declared atol-floor coordinate; 2.2e-4 is s38's measured reference-side relative error constant), rho*,
frac_over*, and the B-arm rho at the bracket endpoints. No threshold other than rho=1 is introduced.

Controls / kill clauses (coded, fired verbatim):
  K_INSTRUMENT_TIEOUT  g=1 rows must reproduce s90's norm-arm rho_1e-2 for idx 5 (0.1395) and idx 4 (0.1473)
                       within 2% relative. Failure => INSTRUMENT_FAILED, no boundary quoted.
  K_INSTRUMENT_REPEAT  A-arm repeat must be bit-identical on all but at most 2 points; else INSTRUMENT_FAILED
                       (the reference draw is not stable at fixed tuple+scale).
  K_MONOTONE           a monotonicity violation is rho_{i+1} < 0.90*rho_i for adjacent grid points at fixed
                       tuple+gate. One or more violations => NON_MONOTONE_GRID: report the grid, the offending
                       pairs and a local slope, claim NO bisected boundary.
  K_NO_CROSSING        a tuple+gate with rho<1 at every grid point up to g=64 => no crossing in range; report the
                       lower bound only.
Branch clauses (only evaluated if no kill fired):
  BRANCH_COLLAPSE_ATOL  u*(1e-2) and u*(1e-4) agree within 25% AND the two tuples' u* agree within 25% at the
                        same gate => the boundary collapses in the atol-floor coordinate (shape- and
                        element-count-independent within this family at this stack).
  BRANCH_GATE_ONLY      u* agrees across tuples within 25% at each gate but not across gates => the boundary is
                        gate-specific (a constant per tolerance, not one coordinate).
  BRANCH_PER_TUPLE      u* differs across tuples by >25% => the boundary is per-tuple; no derived coordinate.
  BRANCH_B_FIRES        B (tf32-off) rho >= 1 at or below the A crossing scale => the crossing is not
                        TF32-class; recorded and the class attribution is withdrawn for this tuple.

Declared limits: GB10 sm_120 only; fp64 anchor is a proxy for exact arithmetic; two conv tuples only, so any
"family" statement is scoped to conv at these shapes; the weight scale is the only varied coordinate (shape,
layout, batch fixed); rho is an extremum over one draw per point (the draw axis is s79a DRAW_STABLE, not
re-tested here).
"""
import json, os, re, signal, sys, time, hashlib
import torch

LAB = os.path.expanduser("~/kb-lab")
OUT = os.path.join(LAB, "results", "s91_scale_boundary.jsonl")
SUMOUT = os.path.join(LAB, "results", "s91_scale_boundary_summary.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

CFG_ALARM_S = 240
GLOBAL_DEADLINE_S = 1200
CHUNK_ELEMS = 1 << 22
STRICT = 1e-4
SHIPPED = 1e-2
REF_REL = 2.2e-4

CLASS_PATTERNS = [
    ("WINOGRAD", re.compile(r"winograd", re.I)),
    ("TF32_TENSOROP", re.compile(r"tf32|tensorop|1688|16816|h1688", re.I)),
    ("FP64", re.compile(r"dgrad|dgemm|dconv|fp64|double|_d_", re.I)),
    ("FP32_SIMT", re.compile(r"fprop|implicit_gemm|sgemm|s1688|fp32|scudnn|simt", re.I)),
    ("CUTLASS", re.compile(r"cutlass", re.I)),
]

GRID = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]

TUPLE_DEFS = [
    dict(tid="T1", s90_idx=5, desc="conv 3x3 s1 151x151 C64->64 B2", seed=1085, op="conv",
         args=dict(n=2, cin=64, h=151, w=151, cout=64, k=3, s=1, p=1)),
    dict(tid="T2", s90_idx=4, desc="conv 1x1 211x211 C192->160 B2", seed=1068, op="conv",
         args=dict(n=2, cin=192, h=211, w=211, cout=160, k=1, s=1, p=0)),
]


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout("config exceeded %ds" % CFG_ALARM_S)


signal.signal(signal.SIGALRM, _alarm)


def classify(names):
    for lab, pat in CLASS_PATTERNS:
        for n in names:
            if pat.search(n):
                return lab, n
    return ("OTHER", names[0] if names else None)


def profile_call(fn):
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    evts = []
    for e in prof.key_averages():
        if e.self_device_time_total > 0:
            evts.append((e.key, float(e.self_device_time_total)))
    evts.sort(key=lambda t: -t[1])
    return evts[:5]


def set_cfg(tf32_cudnn, tf32_matmul):
    torch.backends.cudnn.allow_tf32 = tf32_cudnn
    torch.backends.cuda.matmul.allow_tf32 = tf32_matmul


def build(td, dt):
    a = td["args"]
    x = torch.randn(a["n"], a["cin"], a["h"], a["w"], device="cuda", dtype=dt)
    w = torch.randn(a["cout"], a["cin"], a["k"], a["k"], device="cuda", dtype=dt)
    return x, w


def conv_fwd(x, w, a):
    return torch.nn.functional.conv2d(x, w, stride=a["s"], padding=a["p"])


def chunked_compare(a, x, atol, rtol):
    """Element-based chunked evaluator-exact comparison with argmax diagnostics."""
    n_fail = 0
    max_ratio = 0.0
    n_tot = 0
    best = dict(ratio=-1.0, a=0.0, x=0.0, d=0.0)
    af = a.detach().reshape(-1)
    xf = x.detach().reshape(-1)
    n = af.numel()
    i = 0
    while i < n:
        j = min(n, i + CHUNK_ELEMS)
        d = (af[i:j] - xf[i:j]).abs()
        b = atol + rtol * xf[i:j].abs()      # evaluator-exact: rtol scales by the candidate (anchor)
        r = d / b
        n_fail += int((r > 1).sum().item())
        mr, am = torch.max(r, 0)
        mr = float(mr.item())
        if mr > max_ratio:
            max_ratio = mr
            k = int(am.item())
            best = dict(ratio=mr, a=float(af[i + k].item()), x=float(xf[i + k].item()),
                        d=float(d[k].item()))
        n_tot += (j - i)
        del d, b, r
        i = j
    return n_fail, max_ratio, n_tot, best


def run_point(td, g, kind):
    rec = dict(tid=td["tid"], s90_idx=td["s90_idx"], desc=td["desc"], seed=td["seed"], g=g, kind=kind,
               status="ok")
    t0 = time.time()
    signal.alarm(CFG_ALARM_S)
    try:
        set_cfg(True, False)
        torch.manual_seed(td["seed"])
        x, w = build(td, torch.float32)
        fan_in = w.numel() // w.shape[0]
        w.div_(fan_in ** 0.5)
        if g != 1.0:
            w.mul_(g)
        rec["fan_in"] = fan_in
        a1 = conv_fwd(x, w, td["args"])
        prof = profile_call(lambda: conv_fwd(x, w, td["args"]))
        a2 = conv_fwd(x, w, td["args"])
        rec["a_repeat_bit_identical"] = bool(torch.equal(a1, a2))
        rec["engine_top5"] = [(n, round(t, 2)) for n, t in prof]
        cls, top1 = classify([n for n, _ in prof])
        rec["engine_class"], rec["engine_top1"] = cls, top1
        a = a1
        del a2

        set_cfg(False, False)
        b = conv_fwd(x, w, td["args"])
        xa = conv_fwd(x.double(), w.double(), td["args"])
        del x, w

        if a.shape != xa.shape:
            raise RuntimeError("shape mismatch")

        for tag, tol in (("1e-4", STRICT), ("1e-2", SHIPPED)):
            nf_a, mr_a, nt, diag = chunked_compare(a, xa, tol, tol)
            nf_b, mr_b, _, _ = chunked_compare(b, xa, tol, tol)
            rec["A_" + tag] = dict(n_fail=nf_a, rho=mr_a, frac_over=nf_a / nt, n_elems=nt,
                                   argmax=diag, atol_floor=bool(abs(diag["x"]) <= tol))
            rec["B_" + tag] = dict(n_fail=nf_b, rho=mr_b, frac_over=nf_b / nt)
        rec["maxabs_out"] = float(a.abs().max().item())
        rec["n_elems"] = int(a.numel())
        rec["u_1e-4"] = REF_REL * rec["maxabs_out"] / STRICT
        rec["u_1e-2"] = REF_REL * rec["maxabs_out"] / SHIPPED
        rec["t_s"] = round(time.time() - t0, 2)
        del a, b, xa
        torch.cuda.empty_cache()
    except Timeout:
        rec["status"] = "timeout"
    except torch.cuda.OutOfMemoryError as e:
        rec["status"] = "oom"; rec["err"] = str(e)[:200]; torch.cuda.empty_cache()
    except Exception as e:
        rec["status"] = "error"; rec["err"] = repr(e)[:300]; torch.cuda.empty_cache()
    finally:
        signal.alarm(0)
        rec.setdefault("t_s", round(time.time() - t0, 2))
    return rec


def rho_of(rec, gate):
    return rec.get("A_" + gate, {}).get("rho")


def bisect(td, gate, g_lo, g_hi, rows):
    lo, hi = g_lo, g_hi
    for step in range(5):
        mid = (lo * hi) ** 0.5
        r = run_point(td, mid, "bisect")
        rows.append(r)
        with open(OUT, "a") as f:
            f.write(json.dumps(r) + "\n")
        print("  bisect", td["tid"], gate, "g", round(mid, 5), r["status"],
              "rho", rho_of(r, gate), flush=True)
        if r["status"] != "ok":
            break
        if rho_of(r, gate) >= 1.0:
            hi = mid
        else:
            lo = mid
        if hi / lo < 1.02:
            break
    return lo, hi


def main():
    torch.cuda.init()
    t_start = time.time()
    rows = []
    open(OUT, "w").close()
    sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    print("prereg sha", sha[:12], flush=True)

    # --- monotonicity sweep
    for td in TUPLE_DEFS:
        for g in GRID:
            if time.time() - t_start > GLOBAL_DEADLINE_S:
                print("GLOBAL DEADLINE at", td["tid"], g, flush=True)
                break
            r = run_point(td, g, "grid")
            rows.append(r)
            with open(OUT, "a") as f:
                f.write(json.dumps(r) + "\n")
            print("grid", td["tid"], "g", g, r["status"], "class", r.get("engine_class"),
                  "maxabs", round(r.get("maxabs_out", 0), 3),
                  "rho1e4", round(rho_of(r, "1e-4") or 0, 4), "frac1e4", round(r.get("A_1e-4", {}).get("frac_over", 0), 5),
                  "rho1e2", round(rho_of(r, "1e-2") or 0, 4), "frac1e2", round(r.get("A_1e-2", {}).get("frac_over", 0), 6),
                  "B1e2", round(r.get("B_1e-2", {}).get("rho", 0), 4),
                  "rep", r.get("a_repeat_bit_identical"), "t", r.get("t_s"), flush=True)

    ok = [r for r in rows if r.get("status") == "ok"]
    summary = dict(prereg_sha=sha, n_attempted=len(rows), n_ok=len(ok), grid=GRID)

    # --- controls
    tie = {}
    for td in TUPLE_DEFS:
        g1 = [r for r in ok if r["tid"] == td["tid"] and r["kind"] == "grid" and r["g"] == 1.0]
        s90_ref = 0.1395 if td["tid"] == "T1" else 0.1473
        if not g1:
            tie[td["tid"]] = dict(status="missing")
            continue
        got = rho_of(g1[0], "1e-2")
        rel = abs(got - s90_ref) / s90_ref if got is not None else None
        tie[td["tid"]] = dict(rho_got=got, s90_norm_rho=s90_ref, rel=rel, pass_=bool(rel is not None and rel <= 0.02))
    summary["tieout"] = tie
    tie_ok = all(v.get("pass_") for v in tie.values())
    rep_frac = sum(1 for r in ok if r.get("a_repeat_bit_identical")) / len(ok) if ok else 0.0
    summary["repeat_bit_identical_frac"] = rep_frac
    summary["n_repeat_fail"] = sum(1 for r in ok if not r.get("a_repeat_bit_identical"))
    instr_ok = tie_ok and summary["n_repeat_fail"] <= 2 and len(ok) >= 0.8 * len(rows)
    summary["instrument_ok"] = instr_ok

    # --- monotonicity per tuple/gate
    mono = {}
    violations = []
    for td in TUPLE_DEFS:
        for gate in ("1e-4", "1e-2"):
            seq = []
            for g in GRID:
                rr = [r for r in ok if r["tid"] == td["tid"] and r["kind"] == "grid" and r["g"] == g]
                if rr:
                    seq.append((g, rho_of(rr[0], gate), rr[0].get("maxabs_out")))
            mono[td["tid"] + "_" + gate] = seq
            for i in range(len(seq) - 1):
                if seq[i + 1][1] is not None and seq[i][1] is not None and seq[i + 1][1] < 0.90 * seq[i][1]:
                    violations.append((td["tid"], gate, seq[i][0], seq[i + 1][0], seq[i][1], seq[i + 1][1]))
    summary["monotonicity"] = {k: [[g, (None if r is None else round(r, 4)), m] for g, r, m in v]
                               for k, v in mono.items()}
    summary["monotone_violations"] = violations
    summary["monotone_pass"] = len(violations) == 0

    # --- crossings / bisection
    crossings = {}
    if instr_ok and summary["monotone_pass"]:
        for td in TUPLE_DEFS:
            for gate in ("1e-4", "1e-2"):
                seq = mono[td["tid"] + "_" + gate]
                br = None
                for i in range(len(seq) - 1):
                    if seq[i][1] is not None and seq[i + 1][1] is not None and seq[i][1] < 1.0 <= seq[i + 1][1]:
                        br = (seq[i][0], seq[i + 1][0])
                        break
                if br is None:
                    crossings[td["tid"] + "_" + gate] = dict(status="NO_CROSSING_IN_RANGE",
                                                            rho_max=max((s[1] or 0) for s in seq))
                    continue
                lo, hi = bisect(td, gate, br[0], br[1], rows)
                with open(OUT, "a") as f:
                    for r in rows[-8:]:
                        if r["kind"] == "bisect" and r["tid"] == td["tid"]:
                            f.write(json.dumps(r) + "\n")
                pts = [r for r in rows if r["tid"] == td["tid"] and r["kind"] == "bisect" and r["status"] == "ok"]
                pts.sort(key=lambda r: r["g"])
                inside = [p for p in pts if lo <= p["g"] <= hi]
                gstar = (lo * hi) ** 0.5
                # nearest measured point inside the final bracket, else the bracket midpoint
                near = min(inside, key=lambda p: abs(p["g"] - gstar)) if inside else None
                atol = STRICT if gate == "1e-4" else SHIPPED
                rec = dict(status="CROSSING", bracket=[lo, hi], g_star_geom=gstar,
                           gate=gate, atol=atol)
                if near:
                    rec.update(g_measured=near["g"], maxabs_out=near["maxabs_out"], rho=rho_of(near, gate),
                               frac_over=near["A_" + gate]["frac_over"], u=REF_REL * near["maxabs_out"] / atol,
                               argmax=near["A_" + gate]["argmax"],
                               B_rho=near.get("B_" + gate, {}).get("rho"),
                               B_frac=near.get("B_" + gate, {}).get("frac_over"),
                               engine_class=near.get("engine_class"))
                crossings[td["tid"] + "_" + gate] = rec
    summary["crossings"] = crossings

    # --- verdict
    if not instr_ok:
        summary["verdict"] = "INSTRUMENT_FAILED"
    elif not summary["monotone_pass"]:
        summary["verdict"] = "NON_MONOTONE_GRID"
    else:
        cr = [v for v in crossings.values() if v.get("status") == "CROSSING"]
        summary["n_crossings"] = len(cr)
        if not cr:
            summary["verdict"] = "NO_CROSSING_IN_RANGE"
        else:
            u_by_gate = {}
            for v in cr:
                u_by_gate.setdefault(v["gate"], []).append(v["u"])
            b_fires = any((v.get("B_rho") is not None and v["B_rho"] >= 1.0) for v in cr)
            summary["b_arm_fires"] = b_fires
            same_gate = all((max(us) / min(us)) <= 1.25 for us in u_by_gate.values() if len(us) > 1)
            cross_gate = (len(u_by_gate) == 2 and
                          (max(min(us) for us in u_by_gate.values()) /
                           min(max(us) for us in u_by_gate.values())) <= 1.25)
            summary["u_by_gate"] = u_by_gate
            summary["u_same_gate_within25"] = same_gate
            summary["u_cross_gate_within25"] = cross_gate
            if same_gate and cross_gate:
                summary["verdict"] = "BRANCH_COLLAPSE_ATOL"
            elif same_gate:
                summary["verdict"] = "BRANCH_GATE_ONLY"
            else:
                summary["verdict"] = "BRANCH_PER_TUPLE"
            summary["branch_b_note"] = ("tf32-off arm reaches rho>=1 at a crossing scale: class attribution "
                                        "withdrawn for that tuple" if b_fires else
                                        "tf32-off arm stays clean at every crossing scale")

    with open(SUMOUT, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1)[:4000], flush=True)


if __name__ == "__main__":
    main()
