#!/usr/bin/env python
"""s95_draw_repeat.py — is the low-K non-monotonicity of the fp32 reference's rho_ref a draw artifact
or a real low-depth regime? (draw-repeat at fixed tuples, matmul, M=N=1024.)

PREREG (frozen before any run; commit this file before executing on spark; do not edit after).

Question. s94 measured rho_ref = 0.580 / 0.559 / 1.235 / 2.449 at K=1024/2048/4096/8192 with ONE draw
per tuple, giving a log-log slope of 0.738 (residual 0.32) and a monotonicity violation at the bottom
(-3.7% from K=1024 to K=2048). That dip is the only thing standing between the piecewise reading and a
two-regime law: if it is a real, draw-stable floor then eps(K) = max(F, c*K^alpha) is a law to ship; if
it is draw noise then the depth law is a single exponent with a stated residual and no floor term.

Bold conjecture under test (this tick's reflection): the reference's extremum error at fixed geometry is
a two-regime function of accumulation depth - a K-independent floor F below a boundary K*, and a power
law above it - because below K* the extremum is set by the per-element rounding of the reference's own
arithmetic tree rather than by accumulation depth. Conjecture is TRUE iff the K=1024 -> K=2048 decrease
reproduces in >=7 of 8 independent draws AND exceeds twice the pooled draw standard deviation.

Design. M=N=1024, K in {1024, 2048, 4096, 8192}; 8 independent draws per K (seed recorded per row:
9500 + 100*draw + k_index). Flags: allow_tf32_matmul=False, float32_matmul_precision=highest (the
benchmark's shipped fp32 default). Arms per draw:
  A  fp32 matmul (reference realization)
  X  fp64 anchor (exact widening of the same fp32 operands) - proxy for exact arithmetic
  C1 chunked-accumulation candidate (K split into 8 contiguous chunks, fp32 matmuls summed in fp32 in
     fixed order) - same tree as s94 so the share is comparable
Measurements per (draw,K): rho_ref = max_elements |A-X| / (1e-4 + 1e-4|X|); frac_over_ref at the strict
1e-4 predicate and at the shipped 1e-2 predicate (anchor-scaled, s94 convention); rho_cand =
max_elements |C1-A| / (1e-4 + 1e-4|A|) (evaluator-exact candidate direction, eval.py:804); failing-element
split REFERENCE-INDUCED (|A-X| > b_e) vs CANDIDATE-INDUCED among |C1-A| > b_e; extremum characterization
(argmax |X| relative to median |X|, and the absolute error at that element).

Statistics (predeclared): per-K mean/sd/CV of rho_ref over the 8 draws; per-draw log-log slope of rho_ref
vs K over the four K values; pooled slope; count of draws in which rho_ref(1024) > rho_ref(2048).

Kill clauses (predeclared, coded, fired verbatim):
  K1 instrument: a full repeat of draw 0 at K=1024 must reproduce rho_ref and rho_cand bit-identically;
     otherwise INSTRUMENT_FAILED and no verdict is issued.
  K2 stability: if the pooled CV of rho_ref at any K exceeds 10% => DRAWS_UNSTABLE; the dip cannot be
     resolved and no law claim is made this tick (the piecewise reading stands as written).
  K3 vacuous: if rho_ref is 0 on every row => INSTRUMENT_FAILED.

Branch clauses (only if no kill fired):
  FLOOR_REAL   the K=1024 -> K=2048 decrease appears in >=7/8 draws and |mean(1024)-mean(2048)| >
               2*pooled_sd => a draw-stable low-depth regime exists; ship eps(K) = max(F, c*K^alpha)
               with F measured at K=1024 and alpha from the upper segment only (K>=4096).
  FLOOR_NOISE  the decrease appears in <=4/8 draws OR |diff| <= 2*pooled_sd => the dip is not resolvable;
               report a single-exponent fit over all four K with its residual and retire the floor term.
  FLOOR_MIXED  otherwise (5-6 of 8 draws, or diff between 1 and 2 pooled sd) => report both readings,
               no constant claimed.
Share branch (secondary, descriptive unless it clears its own bar):
  SHARE_STABLE mean share within +/-0.05 of s94's 0.560 across draws => the partial-carrier share is a
               measured constant on this family; otherwise the share is draw-dependent and must be
               quoted per draw.
  SHARE_UNSTABLE mean share differs from 0.560 by >0.05 => the s94 point estimate is not repeatable.

Declared limits: GB10 sm_120 only; fp64 anchor is a proxy for exact arithmetic; matmul only; M=N=1024
fixed so geometry is not varied; chunked C1 is one particular alternative tree; 8 draws is enough to
resolve a 3.7% effect against a ~3.5% CV (s79a) but not to characterize a tail; no timing is measured.
"""
import json
import os
import signal
import time

import torch

LAB = os.path.expanduser("~/kb-lab")
OUT = os.path.join(LAB, "results", "s95_draw_repeat.jsonl")
SUMOUT = os.path.join(LAB, "results", "s95_draw_repeat_summary.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

STRICT = 1e-4
SHIPPED = 1e-2
CFG_ALARM_S = 300
CHUNK = 1 << 22
NDRAWS = 8
KS = [1024, 2048, 4096, 8192]
MN = 1024


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout("config exceeded %ds" % CFG_ALARM_S)


signal.signal(signal.SIGALRM, _alarm)


def set_flags():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def append(rec):
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")


def chunked(a, b, nchunks=8):
    K = a.shape[1]
    step = K // nchunks
    out = None
    for i in range(nchunks):
        sl = slice(i * step, (i + 1) * step)
        p = torch.mm(a[:, sl].contiguous(), b[sl, :].contiguous())
        out = p if out is None else out + p
    return out


def cmax(d, thr):
    """max of d/thr over elements, chunked over rows to bound memory."""
    fa = d.reshape(-1, d.shape[-1])
    ft = thr.reshape(-1, thr.shape[-1])
    m = 0.0
    for i in range(0, fa.shape[0], CHUNK):
        v = (fa[i:i + CHUNK] / ft[i:i + CHUNK]).max().item()
        m = max(m, v)
    return m


def counts(d, thr):
    fa = d.reshape(-1, d.shape[-1])
    ft = thr.reshape(-1, thr.shape[-1])
    c = 0
    for i in range(0, fa.shape[0], CHUNK):
        c += int((fa[i:i + CHUNK] > ft[i:i + CHUNK]).sum())
    return c


def run(draw, kidx, K):
    seed = 9500 + 100 * draw + kidx
    rec = dict(tid="D%d_K%d" % (draw, K), draw=draw, kidx=kidx, M=MN, N=MN, K=K, seed=seed,
               status="ok")
    t0 = time.time()
    signal.alarm(CFG_ALARM_S)
    try:
        set_flags()
        g = torch.Generator(device="cuda").manual_seed(seed)
        a = torch.randn(MN, K, generator=g, device="cuda", dtype=torch.float32)
        b = torch.randn(K, MN, generator=g, device="cuda", dtype=torch.float32)
        A = torch.mm(a, b)
        X = torch.mm(a.double(), b.double())
        C1 = chunked(a, b)
        Ad = A.double()
        dAX = (Ad - X).abs()
        thr_strict = STRICT + STRICT * X.abs()
        thr_ship = SHIPPED + SHIPPED * X.abs()
        rec["rho_ref"] = cmax(dAX, thr_strict)
        rec["rho_ref_ship"] = cmax(dAX, thr_ship)
        rec["frac_over_ref_strict"] = counts(dAX, thr_strict) / float(A.numel())
        rec["frac_over_ref_ship"] = counts(dAX, thr_ship) / float(A.numel())
        rec["max_abs_ref_err"] = dAX.max().item()
        # extremum characterization
        flat = dAX.reshape(-1)
        idx = int(flat.argmax().item())
        xflat = X.abs().reshape(-1)
        rec["extremum_abs_err"] = float(flat[idx].item())
        rec["extremum_abs_x"] = float(xflat[idx].item())
        rec["median_abs_x"] = float(xflat.median().item())
        rec["extremum_rel_to_median"] = rec["extremum_abs_x"] / max(rec["median_abs_x"], 1e-30)
        # candidate arm
        dCA = (C1 - A).abs()
        thr_cand = STRICT + STRICT * A.abs()
        rec["rho_cand"] = cmax(dCA, thr_cand)
        fail = dCA > thr_cand
        nfail = int(fail.sum().item())
        rec["n_fail_cand"] = nfail
        if nfail:
            ref_over = dAX > thr_cand
            rec["n_ref_induced"] = int((fail & ref_over).sum().item())
            rec["share"] = rec["n_ref_induced"] / float(nfail)
            rec["share_ceiling"] = int(ref_over.sum().item()) / float(nfail)
        else:
            rec["n_ref_induced"] = 0
            rec["share"] = None
            rec["share_ceiling"] = None
        rec["wall_s"] = time.time() - t0
        del a, b, A, X, C1, Ad, dAX, thr_strict, thr_ship, dCA, thr_cand, fail
        torch.cuda.empty_cache()
    except Timeout:
        rec["status"] = "timeout"
    except Exception as e:  # noqa: BLE001
        rec["status"] = "error"
        rec["error"] = repr(e)
    finally:
        signal.alarm(0)
    append(rec)
    return rec


def main():
    open(OUT, "w").close()
    rows = []
    for draw in range(NDRAWS):
        for kidx, K in enumerate(KS):
            r = run(draw, kidx, K)
            print("%s status=%s rho=%.6g rho_cand=%.6g share=%s wall=%.1fs"
                  % (r["tid"], r["status"], r.get("rho_ref", -1), r.get("rho_cand", -1),
                     r.get("share"), r.get("wall_s", -1)), flush=True)
            rows.append(r)
    # instrument check: repeat draw 0 K=1024 and compare bit-identity
    rep = run(0, 0, KS[0])
    base = [r for r in rows if r["tid"] == "D0_K1024"][0]
    inst = dict(
        repeat_rho_identical=(rep.get("rho_ref") == base.get("rho_ref")),
        repeat_rho_cand_identical=(rep.get("rho_cand") == base.get("rho_cand")),
        repeat_frac_identical=(rep.get("frac_over_ref_strict") == base.get("frac_over_ref_strict")),
    )
    summary = dict(rows=len(rows), ndraws=NDRAWS, ks=KS, instrument=inst)
    with open(SUMOUT, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(inst))


if __name__ == "__main__":
    main()
