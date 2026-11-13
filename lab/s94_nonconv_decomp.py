#!/usr/bin/env python
"""s94_nonconv_decomp.py — non-conv candidate x reference decomposition at 1e-4, and the reference's
accumulation-depth law (matmul).

PREREG (frozen before any run; commit this file before executing on spark; do not edit after).

Question. Two debts have been carried for several ticks and one of them four times: (1) the deferred
non-conv candidate x reference decomposition - does the 0.9985 reference-induced share measured on the
TF32 conv population (s84/s87) survive in a family where the reference's error is driven by accumulation
depth rather than by an engine-class constant? (2) s91's coordinate carries a class constant eps = 2.2e-4
(TF32 tensorop) and an unexplained 22-31% cross-gate shift, which is a label rather than a law.

Population (declared by rule before outcome, from A6b/A6d/A6f, not selected by this run's results):
  matmul, fp32, allow_tf32_matmul=False, float32_matmul_precision=highest (the benchmark's shipped
  default for fp32 gemm). A6d measured the reference's own rho=1 crossing at K=3608-3632 for M=N=1024,
  so K in {1024, 2048} are below-crossing controls and K in {4096, 8192} are above-crossing.
  T1 M=N=1024 K=1024; T2 M=N=1024 K=2048; T3 M=N=1024 K=4096; T4 M=N=1024 K=8192;
  T5 FRESH GEOMETRY M=N=512 K=8192 (A6f recorded a geometry term at fixed K; T5 tests whether the depth
  law is geometry-independent). randn inputs at scale 1, seeds fixed and recorded.

Arms (same tensors, one instantiation):
  A   fp32 matmul, default benchmark flags (matmul tf32 off, precision highest) - the reference realization
  X   fp64 anchor: exact widening of the same fp32 operands, fp64 matmul
  C1  chunked-accumulation candidate: K split into 8 contiguous chunks, each a fp32 matmul, summed in fp32
      in fixed order (a different accumulation tree, the hand-kernel analog; declared before the run)
  C2  tf32-on candidate (allow_tf32_matmul=True, precision high): class control, expected to fail
  C3  torch.compile(A, dynamic=False) fp32 default: bit-identity control (recorded, not a kill)

Predicates: evaluator-exact direction, b_e = atol + rtol*|candidate| (eval.py:804 convention verified in
earlier ticks), atol=rtol=1e-4 (strict axis) and atol=rtol=1e-2 (shipped gate). rho_ref = max_elements
|A-X| / (1e-4 + 1e-4*|X|); rho_cand = max_elements |C-A| / b_e.

Measurements per tuple and arm: rho_ref, frac_over_ref (elements with |A-X| > 1e-4 + 1e-4|X|), rho_cand,
frac_over_cand at both gates, failing-element split for C1 and C2 among elements where |C-A| > b_e:
REFERENCE-INDUCED (|A-X| > b_e, would fail even with an exact candidate) vs CANDIDATE-INDUCED, pooled
share with a Wilson-free plain ratio and per-tuple counts.

Depth law (predeclared two-branch test, the reflection's extension): fit log-log slope of rho_ref against
K over T1-T4 (M=N=1024 fixed). If rounding errors add incoherently the slope is 0.5; if they add coherently
(systematic bias) the slope is 1.0. Measured slope decides which branch; T5 is reported separately and is
not pooled into the fit.

Kill clauses (predeclared, coded, fired verbatim):
  K1  if C1 fails no element at 1e-4 on any tuple => the decomposition is vacuous on this population; report
      the null and keep the conv constant scoped to conv.
  K2  if the pooled reference-induced share of C1's failing elements is < 5% => the reference's error is not
      a material contributor to non-conv candidate failure margins at 1e-4.
  K3  instrument: A-arm repeat not bit-identical, or X repeat not bit-identical => INSTRUMENT_FAILED.
  K4  if the rho_ref log-log slope over T1-T4 falls outside [0.3, 1.2] => NO_POWER_LAW.

Branch clauses (only if no kill fired):
  DEPTH_LAW_INCOHERENT  slope in [0.3, 0.7]
  DEPTH_LAW_COHERENT    slope in [0.85, 1.15]
  DEPTH_LAW_INTERMEDIATE otherwise (reported, no constant claimed)
  REF_SHARE_DOMINANT    pooled C1 reference-induced share >= 0.95
  REF_SHARE_PARTIAL     0.05 <= share < 0.95
  REF_SHARE_CANDIDATE_DOMINANT < 0.05

Declared limits: GB10 sm_120 only; fp64 anchor is a proxy for exact arithmetic; matmul only (one non-conv
family); one draw per tuple (s79a DRAW_STABLE at fixed tuple, not re-tested here); chunked C1 is one
particular alternative tree, not a survey of trees; the depth law is a fit over four K values, so it is a
power-law exponent, not a mechanism proof.
"""
import hashlib
import json
import os
import signal
import time

import torch

LAB = os.path.expanduser("~/kb-lab")
OUT = os.path.join(LAB, "results", "s94_nonconv_decomp.jsonl")
SUMOUT = os.path.join(LAB, "results", "s94_nonconv_decomp_summary.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

STRICT = 1e-4
SHIPPED = 1e-2
CFG_ALARM_S = 300
CHUNK = 1 << 22

TUPLES = [
    dict(tid="T1", M=1024, N=1024, K=1024, seed=3101),
    dict(tid="T2", M=1024, N=1024, K=2048, seed=3102),
    dict(tid="T3", M=1024, N=1024, K=4096, seed=3103),
    dict(tid="T4", M=1024, N=1024, K=8192, seed=3104),
    dict(tid="T5", M=512, N=512, K=8192, seed=3105),
]


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout("config exceeded %ds" % CFG_ALARM_S)


signal.signal(signal.SIGALRM, _alarm)


def set_flags(tf32_matmul, precision):
    torch.backends.cuda.matmul.allow_tf32 = tf32_matmul
    torch.set_float32_matmul_precision(precision)


def append(rec):
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")


def mm(a, b):
    return torch.mm(a, b)


def chunked(a, b, nchunks=8):
    K = a.shape[1]
    step = K // nchunks
    out = None
    for i in range(nchunks):
        sl = slice(i * step, (i + 1) * step)
        p = torch.mm(a[:, sl].contiguous(), b[sl, :].contiguous())
        out = p if out is None else out + p
    return out


def counts(d, thr):
    """element count of d > thr, chunked over rows to bound memory."""
    fa = d.reshape(-1, d.shape[-1])
    ft = thr.reshape(-1, thr.shape[-1])
    c = 0
    for i in range(0, fa.shape[0], CHUNK):
        c += int((fa[i:i + CHUNK] > ft[i:i + CHUNK]).sum())
    return c


def run_tuple(td):
    rec = dict(tid=td["tid"], M=td["M"], N=td["N"], K=td["K"], seed=td["seed"], status="ok")
    t0 = time.time()
    signal.alarm(CFG_ALARM_S)
    try:
        torch.manual_seed(td["seed"])
        a = torch.randn(td["M"], td["K"], device="cuda", dtype=torch.float32)
        b = torch.randn(td["K"], td["N"], device="cuda", dtype=torch.float32)
        rec["a_absmax"] = float(a.abs().max())
        rec["b_absmax"] = float(b.abs().max())

        set_flags(False, "highest")
        A1 = mm(a, b)
        A2 = mm(a, b)
        rec["a_repeat_bit_identical"] = bool(torch.equal(A1, A2))

        with torch.no_grad():
            X1 = torch.mm(a.double(), b.double())
            X2 = torch.mm(a.double(), b.double())
        rec["x_repeat_bit_identical"] = bool(torch.equal(X1, X2))

        set_flags(False, "highest")
        C1 = chunked(a, b)
        set_flags(True, "high")
        C2 = mm(a, b)
        set_flags(False, "highest")
        A3 = mm(a, b)

        rec["ref_absmax"] = float(X1.abs().max())
        d_ref = (A3.double() - X1).abs()
        rec["rho_ref_1e4"] = float((d_ref / (STRICT + STRICT * X1.abs())).max())
        rec["rho_ref_1e2"] = float((d_ref / (SHIPPED + SHIPPED * X1.abs())).max())
        rec["frac_over_ref_1e4"] = float(counts(d_ref, STRICT + STRICT * X1.abs())) / d_ref.numel()
        rec["frac_over_ref_1e2"] = float(counts(d_ref, SHIPPED + SHIPPED * X1.abs())) / d_ref.numel()

        arms = {}
        for name, C in (("C1_chunked", C1), ("C2_tf32on", C2)):
            b_e = STRICT + STRICT * C.abs()
            d = (C.double() - A3.double()).abs()
            nfail = counts(d, b_e)
            # explicit split: among failing elements, those where d_ref > b_e
            fa = d.reshape(-1, d.shape[-1])
            fe = b_e.reshape(-1, b_e.shape[-1])
            fr = d_ref.reshape(-1, d_ref.shape[-1])
            n_ref = 0
            n_cand = 0
            for i in range(0, fa.shape[0], CHUNK):
                f = fa[i:i + CHUNK] > fe[i:i + CHUNK]
                r = fr[i:i + CHUNK] > fe[i:i + CHUNK]
                n_ref += int((f & r).sum())
                n_cand += int((f & ~r).sum())
            b_e2 = SHIPPED + SHIPPED * C.abs()
            arms[name] = dict(
                rho_cand_1e4=float((d / b_e).max()),
                rho_cand_1e2=float((d / b_e2).max()),
                nfail_1e4=nfail,
                frac_over_cand_1e4=float(nfail) / d.numel(),
                frac_over_cand_1e2=float(counts(d, b_e2)) / d.numel(),
                n_fail_ref_induced=n_ref,
                n_fail_cand_induced=n_cand,
                share_ref_induced=(float(n_ref) / nfail if nfail else None),
            )
            del d, b_e, b_e2, fa, fe, fr
        rec["arms"] = arms

        set_flags(False, "highest")
        try:
            cc = torch.compile(lambda x, y: torch.mm(x, y), dynamic=False)
            C3 = cc(a, b)
            rec["C3_compile_bit_identical_to_A"] = bool(torch.equal(C3, A3))
        except Exception as e:  # noqa: BLE001
            rec["C3_compile_bit_identical_to_A"] = None
            rec["C3_error"] = str(e)[:200]
        rec["seconds"] = round(time.time() - t0, 2)
        return rec
    except Timeout as e:
        rec["status"] = "timeout"
        rec["error"] = str(e)
        return rec
    except Exception as e:  # noqa: BLE001
        rec["status"] = "error"
        rec["error"] = repr(e)[:300]
        return rec
    finally:
        signal.alarm(0)


def main():
    if os.path.exists(OUT):
        os.remove(OUT)
    rows = []
    for td in TUPLES:
        rec = run_tuple(td)
        append(rec)
        rows.append(rec)
        print(json.dumps({k: rec.get(k) for k in ("tid", "status", "rho_ref_1e4", "rho_ref_1e2",
                                                  "frac_over_ref_1e4", "seconds")}), flush=True)
        if "arms" in rec:
            for n, v in rec["arms"].items():
                print("   ", n, json.dumps(v), flush=True)
        torch.cuda.empty_cache()

    ok = [r for r in rows if r.get("status") == "ok"]
    summ = dict(prereg="lab/s94_nonconv_decomp.py (this docstring)", n_rows=len(rows), n_ok=len(ok),
                sha_rows=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:16])
    # K3 instrument
    instr_fail = [r["tid"] for r in ok if not r.get("a_repeat_bit_identical") or not r.get("x_repeat_bit_identical")]
    summ["instrument_fail_tids"] = instr_fail
    # depth law fit on T1-T4
    fit = [(r["K"], r["rho_ref_1e4"]) for r in ok if r["tid"] in ("T1", "T2", "T3", "T4")]
    if len(fit) >= 3 and all(v > 0 for _, v in fit):
        import math
        xs = [math.log(k) for k, _ in fit]
        ys = [math.log(v) for _, v in fit]
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) ** 2 for x in xs)
        slope = num / den if den else None
        resid = max(abs(y - (my + slope * (x - mx))) for x, y in zip(xs, ys))
        summ["depth_fit"] = dict(slope=slope, max_log_resid=resid, points=fit)
    else:
        summ["depth_fit"] = dict(slope=None, points=fit)
    # C1 pooled share
    tot_ref = sum(r["arms"]["C1_chunked"]["n_fail_ref_induced"] for r in ok if "arms" in r)
    tot_cand = sum(r["arms"]["C1_chunked"]["n_fail_cand_induced"] for r in ok if "arms" in r)
    tot_fail = sum(r["arms"]["C1_chunked"]["nfail_1e4"] for r in ok if "arms" in r)
    summ["C1_pooled"] = dict(n_fail=tot_fail, n_ref_induced=tot_ref, n_cand_induced=tot_cand,
                             share_ref_induced=(tot_ref / tot_fail if tot_fail else None))
    tot_fail2 = sum(r["arms"]["C2_tf32on"]["nfail_1e4"] for r in ok if "arms" in r)
    tot_ref2 = sum(r["arms"]["C2_tf32on"]["n_fail_ref_induced"] for r in ok if "arms" in r)
    summ["C2_pooled"] = dict(n_fail=tot_fail2, n_ref_induced=tot_ref2,
                             share_ref_induced=(tot_ref2 / tot_fail2 if tot_fail2 else None))

    # kills, verbatim
    verdicts = []
    if instr_fail:
        verdicts.append("INSTRUMENT_FAILED")
    c1_any = tot_fail > 0
    if not c1_any:
        verdicts.append("K1_DECOMPOSITION_VACUOUS")
    elif (tot_ref / tot_fail) < 0.05:
        verdicts.append("K2_REFERENCE_NOT_MATERIAL")
    slope = summ["depth_fit"].get("slope")
    if slope is None or slope < 0.3 or slope > 1.2:
        verdicts.append("K4_NO_POWER_LAW")
    if not verdicts:
        if 0.3 <= slope <= 0.7:
            verdicts.append("DEPTH_LAW_INCOHERENT")
        elif 0.85 <= slope <= 1.15:
            verdicts.append("DEPTH_LAW_COHERENT")
        else:
            verdicts.append("DEPTH_LAW_INTERMEDIATE")
        share = tot_ref / tot_fail
        if share >= 0.95:
            verdicts.append("REF_SHARE_DOMINANT")
        elif share >= 0.05:
            verdicts.append("REF_SHARE_PARTIAL")
        else:
            verdicts.append("REF_SHARE_CANDIDATE_DOMINANT")
    summ["verdicts"] = verdicts
    json.dump(summ, open(SUMOUT, "w"), indent=1, default=str)
    print("SUMMARY", json.dumps(summ, default=str)[:1500], flush=True)


if __name__ == "__main__":
    main()
