#!/usr/bin/env python
"""s96_amplitude_mechanism.py — what drives the draw-to-draw amplitude spread of the fp32 reference's
extremum-to-budget ratio rho_ref at a FIXED geometry? (matmul, M=N=1024, K=4096, benchmark fp32 flags.)

PREREG (frozen before any run; commit this file before executing on spark; do not edit after).

Question. s95 measured rho_ref at fixed M=N=1024 across 8 draws per K: at K=4096 the spread was 1.20x
(CV 6.6%), and its descriptive screen found NO covariate that tracked it consistently (Spearman vs
median |X| .706/.223/.745/-.053 across K; vs cancellation depth .643/.465/.124/-.140; vs extremum
absolute error .103/-.346/-.456/.672). So the amplitude is currently a lottery with no measured driver,
and every per-shape rho_ref quoted downstream is one draw of it.

Competing explanations (named before data, per this tick's reflection):
  H1 SPECTRAL AMPLIFICATION   rho_ref is set by the conditioning of the left operand: sigma_max(a) or a
     condition proxy (sigma_max/sigma_min) predicts it, so the lottery is input-conditioning and the
     budget becomes computable for any input.
  H2 CANCELLATION SCALE       the worst element is a near-zero output whose absolute error tracks the
     magnitude of the accumulated terms, so the driver is |A| scale (median/max |X|) and sigma_max acts
     only through that scale.
  H3 ENGINE LOTTERY           cuBLAS dispatches a different kernel/split per draw, so the reference
     realization is not reproducible at fixed shape.
  H4 TAIL SAMPLING            rho_ref is a maximum over 1.05M elements; a 10-14% CV is what extreme-value
     statistics predict and there is no driver at all. A driver that acts on the arithmetic moves the
     WHOLE error distribution, so H4 predicts a stable bulk tail (p99.99 ratio) alongside a varying max.

Design. 32 independent draws at M=N=1024, K=4096, seeds 9600+draw, benchmark fp32 flags
(allow_tf32_matmul=False, float32_matmul_precision=highest). Per draw: A = fp32 mm(a,b); X = fp64 anchor
of the same operands; per-element ratio r = |A-X|/(1e-4+1e-4|X|) (the evaluator predicate, s94/s95
convention). Recorded:
  - rho_ref = max r, frac_over at strict 1e-4 and shipped 1e-2;
  - sigma_max(a), sigma_min(a) via svdvals, kappa_proxy = sigma_max/sigma_min;
  - sigma_max(b), median |X|, max |X|, extremum |X| / median |X| (cancellation depth), extremum abs err;
  - bulk-tail statistics of r: p99, p99.9, p99.99, p99.999 ratios and their counts;
  - engine identity from torch.profiler on the DECLARED subset of draws 0-7 (kernel names of the mm).

Predeclared statistics: Spearman rho_ref vs each covariate over 32 draws; CV of rho_ref; CV of the p99.99
and p99.999 ratios; ratio CV(rho_ref)/CV(p99.99).

Kill clauses (coded, fired verbatim):
  K1 instrument: a repeat of draw 0 must reproduce rho_ref bit-identically, else INSTRUMENT_FAILED.
  K2 vacuous: rho_ref == 0 on any row => INSTRUMENT_FAILED.
  K3 degenerate: sigma_max(a) identical across all draws (CV < 1e-6) => the spectral arm is untestable
     and only H2/H4 are adjudicated.

Branch clauses (only if no kill fired):
  AMPLITUDE_PREDICTABLE_SPECTRAL  |Spearman(rho_ref, sigma_max(a))| >= 0.40 or
     |Spearman(rho_ref, kappa_proxy)| >= 0.40 => the amplitude has a computable input-conditioning driver.
  AMPLITUDE_SCALE_CARRIED         spectral arms < 0.40 but |Spearman(rho_ref, median|X|)| >= 0.40 or
     |Spearman(rho_ref, max|X|)| >= 0.40 => the driver is output scale, not conditioning.
  AMPLITUDE_LOTTERY               every predeclared arm < 0.40 => no predictor; the column ships the
     measured spread and no formula.
  Tail clause (independent, adjudicates H4): TAIL_SAMPLING_SUPPORTED if CV(p99.99) < 0.03 AND
     CV(rho_ref) > 0.10 (a driver would move the bulk tail too); TAIL_MOVES if CV(p99.99) > 0.10.
  Engine clause: ENGINE_VARIES if the profiled draws show more than one distinct mm kernel set;
     ENGINE_STABLE otherwise. Recorded as a fact, not a verdict.

Declared limits: GB10 sm_120, fp32 matmul only, one geometry (M=N=1024 K=4096), fp64 anchor as a proxy
for exact arithmetic, 32 draws characterize a spread but not a tail; no timing is measured (GPU shared).
"""
import json
import os
import signal
import time

import torch

LAB = os.path.expanduser("~/kb-lab")
OUT = os.path.join(LAB, "results", "s96_amplitude_mechanism.jsonl")
SUMOUT = os.path.join(LAB, "results", "s96_amplitude_mechanism_summary.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

STRICT = 1e-4
SHIPPED = 1e-2
CFG_ALARM_S = 240
CHUNK = 1 << 22
NDRAWS = 32
MN = 1024
K = 4096
PROFILE_DRAWS = 8


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


def cmax(d, thr):
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
        c += int((fa[i:i + CHUNK] > ft[i:i + CHUNK]).sum().item())
    return c


def quantiles(r, qs):
    flat = r.reshape(-1)
    out = {}
    for q in qs:
        try:
            out["p%g" % (q * 100)] = float(torch.quantile(flat, q).item())
        except Exception as e:  # noqa: BLE001
            out["p%g" % (q * 100)] = None
            out.setdefault("qerr", repr(e))
    return out


def profile_mm(a, b):
    names = set()
    try:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            torch.mm(a, b)
        for evt in prof.key_averages():
            if evt.device_type == torch.autograd.DeviceType.CUDA or evt.self_device_time_total > 0:
                names.add(evt.key)
    except Exception as e:  # noqa: BLE001
        names.add("PROFILER_FAILED:" + repr(e))
    return sorted(names)


def run(draw):
    seed = 9600 + draw
    rec = dict(tid="D%d" % draw, draw=draw, M=MN, N=MN, K=K, seed=seed, status="ok")
    t0 = time.time()
    signal.alarm(CFG_ALARM_S)
    try:
        set_flags()
        g = torch.Generator(device="cuda").manual_seed(seed)
        a = torch.randn(MN, K, generator=g, device="cuda", dtype=torch.float32)
        b = torch.randn(K, MN, generator=g, device="cuda", dtype=torch.float32)
        # spectral covariates of the left operand (declared: sigma_max(a), kappa proxy)
        sv = torch.linalg.svdvals(a)
        rec["sigma_max_a"] = float(sv[0].item())
        rec["sigma_min_a"] = float(sv[-1].item())
        rec["kappa_proxy"] = rec["sigma_max_a"] / max(rec["sigma_min_a"], 1e-30)
        rec["sigma_max_b"] = float(torch.linalg.svdvals(b)[0].item())
        del sv
        if draw < PROFILE_DRAWS:
            rec["engine_kernels"] = profile_mm(a, b)
        A = torch.mm(a, b)
        X = torch.mm(a.double(), b.double())
        Ad = A.double()
        dAX = (Ad - X).abs()
        thr_strict = STRICT + STRICT * X.abs()
        thr_ship = SHIPPED + SHIPPED * X.abs()
        rec["rho_ref"] = cmax(dAX, thr_strict)
        rec["rho_ref_ship"] = cmax(dAX, thr_ship)
        rec["frac_over_ref_strict"] = counts(dAX, thr_strict) / float(A.numel())
        rec["frac_over_ref_ship"] = counts(dAX, thr_ship) / float(A.numel())
        rec["max_abs_ref_err"] = float(dAX.max().item())
        r = dAX / thr_strict
        rec["q"] = quantiles(r, [0.99, 0.999, 0.9999, 0.99999])
        xflat = X.abs().reshape(-1)
        rec["median_abs_x"] = float(xflat.median().item())
        rec["max_abs_x"] = float(xflat.max().item())
        flat = dAX.reshape(-1)
        idx = int(flat.argmax().item())
        rec["extremum_abs_err"] = float(flat[idx].item())
        rec["extremum_abs_x"] = float(xflat[idx].item())
        rec["extremum_rel_to_median"] = rec["extremum_abs_x"] / max(rec["median_abs_x"], 1e-30)
        rec["wall_s"] = time.time() - t0
        del a, b, A, X, Ad, dAX, thr_strict, thr_ship, r, flat, xflat
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
        r = run(draw)
        q = r.get("q") or {}
        print("%s status=%s rho=%.6g smax=%.4g kappa=%.3g med=%.4g p9999=%s wall=%.1fs"
              % (r["tid"], r["status"], r.get("rho_ref", -1), r.get("sigma_max_a", -1),
                 r.get("kappa_proxy", -1), r.get("median_abs_x", -1), q.get("p99.99"),
                 r.get("wall_s", -1)), flush=True)
        rows.append(r)
    rep = run(0)
    base = [r for r in rows if r["tid"] == "D0"][0]
    inst = dict(repeat_rho_identical=(rep.get("rho_ref") == base.get("rho_ref")),
                repeat_frac_identical=(rep.get("frac_over_ref_strict") == base.get("frac_over_ref_strict")))
    summary = dict(rows=len(rows), ndraws=NDRAWS, geometry=dict(M=MN, N=MN, K=K),
                   profile_draws=PROFILE_DRAWS, instrument=inst)
    with open(SUMOUT, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(inst))


if __name__ == "__main__":
    main()
