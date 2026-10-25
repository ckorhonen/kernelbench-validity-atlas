"""A10 — matmul rho=1 crossing: neighborhood + seed study. Answers f01MfL review finding 5.

PREREG FROZEN HERE BEFORE THE RUN.

The review's finding 5: A6d bisected K while drawing fresh shape-dependent random matrices and
evaluating a max-error statistic, with no monotonicity argument. It also points out the retained data
contradict the "monotone in K" claim: at M=N=1024 rho is 1.232 at K=3632 and 1.009 at K=3968. So
"the crossing is between 3608 and 3632" is two observed verdicts on two draws, not a threshold.

This leg measures the two things the reviewer says are missing:
  (1) SEED: at each K in a neighborhood of the bracket, measure rho for S independent seeds. Report the
      fraction of seeds with rho<1 (the empirical pass probability) and the rho distribution.
  (2) NEIGHBORHOOD: a dense K grid across the bracket so the shape of the transition is measured, not
      bisected. Report whether rho is monotone in K at FIXED seed (the only monotonicity claim that can
      be tested), and how often per-seed monotonicity holds.

Grid: M=N=1024, K in {3584, 3600, 3616, 3632, 3648, 3680, 3712, 3776, 3840, 3968, 4096}, S=6 seeds.
rho defined exactly as in a6/a6d: max_i |R_def,i - R_64,i| / (1e-4 + 1e-4|R_64,i|), allow_tf32=False,
float32_matmul_precision=highest.

FROZEN PREDICTIONS (written before the run; a fired falsifier is reported as a retraction, not a rescue):
  P1 SEED STABILITY OF THE BRACKET. At K=3608 and K=3632, all S seeds should give the same verdict
     (rho<1 at 3608, rho>=1 at 3632). If ANY seed disagrees at either K, the "two observed verdicts"
     language in the paper is WRONG and must be replaced by a probabilistic transition statement with
     the measured flip rate.
  P2 PER-SEED MONOTONICITY. rho should increase with K at fixed seed across the grid. If it does not for
     a majority of seeds, the max-error statistic is not monotone in K at fixed input and bisection on K
     has no justification at all; the crossing must then be reported as a measured bracket with the
     non-monotone region named.
  P3 TRANSITION SHARPNESS. The empirical pass probability should drop from 1.0 to 0.0 within the grid
     span (<= 512 in K). If it does not, the transition is gradual and the "crosses at K~3.6k" phrasing
     becomes "pass probability falls across K 3.5k-4.1k" with the measured curve.

Verdict mapping (frozen): the paper's §2 matmul sentence is replaced by whichever statement the data
support — unanimous-verdict bracket (P1 holds), stochastic transition with measured rates (P1 fires),
or non-monotone-with-bracket (P2 fires). All three are reportable; none is invented after the fact.

Run on spark: python3 a10_crossing_neighborhood.py -> results/a10_crossing_neighborhood.json
"""
import json, os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def _kb_res_dir(here):
    for cand in (os.path.join(here, "results"), os.path.join(os.path.dirname(here), "results")):
        if os.path.isdir(cand):
            return cand
    return os.path.join(here, "results")


RES = _kb_res_dir(HERE)
OUT = os.path.join(RES, "a10_crossing_neighborhood.json")
ATOL = RTOL = 1e-4
KS = [3584, 3600, 3616, 3632, 3648, 3680, 3712, 3776, 3840, 3968, 4096]
SEEDS = [9100, 9101, 9102, 9103, 9104, 9105]
M = N = 1024
dev = "cuda"


def rho_at(M, K, N, seed):
    torch.manual_seed(seed)
    A = torch.randn(M, K, device=dev)
    B = torch.randn(K, N, device=dev)
    d32 = A @ B
    d64 = A.double() @ B.double()
    r = ((d32.double() - d64).abs() / (ATOL + RTOL * d64.abs())).flatten()
    out = {"M": M, "K": K, "N": N, "seed": seed, "rho": float(r.max()),
           "med": float(r.median()), "frac_over": float((r > 1).double().mean())}
    del A, B, d32, d64
    torch.cuda.empty_cache()
    return out


hdr = {"date": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()), "torch": torch.__version__,
       "gpu": torch.cuda.get_device_name(0),
       "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
       "f32_prec": torch.get_float32_matmul_precision(),
       "prereg": "docstring frozen before run (f01MfL review finding 5)",
       "grid": {"M": M, "N": N, "K": KS, "seeds": SEEDS}}

rows = []
for K in KS:
    for s in SEEDS:
        r = rho_at(M, K, N, s)
        rows.append(r)
        print(f"K={K} seed={s} rho={r['rho']:.4f} frac_over={r['frac_over']:.3g}", flush=True)

by_k = {}
for K in KS:
    sub = [r for r in rows if r["K"] == K]
    rhos = sorted(x["rho"] for x in sub)
    by_k[str(K)] = {
        "n": len(sub),
        "pass_fraction": sum(1 for x in sub if x["rho"] < 1.0) / len(sub),
        "rho_min": rhos[0], "rho_med": rhos[len(rhos) // 2], "rho_max": rhos[-1],
        "verdicts": ["pass" if x["rho"] < 1.0 else "fail" for x in sub],
    }

# P2: monotonicity at fixed seed
mono = {}
for s in SEEDS:
    seq = [next(r["rho"] for r in rows if r["K"] == K and r["seed"] == s) for K in KS]
    inc = all(seq[i + 1] >= seq[i] for i in range(len(seq) - 1))
    viol = [{"K_from": KS[i], "K_to": KS[i + 1], "d_rho": seq[i + 1] - seq[i]}
            for i in range(len(seq) - 1) if seq[i + 1] < seq[i]]
    mono[str(s)] = {"monotone_increasing": inc, "violations": viol, "rho_by_K": seq}

n_mono = sum(1 for v in mono.values() if v["monotone_increasing"])

# P1: bracket unanimity at the two K values the paper names
bracket = {}
for K in (3608, 3632):
    if str(K) in by_k:
        bracket[str(K)] = by_k[str(K)]
    else:
        sub = [rho_at(M, K, N, s) for s in SEEDS]
        rows.extend(sub)
        rhos = sorted(x["rho"] for x in sub)
        bracket[str(K)] = {"n": len(sub),
                           "pass_fraction": sum(1 for x in sub if x["rho"] < 1.0) / len(sub),
                           "rho_min": rhos[0], "rho_med": rhos[len(rhos) // 2], "rho_max": rhos[-1],
                           "verdicts": ["pass" if x["rho"] < 1.0 else "fail" for x in sub]}
        by_k[str(K)] = bracket[str(K)]

p1_holds = all(bracket[k]["pass_fraction"] in (0.0, 1.0) for k in bracket)
p2_holds = n_mono > len(SEEDS) / 2
pass_fracs = [by_k[str(K)]["pass_fraction"] for K in KS]
p3_holds = (pass_fracs[0] == 1.0) and (pass_fracs[-1] == 0.0) and \
           any(0.0 < p < 1.0 for p in pass_fracs)

verdict = {
    "P1_bracket_unanimous": bool(p1_holds),
    "P2_per_seed_monotone_majority": bool(p2_holds),
    "P2_n_monotone_seeds": n_mono,
    "P3_sharp_transition": bool(p3_holds),
    "transition_pass_fraction_by_K": {str(K): by_k[str(K)]["pass_fraction"] for K in KS},
    "reading": None,
}
if not p1_holds:
    verdict["reading"] = "STOCHASTIC_TRANSITION: verdict at the bracket is not seed-stable; report the measured pass probability curve, not a threshold."
elif not p2_holds:
    verdict["reading"] = "NON_MONOTONE: rho is not monotone in K at fixed input; the bracket is a measured bracket, and the non-monotone region is named."
elif not p3_holds:
    verdict["reading"] = "GRADUAL: transition spans more than the grid; report the curve."
else:
    verdict["reading"] = "UNANIMOUS_BRACKET: verdicts are seed-stable at the named K values and monotone at fixed seed; the two-verdict statement stands, scoped to sampled inputs."

out = {"header": hdr, "rows": rows, "by_K": by_k, "monotonicity": mono,
       "bracket": bracket, "verdict": verdict}
json.dump(out, open(OUT, "w"), indent=1)
print("wrote", OUT)
print(json.dumps(verdict, indent=1))
