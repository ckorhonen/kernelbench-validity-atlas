#!/usr/bin/env python
"""s99_conditioning.py — E2b designed-conditioning probe (H1 spectral amplification), matmul fp32 on GB10.

PREREG: lab/s99_prereg.md, committed BEFORE this run. Do not edit thresholds after data.

Design in brief: M=N=1024, K=4096, benchmark fp32 flags. b is CONSTRUCTED with a prescribed singular
spectrum (orthonormal factors from QR of Gaussian draws, log-spaced singular values, Frobenius norm fixed
at unit-RMS scale) so kappa(b) sweeps {1,10,100,1e3,1e4} while shape, output scale and engine are held.
5 independent draws per level => 25 rows. Reference A = fp32 mm; X = fp64 anchor; per-element evaluator
ratio r = |A-X|/(1e-4+1e-4|X|).

Measurements per row: rho_ref, frac_over at 1e-4 and 1e-2, prescribed and measured (svdvals, fp64)
sigma_max/sigma_min/kappa, ||b||_F, sigma_max(a), median|X|, max|X|, extremum |X|/median|X|, extremum
abs error, engine identity on draw 0 of each level.

Verdict rule (frozen in the prereg): P1 PASS_CONDITIONING_PREDICTOR if |Spearman(kappa,rho_ref)|>=0.40 AND
span>=1.3x; P2 RETIRE_H1 if |rho|<0.40 AND span>=1.3x; P3 NO_LEVERAGE if span<1.3x.
Kills: K1 instrument repeat must be bit-identical on rho_ref and frac_over; K2 engine set must be identical
across levels else ENGINE_CONFOUNDED; K3 construction check (svdvals vs prescribed) must match <=1e-3 rel.

No timing claim (GPU may be shared).
"""
import json
import os
import time

import torch

LAB = os.path.expanduser("~/kb-lab")
OUT = os.path.join(LAB, "results", "s99_conditioning.jsonl")
SUMOUT = os.path.join(LAB, "results", "s99_conditioning_summary.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

STRICT = 1e-4
SHIPPED = 1e-2
MN = 1024
K = 4096
LEVELS = [1.0, 10.0, 100.0, 1000.0, 10000.0]
NDRAWS = 5
CHUNK = 1 << 22


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


def build_b(kappa, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q1 = torch.linalg.qr(torch.randn(K, K, generator=g, device="cuda", dtype=torch.float32))[0]
    q2 = torch.linalg.qr(torch.randn(MN, MN, generator=g, device="cuda", dtype=torch.float32))[0]
    s = torch.logspace(0.0, torch.log10(torch.tensor(kappa)).item(), K, device="cuda", dtype=torch.float32)
    s = s / s.norm() * (K * MN) ** 0.5  # ||b||_F = sqrt(K*N) => unit-RMS entries
    b = (q1 * s.unsqueeze(0)) @ q2.t()
    return b, s


def profile_mm(a, b):
    names = set()
    try:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            torch.mm(a, b)
            torch.cuda.synchronize()
        for evt in prof.key_averages():
            if evt.key and ("gemm" in evt.key.lower() or "sgemm" in evt.key.lower()
                            or "cutlass" in evt.key.lower() or "nvjet" in evt.key.lower()):
                names.add(evt.key)
    except Exception as e:  # noqa: BLE001
        names.add("PROFILE_ERROR:" + repr(e))
    return sorted(names)


def run_row(level, level_idx, draw, profile=False):
    seed = 9900 + 10 * level_idx + draw
    gen = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(MN, K, generator=gen, device="cuda", dtype=torch.float32)
    b, s_presc = build_b(level, seed + 500)
    a = a / a.norm() * (MN * K) ** 0.5

    A = torch.mm(a, b)
    X = torch.mm(a.double(), b.double())
    thr = STRICT * (1.0 + X.abs())
    D = (A.double() - X).abs()
    r = D / thr
    rho = cmax(r, torch.ones_like(r))
    frac_strict = counts(r, torch.ones_like(r)) / r.numel()
    thr2 = SHIPPED * (1.0 + X.abs())
    r2 = D / thr2
    frac_ship = counts(r2, torch.ones_like(r2)) / r2.numel()

    sv = torch.linalg.svdvals(b.double())
    smax_m = sv[0].item()
    smin_m = sv[-1].item()
    smax_p = s_presc.max().item()
    smin_p = s_presc.min().item()
    sv_a = torch.linalg.svdvals(a.double())

    idx = torch.argmax(r).item()
    xe = X.reshape(-1)[idx].item()
    ae = A.reshape(-1)[idx].item()
    err_e = abs(ae - xe)
    medX = X.abs().median().item()
    rec = {
        "script": "s99_conditioning",
        "level_kappa": level,
        "level_idx": level_idx,
        "draw": draw,
        "seed": seed,
        "shape": [MN, MN, K],
        "rho_ref": rho,
        "frac_over_strict": frac_strict,
        "frac_over_shipped": frac_ship,
        "sigma_max_b_prescribed": smax_p,
        "sigma_min_b_prescribed": smin_p,
        "sigma_max_b_measured": smax_m,
        "sigma_min_b_measured": smin_m,
        "kappa_measured": smax_m / smin_m if smin_m > 0 else None,
        "b_frobenius": float(torch.linalg.norm(b.double()).item()),
        "sigma_max_a": float(sv_a[0].item()),
        "median_absX": medX,
        "max_absX": float(X.abs().max().item()),
        "extremum_absX": abs(xe),
        "extremum_depth": abs(xe) / medX if medX > 0 else None,
        "extremum_abs_err": err_e,
        "n_fail_strict": int(round(frac_strict * r.numel())),
    }
    if profile:
        rec["engine"] = profile_mm(a, b)
    return rec


def main():
    set_flags()
    t0 = time.time()
    rows = []
    if os.path.exists(OUT):
        os.remove(OUT)
    # K1 instrument: repeat of level kappa=100 draw 0
    base = run_row(100.0, 2, 0)
    rep = run_row(100.0, 2, 0)
    k1 = (base["rho_ref"] == rep["rho_ref"]) and (base["frac_over_strict"] == rep["frac_over_strict"])
    with open(os.path.join(LAB, "results", "s99_instrument.json"), "w") as f:
        json.dump({"k1_bit_identical": k1, "base": base, "repeat": rep}, f, indent=1)
    print("K1 instrument bit-identical:", k1)

    engines = {}
    for li, lvl in enumerate(LEVELS):
        for d in range(NDRAWS):
            prof = (d == 0)
            rec = run_row(lvl, li, d, profile=prof)
            if prof:
                engines[lvl] = rec.get("engine")
            rows.append(rec)
            append(rec)
            print("lvl %.0f draw %d rho %.4f frac %.6f" % (lvl, d, rec["rho_ref"], rec["frac_over_strict"]))

    eng_sets = {json.dumps(v) for v in engines.values()}
    k2 = len(eng_sets) == 1
    # K3 construction check
    worst = 0.0
    for rec in rows:
        for kp, km in (("sigma_max_b_prescribed", "sigma_max_b_measured"),
                       ("sigma_min_b_prescribed", "sigma_min_b_measured")):
            p, m = rec[kp], rec[km]
            if p and p != 0:
                worst = max(worst, abs(m - p) / abs(p))
    k3 = worst <= 1e-3
    summary = {
        "script": "s99_conditioning",
        "prereg": "lab/s99_prereg.md",
        "levels": LEVELS,
        "ndraws": NDRAWS,
        "n_rows": len(rows),
        "k1_instrument_bit_identical": k1,
        "k2_engine_identical_across_levels": k2,
        "engines": engines,
        "k3_construction_worst_rel": worst,
        "k3_construction_ok": k3,
        "wall_s": time.time() - t0,
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(SUMOUT, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "engines"}, indent=1))


if __name__ == "__main__":
    main()
