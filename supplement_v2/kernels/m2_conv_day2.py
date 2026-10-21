"""M2: conv day-2 collector -- seed repeat + M-stacked (weight-traffic) arms.

Pre-registered BEFORE evaluation 2026-10-18 ~07:40Z (this docstring; DIARY same tick).
Plan milestone M2 (revival_plan.md): day-2 stability of the conv family-local boundary,
the one-flawed-variant confound broken, candidate check. GPU run is the NEXT tick; this
file only freezes the design. NO threshold peeking before the run (single evaluation
pass afterwards; rerun only for harness bugs).

Declared deltas vs w3_conv_gemm.py (all else identical: grid H{4,7,10,14,20,28} x
N{1,4} x C{16,64,256}; e2 harness; events med-12 IQR; 256MiB flush; immediate
correctness read; torch_cmp context-only):
  D2a seed stream 4321-based (day-1: 1234-based) -- pure repeat, stability test.
  D2b M-stacked arms `impw_*`: same kernel body with row-block BM*T at fixed BN,BK,
      warps -- the declared W-traffic intervention (weights loaded once per (r,s,c0)
      per CTA, reused across T m-tiles; day-1 flaw: traffic ~ w x ctas_m). Arms:
      B32x32T2 B32x32T4 B64x32T2 B64x32T4 (T=1 == day-1 form, kept as in-run
      replication check vs day-1 medians). Declared mechanism prediction, NOT a new
      tile search: if the large-H hand loss is W-traffic, impw T-response is
      monotone at H>=20 on fixed (BN,BK,warps).
  D2c day-1 arms repeated verbatim (imp_* 5 tiles) for cross-day comparison.

Pre-registered analyses (run in m2 analysis AFTER data lands; acceptance now):
  A1 day-2 stability: day-1-frozen rule (H<=14 & C<=64) OR (H<=7) applied to day-2
     labels: point-label agreement >= 33/36 AND pooled regret of the frozen rule on
     day-2 <= 2x day-1 (0.0953). Else: conv boundary DOWNgraded to single-run
     observation per wrapup_revival stop condition (P5, reported either way).
  A2 conv-basis refit on day-2: per-LOO-H-fold (H,C) 2-feature rule, pooled regret
     reported next to day-1's 0.0953 and day-1 E-deployed 0.1729 (m1). No acceptance
     beyond A1; this is the basis-transfer replication.
  A3 candidate check (report-only, stays `candidate`): imp_B32x32 vs torch_cmp
     median relative advantage at H<=14 & C<=64, day 2 vs day 1 (25-48% band).
  A4 mechanism: impw T-response at H in {20,28}, fixed BN/BK/warps:
     med(impw_T4) < med(impw_T2) < med(imp_T1) on >= 2 of 3 bases x {H20,H28}...
     tightened: monotone-in-T on B32x32 at H=28 AND H=20 => mechanism supported;
     non-monotone or flat => W-traffic account incomplete for large-H (recorded, the
     (H,C) rule stays empirical). Falsifier written, P5.
  Admission note (carried): E-bt gate on conv remains non-adaptive (m1: pooled
  0.1729, 1 worse point); conv admission rule is the (H,C) family-local predicate.

Run (on spark): python3 m2_conv_day2.py -> results/m2_conv_day2.json (+ .jsonl).
"""
import json, os, time
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from w3_conv_gemm import evt_time  # identical harness, provenance P1

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "m2_conv_day2.json")
OUTL = OUT + ".jsonl"
DATE = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
FLUSH_BYTES = 256 * 2**20
SEED_BASE = 4321  # D2a; day-1 used 1234

IMP = [  # day-1 replication (D2c): name, BM, BN, BK, warps
    ("imp_B32x32", 32, 32, 32, 4), ("imp_B64x32", 64, 32, 32, 4),
    ("imp_B64x64", 64, 64, 32, 4), ("imp_B128x64", 128, 64, 32, 8),
    ("imp_B32x64K64", 32, 64, 64, 4)]
IMPW = [  # D2b M-stacked: (name, BM, T, BN, BK, warps)
    ("impw_B32x32T2", 32, 2, 32, 32, 4), ("impw_B32x32T4", 32, 4, 32, 32, 4),
    ("impw_B64x32T2", 64, 2, 32, 32, 4), ("impw_B64x32T4", 64, 4, 32, 32, 4)]
HS = [4, 7, 10, 14, 20, 28]
NS = [1, 4]
CS = [16, 64, 256]


@triton.jit
def conv_impgemm_T(X, Wt, O, N, H, W, C, CO,
                   BM: tl.constexpr, T: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # day-1 body; single delta: row block = BM*T (T=1 reproduces w3_conv_gemm exactly)
    pid = tl.program_id(0)
    n_m = tl.cdiv(N * H * W, BM * T)
    pid_m = pid % n_m
    pid_n = pid // n_m
    offs_m = pid_m * BM * T + tl.arange(0, BM * T)
    mmask = offs_m < N * H * W
    n_i = offs_m // (H * W)
    rem = offs_m % (H * W)
    oh = rem // W
    ow = rem % W
    offs_o = pid_n * BN + tl.arange(0, BN)
    omask = offs_o < CO
    acc = tl.zeros((BM * T, BN), dtype=tl.float32)
    for r in tl.static_range(3):
        for s in tl.static_range(3):
            hh = oh + (r - 1)
            ww = ow + (s - 1)
            hok = (hh >= 0) & (hh < H)
            wok = (ww >= 0) & (ww < W)
            for c0 in range(0, C, BK):
                offs_c = c0 + tl.arange(0, BK)
                cok = offs_c < C
                aptr = (X + n_i[:, None] * (C * H * W) + offs_c[None, :] * (H * W)
                        + hh[:, None] * W + ww[:, None])
                a = tl.load(aptr, mask=mmask[:, None] & cok[None, :] & hok[:, None] & wok[:, None], other=0.0)
                bptr = Wt + offs_o[None, :] * (C * 9) + offs_c[:, None] * 9 + r * 3 + s
                b = tl.load(bptr, mask=omask[None, :] & cok[:, None], other=0.0)
                acc += tl.dot(a, b)
    optr = (O + n_i[:, None] * (CO * H * W) + offs_o[None, :] * (H * W)
            + oh[:, None] * W + ow[:, None])
    tl.store(optr, acc, mask=mmask[:, None] & omask[None, :])


def main():
    dev = torch.device("cuda:0")
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.float32, device=dev)
    done = set()
    rows = []
    if os.path.exists(OUTL):
        for line in open(OUTL):
            line = line.strip()
            if line:
                r = json.loads(line); done.add((r["N"], r["H"], r["C"])); rows.append(r)
    outf = open(OUTL, "a")
    for H in HS:
        for N in NS:
            for C in CS:
                if (N, H, C) in done:
                    continue
                torch.manual_seed(SEED_BASE + H * 100 + N * 10 + C)
                x = torch.randn(N, C, H, H, device=dev)
                w = torch.randn(C, C, 3, 3, device=dev)
                ref = F.conv2d(x, w, padding=1)
                row = dict(N=N, H=H, W=H, C=C, CO=C, date=DATE, seed_base=SEED_BASE,
                           M=N * H * H, K=C * 9,
                           bytes=x.numel() * 4 + w.numel() * 4 + ref.numel() * 4,
                           variants={})
                row["variants"]["torch"] = evt_time(lambda: F.conv2d(x, w, padding=1), flush=flush)
                row["variants"]["torch"]["max_abs_err"] = 0.0
                try:
                    fn = torch.compile(lambda xx: F.conv2d(xx, w, padding=1), dynamic=False)
                    row["variants"]["torch_cmp"] = evt_time(lambda: fn(x), flush=flush)
                    row["variants"]["torch_cmp"]["max_abs_err"] = \
                        (fn(x) - ref).abs().max().item()
                except Exception as ex:
                    row["variants"]["torch_cmp"] = {"error": repr(ex)[:200]}
                M = N * H * H
                for name, BM, BN, BK, nw in IMP:
                    out = torch.empty_like(ref)
                    grid = (triton.cdiv(M, BM) * triton.cdiv(C, BN),)

                    def go():
                        conv_impgemm_T[grid](x, w, out, N, H, H, C, C,
                                             BM=BM, T=1, BN=BN, BK=BK, num_warps=nw)
                        return out
                    go()
                    err = (out - ref).abs().max().item()
                    t = evt_time(go, flush=flush)
                    t.update(max_abs_err=err, ctas=grid[0], warps=nw, BM=BM, T=1, BN=BN, BK=BK)
                    row["variants"][name] = t
                for name, BM, T, BN, BK, nw in IMPW:
                    out = torch.empty_like(ref)
                    grid = (triton.cdiv(M, BM * T) * triton.cdiv(C, BN),)

                    def go():
                        conv_impgemm_T[grid](x, w, out, N, H, H, C, C,
                                             BM=BM, T=T, BN=BN, BK=BK, num_warps=nw)
                        return out
                    go()
                    err = (out - ref).abs().max().item()
                    t = evt_time(go, flush=flush)
                    t.update(max_abs_err=err, ctas=grid[0], warps=nw, BM=BM, T=T, BN=BN, BK=BK)
                    row["variants"][name] = t
                rows.append(row)
                tk = row["variants"]["torch"]["med_ms"]
                best = min((v["med_ms"], k) for k, v in row["variants"].items()
                           if k != "torch_cmp" and "med_ms" in v)
                print("  N%d H%d C%d: torch %.4f best %s %.4f" % (N, H, C, tk, best[1], best[0]))
                outf.write(json.dumps(row) + "\n"); outf.flush()
    json.dump({"meta": {"date": DATE, "sm": sm, "seed_base": SEED_BASE,
                        "variants": [v[0] for v in IMP] + [v[0] for v in IMPW],
                        "grid": "H{4,7,10,14,20,28} x N{1,4} x C{16,64,256} conv3x3 s1 p1 fp32",
                        "pre_reg": "this docstring; analyses A1-A4 on day-2 data only"},
               "rows": rows}, open(OUT, "w"), indent=1)
    print("wrote", OUT, len(rows), "rows")


if __name__ == "__main__":
    main()
