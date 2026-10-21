"""W3 conv-as-gemm collector: implicit-GEMM conv (Triton) vs torch, shape grid.

Pre-registered BEFORE evaluation 2026-10-18 ~00:50Z (this docstring; diary same tick).
Purpose (reflection #7 decision, verbatim): the decisive 1-vs-2-feature test. W4's
ladder verdict (E: hand iff M>=16 AND bytes/thread>=1024) comes from ONE op family
(column-sum); a 1-feature result on conv is compatible with both the atlas and a
per-op-knob null. Decision rule, written now:
  - ladder winner on conv needs >=2 features  -> spine exhibit, atlas proceeds;
  - 1-feature winner on conv too              -> draft null-paper skeleton (W5 pivot).

Design (declared up front):
  Op: conv2d 3x3 stride1 pad1, fp32, NCHW, C_in=C_out=C. GEMM framing:
  M=N*OH*OW rows, K=C*9 reduction, N'=CO cols. OH=OW=H (square inputs).
  Arms:
    torch       F.conv2d (eager; ladder-comparable, same convention as e2/w4)
    torch_cmp   torch.compile(dynamic=False) -- IDENTITY#3 honest comparator,
                reported as CONTEXT, excluded from gate and oracle (declared)
    imp_*       5 Triton implicit-GEMM tiles (BM,BN,BK,warps):
                B32x32(32,32,32,4) B64x32(64,32,32,4) B64x64(64,64,32,4)
                B128x64(128,64,32,8) B32x64K64(32,64,64,4)
  Grid: H=W in {4,7,10,14,20,28} x N in {1,4} x C in {16,64,256} = 36 shapes.
  Timing: e2 harness verbatim (events, med of 12, IQR, 256MiB L2 flush each rep,
  3 warmups). Correctness: max abs err vs fp32 conv2d, computed immediately from a
  fresh run per variant (e2_debug2 lesson: no shared/closure-written buffers read
  late); tf32 dot declared (matches cudnn fp32 conv numerics latitude).
  Ladder (w3_ladder.py, run after): families A-E IDENTICAL to w4_dispatcher_v1 --
  refit hand prior per fold on train, thresholds refit min train regret, ties ->
  smaller hand region then fewer features; bt = 2*bytes/(ctas*128) via fallback
  variant ctas (v1's declared approximation, kept for identity); grids extended for
  conv scales, declared: m0 +{128,256,512}, K0 +{128,...,4096}, b0 +16.
  Fold axis: LOO over the 6 (H,W) sizes -- tests shape generalization to unseen
  spatial sizes, the operationally relevant axis; declared instead of LOO-K.
  Fallback hand variant when refit choice absent at a test point: imp_B64x32
  (a priori middle tile; coverage is complete on this grid, expected unused).
  Adoption bar unchanged: 0 held-out pts >1% worse than torch AND mean regret <
  always-torch; report honestly either way (P5).
  One rerun allowed for harness bugs only; no threshold peeking then refit.

Run: python3 w3_conv_gemm.py  (on spark) -> results/w3_conv_gemm.json (+ .jsonl
incremental; safe to rerun, completed shapes are skipped).
"""
import json, os, time
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "w3_conv_gemm.json")
OUTL = OUT + ".jsonl"
DATE = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
FLUSH_BYTES = 256 * 2**20

VARIANTS = [  # name, BM, BN, BK, warps
    ("imp_B32x32", 32, 32, 32, 4),
    ("imp_B64x32", 64, 32, 32, 4),
    ("imp_B64x64", 64, 64, 32, 4),
    ("imp_B128x64", 128, 64, 32, 8),
    ("imp_B32x64K64", 32, 64, 64, 4),
]
FALLBACK = "imp_B64x32"
HS = [4, 7, 10, 14, 20, 28]
NS = [1, 4]
CS = [16, 64, 256]


@triton.jit
def conv_impgemm(X, Wt, O, N, H, W, C, CO,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    n_m = tl.cdiv(N * H * W, BM)
    pid_m = pid % n_m
    pid_n = pid // n_m
    offs_m = pid_m * BM + tl.arange(0, BM)
    mmask = offs_m < N * H * W
    n_i = offs_m // (H * W)
    rem = offs_m % (H * W)
    oh = rem // W
    ow = rem % W
    offs_o = pid_n * BN + tl.arange(0, BN)
    omask = offs_o < CO
    acc = tl.zeros((BM, BN), dtype=tl.float32)
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


def evt_time(fn, warmup=3, reps=12, flush=None):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(reps):
        if flush is not None:
            flush.zero_()
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort(); n = len(ts)
    return {"med_ms": ts[n // 2], "iqr_ms": [ts[int(.25 * n)], ts[min(n - 1, int(.75 * n))]], "n_reps": n}


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
                torch.manual_seed(1234 + H * 100 + N * 10 + C)
                x = torch.randn(N, C, H, H, device=dev)
                w = torch.randn(C, C, 3, 3, device=dev)
                ref = F.conv2d(x, w, padding=1)
                row = dict(N=N, H=H, W=H, C=C, CO=C, date=DATE,
                           M=N * H * H, K=C * 9,
                           bytes=x.numel() * 4 + w.numel() * 4 + ref.numel() * 4,
                           variants={})
                # eager
                row["variants"]["torch"] = evt_time(lambda: F.conv2d(x, w, padding=1), flush=flush)
                row["variants"]["torch"]["max_abs_err"] = 0.0
                # compiled (context arm; excluded from gate/oracle)
                try:
                    fn = torch.compile(lambda xx: F.conv2d(xx, w, padding=1), dynamic=False)
                    row["variants"]["torch_cmp"] = evt_time(lambda: fn(x), flush=flush)
                    row["variants"]["torch_cmp"]["max_abs_err"] = \
                        (fn(x) - ref).abs().max().item()
                except Exception as ex:
                    row["variants"]["torch_cmp"] = {"error": repr(ex)[:200]}
                # hand implicit-gemm tiles
                M = N * H * H
                for name, BM, BN, BK, nw in VARIANTS:
                    out = torch.empty_like(ref)
                    grid = (triton.cdiv(M, BM) * triton.cdiv(C, BN),)
                    def go():
                        conv_impgemm[grid](x, w, out, N, H, H, C, C,
                                           BM=BM, BN=BN, BK=BK, num_warps=nw)
                        return out
                    go()  # fresh correctness run, read immediately
                    err = (out - ref).abs().max().item()
                    t = evt_time(go, flush=flush)
                    t["max_abs_err"] = err
                    t["ctas"] = grid[0]
                    t["warps"] = nw
                    t["BM"], t["BN"], t["BK"] = BM, BN, BK
                    row["variants"][name] = t
                rows.append(row)
                tk = row["variants"]["torch"]["med_ms"]
                best = min((v["med_ms"], k) for k, v in row["variants"].items()
                           if k != "torch_cmp" and "med_ms" in v)
                cmp_t = row["variants"]["torch_cmp"].get("med_ms")
                print("  N%d H%d C%d: torch %.4f best %s %.4f cmp %s" % (
                    N, H, C, tk, best[1], best[0],
                    "%.4f" % cmp_t if cmp_t else "n/a"))
                outf.write(json.dumps(row) + "\n"); outf.flush()
    json.dump({"meta": {"date": DATE, "sm": sm, "variants": [v[0] for v in VARIANTS],
                        "grid": "Hx{4,7,10,14,20,28} N{1,4} C{16,64,256} conv3x3 s1 p1 fp32",
                        "note": "torch_cmp = context arm only"} ,
               "rows": rows}, open(OUT, "w"), indent=1)
    print("wrote", OUT, len(rows), "rows")


if __name__ == "__main__":
    main()
