"""s83 — pool behavioural diversity on an ARITHMETIC-SPANNING pool (the opposing school's
discriminating test, carried as the named next action since s60/s62).

Meta-reflection gihOpRvRb1TT5WQV ("Choose a useful smaller problem") answered 2026-11-02 12:22Z:
the smaller question is whether a nominal arm pool has behavioural content at all, because that
sets the granularity of the deliverable (per-arm entries vs class-cells) without requiring a
magnitude law or a dispatcher.

PRE-REGISTRATION — frozen before the first run (this docstring, committed pre-run).

Population (declared): two conv2d 3x3 s1 p1 fp32 shapes, GB10, torch 2.14.1+cu130, rev 423217d:
  S1 = (N=2, C=64, H=W=14)   -- inside the frozen hand-domain (H<=14 & C<=64)
  S2 = (N=2, C=256, H=W=28)  -- outside it (tensor-op / winograd region)
Seeds 1234 + 100*H + 10*N + C (r1_conv_gated convention).

Nominal arms (12 per shape, declared): torch_default (the reference realization),
torch_tf32off, torch_cudnn_off, torch_channels_last, torch_compile, torch_fp64 (anchor),
and 5 Triton implicit-GEMM tiles at ieee precision + 1 at tf32 (imp_B32x32[tf32]).
=> the pool spans arithmetic (fp64 accumulation, cudnn-disabled fallback, layout change,
TF32 vs IEEE dot) rather than only tile geometry, which is what s60 could not separate.

Measured per arm: candidate-scaled budget ratio (s73 predicate: |cand-ref| / (1e-4 + 1e-4*|cand|),
median + frac_over + q90/q99/max), rho vs the fp64 anchor (max normalized deviation), median time
(12 reps, interleaved with a 256MB L2 flush, warmup 3).

Dedup convention (s62's resolution convention, declared): arms are grouped by ratio value at
relative resolution {exact, 5%, 10%} and by (ratio, time) joint signature at the same resolutions.
"Noise-floor resolution" = the spread of repeated measurements of the SAME arm, measured here as
the max/min ratio across 3 independent re-measurements of torch_default (R0).

Pre-registered verdict rule:
  R1 per shape, distinct ratio-classes at 5% resolution: LOW if <=4 of 12, HIGH if >=8, else MIXED.
  R1 pooled: EFFECTIVE_LOW only if BOTH shapes LOW; EFFECTIVE_HIGH only if BOTH HIGH; else MIXED.
  R2 joint (ratio,time) signature count at 5% resolution, reported as the speed-axis check.
Kill / decision clause: EFFECTIVE_LOW => the degenerate-pool result generalizes beyond Triton
sweeps; the per-arm enumeration is dropped from the deliverable and the attribution column is
class-level. EFFECTIVE_HIGH => per-arm language stands and the column must carry per-arm entries.
MIXED => the column carries classes plus a measured within-class spread, no per-arm claim.

Output: results/s83_pool_spanning.{jsonl,json}
Run on spark: python3 s83_pool_spanning.py
"""
import json, os, time
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
OUTL = os.path.join(HERE, "results", "s83_pool_spanning.jsonl")
DATE = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
SHAPES = [(2, 64, 14), (2, 256, 28)]
VARIANTS = [
    ("imp_B32x32", 32, 32, 32, 4),
    ("imp_B64x32", 64, 32, 32, 4),
    ("imp_B64x64", 64, 64, 32, 4),
    ("imp_B128x64", 128, 64, 32, 8),
    ("imp_B32x64K64", 32, 64, 64, 4),
]
ATOL = RTOL = 1e-4


@triton.jit
def conv_impgemm_ip(X, Wt, O, N, H, W, C, CO,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                    IP: tl.constexpr):
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
                acc += tl.dot(a, b, input_precision=IP)
    optr = (O + n_i[:, None] * (CO * H * W) + offs_o[None, :] * (H * W)
            + oh[:, None] * W + ow[:, None])
    tl.store(optr, acc, mask=mmask[:, None] & omask[None, :])


def med(ts, n):
    return {"med_ms": ts[n // 2], "iqr_ms": [ts[int(.25 * n)], ts[min(n - 1, int(.75 * n))]]}


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
    ts.sort()
    return med(ts, len(ts)) | {"n_reps": len(ts)}


def ratio_stats(cand, ref):
    """s73 predicate, candidate-scaled, elementwise; returns the admission-relevant statistics."""
    d = (cand - ref).abs()
    r = d / (ATOL + RTOL * ref.abs())
    q = torch.quantile(r.flatten().float(), torch.tensor([0.5, 0.9, 0.99, 1.0], device=r.device))
    return {
        "ratio_median": float(q[0]), "ratio_q90": float(q[1]), "ratio_q99": float(q[2]),
        "ratio_max": float(q[3]), "frac_over": float((r > 1).float().mean()),
        "pass_1e4": bool(torch.allclose(ref, cand, rtol=RTOL, atol=ATOL)),
    }


def rho_vs_anchor(cand, anchor):
    d = (cand.double() - anchor).abs()
    return float((d / (ATOL + RTOL * anchor.abs())).max())


def main():
    dev = torch.device("cuda:0")
    flush = torch.empty(256 * 2**20 // 4, dtype=torch.float32, device=dev)
    out = []
    for (N, C, H) in SHAPES:
        torch.manual_seed(1234 + 100 * H + 10 * N + C)
        x = torch.randn(N, C, H, H, device=dev)
        w = torch.randn(C, C, 3, 3, device=dev)
        ref = F.conv2d(x, w, padding=1)
        anchor = F.conv2d(x.double(), w.double(), padding=1).float()
        row = dict(shape=[N, C, H], date=DATE, op_class="conv2d_3x3_s1p1_fp32",
                   arms={}, anchor_rho_ref=None)
        row["anchor_rho_ref"] = rho_vs_anchor(ref, anchor.double())
        arms = row["arms"]

        def add(name, fn, note=""):
            o = fn()
            if o.dtype == torch.float64:
                o = o.float()
            o = o.contiguous()
            st = ratio_stats(o, ref)
            t = evt_time(fn, flush=flush)
            st.update(t)
            st.update(rho_anchor=rho_vs_anchor(o.double(), anchor.double()), note=note)
            arms[name] = st

        add("torch_default", lambda: F.conv2d(x, w, padding=1), "reference realization")

        def tf32off():
            sv = torch.backends.cudnn.allow_tf32
            torch.backends.cudnn.allow_tf32 = False
            try:
                return F.conv2d(x, w, padding=1)
            finally:
                torch.backends.cudnn.allow_tf32 = sv
        add("torch_tf32off", tf32off, "cudnn tf32 disabled")

        def cudnn_off():
            sv = torch.backends.cudnn.enabled
            torch.backends.cudnn.enabled = False
            try:
                return F.conv2d(x, w, padding=1)
            finally:
                torch.backends.cudnn.enabled = sv
        add("torch_cudnn_off", cudnn_off, "cudnn disabled -> fallback path")

        xc = x.contiguous(memory_format=torch.channels_last)
        wc = w.contiguous(memory_format=torch.channels_last)
        add("torch_channels_last", lambda: F.conv2d(xc, wc, padding=1), "layout change")

        try:
            fnc = torch.compile(lambda xx: F.conv2d(xx, w, padding=1), dynamic=False)
            fnc(x); torch.cuda.synchronize()
            add("torch_compile", lambda: fnc(x), "inductor")
        except Exception as ex:
            arms["torch_compile"] = {"error": repr(ex)[:200]}

        add("torch_fp64", lambda: F.conv2d(x.double(), w.double(), padding=1), "anchor realization")

        M = N * H * H
        for name, BM, BN, BK, nw in VARIANTS:
            grid = (triton.cdiv(M, BM) * triton.cdiv(C, BN),)

            def mk(BM=BM, BN=BN, BK=BK, nw=nw, grid=grid, prec="ieee"):
                def go():
                    o = torch.empty(N, C, H, H, dtype=torch.float32, device=dev)
                    conv_impgemm_ip[grid](x, w, o, N, H, H, C, C, BM=BM, BN=BN, BK=BK,
                                          IP=prec, num_warps=nw)
                    return o
                return go
            add(f"{name}[ieee]", mk(prec="ieee"), f"triton tile {BM}x{BN}x{BK}")
        add("imp_B32x32[tf32]", mk(BM=32, BN=32, BK=32, nw=4, grid=(triton.cdiv(M, 32) * triton.cdiv(C, 32),), prec="tf32"),
            "triton tile 32x32 tf32")

        # R0: noise-floor resolution -- three independent re-measurements of the same arm
        reps = []
        for _ in range(3):
            o = F.conv2d(x, w, padding=1).contiguous()
            st = ratio_stats(o, ref)
            st.update(evt_time(lambda: F.conv2d(x, w, padding=1), flush=flush))
            reps.append(st)
        row["r0_same_arm_repeats"] = reps
        row["r0_ratio_spread"] = max(r["ratio_max"] for r in reps) / max(min(r["ratio_max"] for r in reps), 1e-30)
        out.append(row)
        print("done shape", row["shape"], "arms", len(arms), flush=True)

    with open(OUTL, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print("wrote", OUTL)


if __name__ == "__main__":
    main()
