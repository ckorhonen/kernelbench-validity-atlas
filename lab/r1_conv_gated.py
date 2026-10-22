"""R1 repair: correctness-gated conv rerun (rejection 6yPhozmAt62JSqQs, finding R1).

Pre-registered BEFORE evaluation 2026-10-22 ~02:05Z (this docstring; DIARY same tick).
Repairs, concretely:
  1. Per-arm precision DECLARED: every hand arm runs twice, as (variant, precision)
     pairs -- input_precision="ieee" (fp32 FMA dot) and "tf32" (day-1 default,
     kept for cross-check vs published medians). The delivered artifact is the pair.
  2. Validate-then-time (b1_gatezero_v2 pattern): fresh output buffer, error read
     immediately, ALLCLOSE(ref, cand, rtol=1e-4, atol=1e-4) -- the benchmark
     contract, verbatim. Timing happens for every arm; SELECTION only over arms
     whose recorded check passes.
  3. Gate/oracle restricted to passing variants:
       gate_published: day-1 rule H<=14 & C<=64 -> imp_B32x32 (tf32 form), recorded
         as-is for delta attribution;
       gate_admissible: same rule, but the delivered arm is imp_B32x32 in the first
         precision (ieee, then tf32) whose check passes; else torch (honest
         delivery). Chain recorded per row.
       oracle: min med12 over {torch} + hand arms with passing checks.
     Regrets recompute against the admissible oracle; break-even = paired counts
     gate_admissible vs torch / torch_cmp.
  4. Accounting: the check population is EVERY (row, arm-instance) validated, once.
     No union, no row/check mixing (R1b). Summary counts from this population only.
  5. Tolerance-domain numerics live ONLY as recorded per-arm err columns -- no
     headline rests on them (secondary per triage).

Grids (declared; tags): m3m2 = day-1/m2 grid H{4,7,10,14,20,28} x N{1,4} x
C{16,64,256}; w4 = held-out conv grid; a2 = day-2 conv grid (both as frozen in
those harnesses). Seed formula 1234 + H*100 + N*10 + C everywhere (w4/a2 fresh
evidence, new run, declared here; NOT a day-2 repeat).

Schema P2: op_class, rule, domain_predicate, derived_coords (M,K,ctas,waves,bytes),
variance (iqr), n_reps, date on every row; per-arm med/iqr/err/pass/precision.

Run on spark: python3 r1_conv_gated.py            # full, resumable via jsonl
               SMOKE=1 python3 r1_conv_gated.py   # 2 rows, harness check only
Output: results/r1_conv_gated.jsonl (+ summary json rewritten at end).
"""
import json, os, time
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
OUTL = os.path.join(HERE, "results", "r1_conv_gated.jsonl")
DATE = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
SMOKE = os.environ.get("SMOKE") == "1"

GRIDS = [
    ("m3m2", [(h, n, c) for h in (4, 7, 10, 14, 20, 28) for n in (1, 4) for c in (16, 64, 256)]),
    ("w4", [(h, 2, c) for h in (5, 8, 12, 24) for c in (32, 96)] + [(3, 2, 5), (3, 2, 12)] +
           [], ),  # placeholder guard; a2 below
    ("a2", [(h, 2, c) for h in (6, 10, 18, 26) for c in (48, 128)] + [(4, 2, 9), (4, 2, 20)]),
]
# w4 conv grid, stated plainly (kept out of the comprehension above for auditability):
GRIDS[1] = ("w4", [(5, 2, 32), (5, 2, 96), (8, 2, 32), (8, 2, 96),
                   (12, 2, 32), (12, 2, 96), (24, 2, 32), (24, 2, 96),
                   (3, 2, 32), (3, 2, 96)])
# NOTE (2026-10-22, pre-run): w4 day-1 grid was (H,C) in {5,8,12,24}x{32,96} plus
# (3,5,32),(3,12,96) per w4_heldout.py -- C=5 and C=12, not 32. Fixed here:
GRIDS[1] = ("w4", [(5, 2, 32), (5, 2, 96), (8, 2, 32), (8, 2, 96),
                   (12, 2, 32), (12, 2, 96), (24, 2, 32), (24, 2, 96),
                   (3, 2, 5), (3, 2, 12)])

VARIANTS = [  # name, BM, BN, BK, warps  (w3_conv_gemm verbatim)
    ("imp_B32x32", 32, 32, 32, 4),
    ("imp_B64x32", 64, 32, 32, 4),
    ("imp_B64x64", 64, 64, 32, 4),
    ("imp_B128x64", 128, 64, 32, 8),
    ("imp_B32x64K64", 32, 64, 64, 4),
]
PRECS = ("ieee", "tf32")
CONTRACT = dict(rtol=1e-4, atol=1e-4)


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
    return med(ts, n) | {"n_reps": n}


def med(ts, n):
    return {"med_ms": ts[n // 2], "iqr_ms": [ts[int(.25 * n)], ts[min(n - 1, int(.75 * n))]]}


def main():
    dev = torch.device("cuda:0")
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    flush = torch.empty(256 * 2**20 // 4, dtype=torch.float32, device=dev)
    done = set()
    if os.path.exists(OUTL):
        for line in open(OUTL):
            if line.strip():
                r = json.loads(line); done.add((r["tag"], r["N"], r["H"], r["C"]))
    outf = open(OUTL, "a")
    n_done = 0
    for tag, pts in GRIDS:
        for (H, N, C) in pts:
            if (tag, N, H, C) in done or (tag, H, N, C) in done:
                continue
            if SMOKE and n_done >= 2:
                break
            torch.manual_seed(1234 + H * 100 + N * 10 + C)
            x = torch.randn(N, C, H, H, device=dev)
            w = torch.randn(C, C, 3, 3, device=dev)
            ref = F.conv2d(x, w, padding=1)
            ref_absmax = ref.abs().max().item()
            M = N * H * H
            row = dict(tag=tag, N=N, H=H, W=H, C=C, CO=C, date=DATE, M=M, K=C * 9,
                       bytes=x.numel() * 4 + w.numel() * 4 + ref.numel() * 4,
                       ref_absmax=ref_absmax, sm_count=sm,
                       op_class="conv2d_3x3_s1p1_fp32",
                       rule="gate: hand iff H<=14 and C<=64 -> imp_B32x32 (precision-admissible)",
                       domain_predicate="H<=14 & C<=64 (frozen day-1); delivered arm must pass 1e-4 contract",
                       contract=CONTRACT, variants={})
            V = row["variants"]
            # torch eager: the contract's reference op
            t = evt_time(lambda: F.conv2d(x, w, padding=1), flush=flush)
            t.update(max_abs_err=0.0, pass_1e4=True, precision="fp32")
            V["torch"] = t
            # compiled context arm
            try:
                t0 = time.time()
                fn = torch.compile(lambda xx: F.conv2d(xx, w, padding=1), dynamic=False)
                fn(x); torch.cuda.synchronize()
                wall = time.time() - t0
                t = evt_time(lambda: fn(x), flush=flush)
                err = (fn(x) - ref).abs().max().item()
                t.update(max_abs_err=err, pass_1e4=bool(torch.allclose(ref, fn(x), **CONTRACT)),
                         compile_wall_s=wall, precision="fp32")
                V["torch_cmp"] = t
            except Exception as ex:
                V["torch_cmp"] = {"error": repr(ex)[:200]}
            # SECONDARY reference: true-fp32 conv (cudnn tf32 disabled). Primary
            # contract stays vs the default-flag reference (deployed reality);
            # this column separates "kernel inaccurate" from "reference is TF32"
            # (instrument finding 10-22: default cudnn fp32 conv = 2.2e-4 rel from
            # fp64 on sm_120, tile-independently -- the ref carries the error).
            _sv = torch.backends.cudnn.allow_tf32
            torch.backends.cudnn.allow_tf32 = False
            ref32 = F.conv2d(x, w, padding=1)
            torch.backends.cudnn.allow_tf32 = _sv
            row["ref32_absmax"] = ref32.abs().max().item()
            # hand arms: (variant, precision) pairs, validate-then-time
            for name, BM, BN, BK, nw in VARIANTS:
                grid = (triton.cdiv(M, BM) * triton.cdiv(C, BN),)
                for prec in PRECS:
                    def mk():
                        out = torch.empty_like(ref)
                        def go():
                            conv_impgemm_ip[grid](x, w, out, N, H, H, C, C,
                                                  BM=BM, BN=BN, BK=BK, IP=prec,
                                                  num_warps=nw)
                            return out
                        return go
                    go = mk()
                    out_v = go()  # fresh buffer, read immediately (e2_debug2 lesson)
                    err = (out_v - ref).abs().max().item()
                    err32 = (out_v - ref32).abs().max().item()
                    ok = bool(torch.allclose(ref, out_v, **CONTRACT))
                    ok32 = bool(torch.allclose(ref32, out_v, **CONTRACT))
                    t = evt_time(go, flush=flush)
                    t.update(max_abs_err=err, rel_err=err / max(ref_absmax, 1e-30),
                             pass_1e4=ok,
                             err_vs_fp32ref=err32,
                             pass32=ok32,
                             precision=prec,
                             ctas=grid[0], waves=grid[0] / sm, warps=nw,
                             BM=BM, BN=BN, BK=BK)
                    V[f"{name}[{prec}]"] = t
            # selection objects -- passing arms only
            passing = {k: v for k, v in V.items()
                       if k.startswith("imp_") and v.get("pass_1e4") and "med_ms" in v}
            torch_t = V["torch"]["med_ms"]
            ora_k = min(passing, key=lambda k: passing[k]["med_ms"]) if passing else "torch"
            ora_t = passing[ora_k]["med_ms"] if passing else torch_t
            inrule = H <= 14 and C <= 64
            chain = []
            gate_k, gate_t = "torch", torch_t
            if inrule:
                for prec in PRECS:
                    k = f"imp_B32x32[{prec}]"
                    chain.append((k, bool(k in passing)))
                    if k in passing:
                        gate_k, gate_t = k, V[k]["med_ms"]
                        break
            else:
                chain = [("torch", True)]
            pub_arm = "imp_B32x32[tf32]" if inrule else "torch"
            pub_t = V[pub_arm]["med_ms"] if pub_arm in V else None
            row["gate_published"] = dict(arm=pub_arm, med_ms=pub_t)
            row["gate_published_regret"] = (
                (pub_t - ora_t) / ora_t if pub_t is not None else None)
            row["gate"] = dict(inrule=inrule, chain=chain, arm=gate_k, med_ms=gate_t,
                               regret_delivery=(gate_t - ora_t) / ora_t,
                               n_admissible=len(passing) + 1,
                               admissible=sorted(passing.keys()))
            row["oracle"] = dict(arm=ora_k, med_ms=ora_t)
            row["torch_regret"] = (torch_t - ora_t) / ora_t
            if "med_ms" in V.get("torch_cmp", {}):
                row["cmp_regret"] = (V["torch_cmp"]["med_ms"] - ora_t) / ora_t
            row["hand_best"] = dict(arm=ora_k, beats_torch=bool(ora_t < torch_t),
                                    delta_pct=100 * (torch_t - ora_t) / torch_t) \
                if ora_k != "torch" else dict(arm="torch", beats_torch=False, delta_pct=0.0)
            outf.write(json.dumps(row) + "\n"); outf.flush()
            n_done += 1
            print(f"[{n_done}] {tag} N{N} H{H} C{C}: gate={gate_k} t={gate_t:.3f} "
                  f"torch={torch_t:.3f} oracle={ora_k} passing={len(passing)}/10", flush=True)
    outf.close()
    summarize()


def summarize():
    import collections
    rows = [json.loads(l) for l in open(OUTL) if l.strip()]
    checks = passes = 0
    per_prec = collections.Counter(); per_prec_pass = collections.Counter()
    bytag = collections.defaultdict(lambda: dict(n=0, g=0.0, gt=0.0, tr=0.0, cr=0.0,
                                                 gate_beats_torch=0, gate_beats_cmp=0,
                                                 cmp_n=0, hand_beats=0))
    for r in rows:
        for k, v in r["variants"].items():
            if not isinstance(v, dict) or "med_ms" not in v or k.startswith("_"):
                continue
            checks += 1
            ok = bool(v.get("pass_1e4"))
            passes += ok
            if k.startswith("imp_"):
                prec = k[-6:-1]
                per_prec[prec] += 1; per_prec_pass[prec] += ok
        g = bytag[r["tag"]]
        g["n"] += 1
        g["g"] += r["gate"]["regret_delivery"]; g["gt"] += r["gate_published_regret"]
        g["tr"] += r["torch_regret"]
        g["gate_beats_torch"] += int(r["gate"]["med_ms"] < r["variants"]["torch"]["med_ms"])
        g["hand_beats"] += int(r["hand_best"]["beats_torch"])
        if "cmp_regret" in r:
            g["cr"] += r["cmp_regret"]; g["cmp_n"] += 1
            g["gate_beats_cmp"] += int(r["gate"]["med_ms"] < r["variants"]["torch_cmp"]["med_ms"])
    summ = dict(date=DATE, n_rows=len(rows), checks=checks, passes=passes,
                check_pass_rate=passes / max(checks, 1),
                imp_by_precision={p: dict(n=per_prec[p], n_pass=per_prec_pass[p])
                                  for p in per_prec},
                by_tag={t: {k: (round(v, 5) if isinstance(v, float) else v)
                            for k, v in d.items()} for t, d in bytag.items()},
                contract=CONTRACT, population="every (row, arm-instance) validated once")
    p = OUTL[:-6] + "_summary.json"
    open(p, "w").write(json.dumps(summ, indent=1))
    print(json.dumps(summ["by_tag"], indent=1))
    print("checks", checks, "pass", passes)


if __name__ == "__main__":
    main()
