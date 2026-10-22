"""A2 day-2 delivery replication: gate vs compile on FRESH shapes (delivery arms only).

Pre-registered BEFORE evaluation 2026-10-19 07:59Z (this docstring; DIARY same
tick). A2's killer objection (ATLAS.md): reduction leg n.s. at n=24; day-2
candidate. This run is the A2 day-2 check named there, and the atlas lane
wrap-up artifact (strategic reflection Exploitation, stage 0).

GATE FROZEN exactly as day-1 (w4_heldout.py; asserted vs m3 meta at runtime):
  reduction: hand iff M>=16 and bt>=1024, bt = 2*bytes/(ctas*128),
             ctas = cdiv(M,32)*64; variant tsplit_S64_KT64.
  conv:      hand iff H<=14 and C<=64; variant imp_B32x32.

FRESH GRIDS, disjoint as point-sets from every prior surface (day-1 heldout
K=3*2^k M{3,12,20,48,192,384}; e2 pow2; e2b; w3/m3 conv (H,C) pairs):
  reduction: K in {5,7}*2^{13,15,17} (odd multipliers, never pow2),
             M in {5,10,24,40,224,448} (none equal day-1 values; 10/24 and
             40 straddle the M>=16 edge from both sides).
  conv:      N=2 with (H,C) in {6,10,18,26}x{48,128}; plus (4,9,48),(4,20,128).

ARMS per point (delivery space only; no probe arms -- A2 is the v->0 end):
  gate     0 probes, frozen rule, med-of-12, flush-charged.
  torch    delivery, med-of-12.
  compile  torch.compile dynamic=False; med-of-12; wall recorded, NOT charged.
  oracle   med-of-12 min over {torch} + hand variants (regret denominator).
SEED 1234 for arm-order shuffles (day-1 used 4321; declared).

PRIMARY (pre-registered; per family, paired over points, delivery regret):
  conv:      REPLICATION of A2's decisive leg iff mean(regret_gate -
             regret_compile) > 0 AND paired t >= 2 (day-1: +56.8pp, t=5.54).
  reduction: NOT-SEPARABLE replicates iff |t| < 2 (day-1: t=-0.92, n=24).
             If |t| >= 2 either way: record direction; A2's hybrid claim is
             REVISION-REQUIRED toward the winning arm on reduction.
FALSIFIER (P5, written now): conv t < 2 or mean delta <= 0 => NO-REPLICATION;
A2 downgrades to 'unresolved in both families', hybrid retracted to n.s.
HONESTY: delivered-choice max-abs err per arm; compile walls; fires fraction.

Run on spark: python3 a2_day2.py -> results/a2_day2.json(.jsonl incr);
rows present in the .jsonl are skipped on rerun.
"""
import json, os, time, random
import torch
import torch.nn.functional as F
import triton
from e2_splitk_v3 import split_tiled, split_reduce
from w3_conv_gemm import conv_impgemm, VARIANTS
from m3_autotune_curve import evt, RED_SKT, FLUSH_BYTES, BM

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "a2_day2.json")
OUTL = OUT + ".jsonl"
M3 = os.path.join(RES, "m3_curve.json")
DATE = time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())
SEED = 1234

RED_VARIANT = "tsplit_S64_KT64"
CONV_VARIANT, CONV_H0, CONV_C0 = "imp_B32x32", 14, 64

RED_KS = [m * 2 ** k for m in (5, 7) for k in (13, 15, 17)]
RED_MS = [5, 10, 24, 40, 224, 448]
CONV_POINTS = [(2, h, c) for h in (6, 10, 18, 26) for c in (48, 128)] + \
              [(4, 9, 48), (4, 20, 128)]


def red_gate(M, bytes_):
    ctas = ((M + BM - 1) // BM) * 64
    bt = 2 * bytes_ / (ctas * 128)
    return (M >= 16 and bt >= 1024), bt


def main():
    m3meta = json.load(open(M3))["meta"]
    cg = m3meta["conv_gate_fit"]
    assert (cg["h0"], cg["c0"], cg["variant"]) == (CONV_H0, CONV_C0, CONV_VARIANT), cg

    dev = torch.device("cuda:0")
    try:
        torch._dynamo.config.recompile_limit = 256
    except Exception:
        pass
    flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.float32, device=dev)
    rng = random.Random(SEED)
    red_order = [n for n, _, _ in RED_SKT]; rng.shuffle(red_order)
    conv_order = [v[0] for v in VARIANTS]; rng.shuffle(conv_order)

    rows = []
    done = set()
    if os.path.exists(OUTL):
        for line in open(OUTL):
            if line.strip():
                r = json.loads(line)
                if r["family"] == "reduction":
                    done.add(("reduction", r["K"], r["M"], None))
                else:
                    done.add(("conv", r["N"], r["H"], r["C"]))
                rows.append(r)
    outl = open(OUTL, "a")

    def finish(row):
        row["date"] = DATE
        rows.append(row); outl.write(json.dumps(row) + "\n"); outl.flush()

    # ---------------- reduction (delivery arms) ----------------
    X = torch.randn(max(RED_KS) * max(RED_MS), dtype=torch.float32, device=dev)
    P = torch.empty(64 * max(RED_MS) * 2, dtype=torch.float32, device=dev)
    for K in RED_KS:
        for M in RED_MS:
            if ("reduction", K, M, None) in done:
                continue
            x = X[:K * M].view(K, M).contiguous()
            o = torch.empty(M, dtype=torch.float32, device=dev)
            ref = torch.sum(x, dim=0)
            bytes_ = K * M * 4 + M * 4
            fires, bt = red_gate(M, bytes_)
            row = dict(family="reduction", K=K, M=M, bytes=bytes_, bt_deployed=bt,
                       domain_pred="FROZEN day-1: M>=16 and bt>=1024 -> tsplit_S64_KT64",
                       gate_fires=bool(fires), variants={})
            pool = {"torch": (lambda: torch.sum(x, dim=0, out=o))}
            for name, S, KT in RED_SKT:
                if K // S < KT:
                    continue
                gm = triton.cdiv(M, BM)

                def run(S=S, KT=KT, gm=gm, x=x, P=P, K=K, M=M, o=o):
                    split_tiled[(gm, S)](x, P, K, M, S, BM=BM, KT=KT, num_warps=4)
                    split_reduce[(triton.cdiv(M, 256),)](P, o, M, S, BM=256, num_warps=4)
                pool["tsplit_" + name] = run
            times, errs = {}, {}
            for nm, fn in pool.items():
                t, iq = evt(fn, 2, 12, flush)
                times[nm] = t
                if nm.startswith("tsplit"):
                    o.zero_(); fn(); torch.cuda.synchronize()
                    errs[nm] = float((o - ref).abs().max())
            cw = None; cerr = None
            cf = torch.compile(lambda: torch.sum(x, dim=0, out=o), dynamic=False)
            t0 = time.perf_counter()
            try:
                cf(); torch.cuda.synchronize()
                cw = time.perf_counter() - t0
                o.zero_(); cf(); torch.cuda.synchronize()
                cerr = float((o - ref).abs().max())
                tc, iqc = evt(cf, 0, 12, flush)
            except Exception as ex:
                cw = -1.0; tc, iqc, cerr = None, None, "compile-fail:%r" % (ex,)
            t_oracle = min(times.values())
            gname = RED_VARIANT if fires else "torch"
            row["t_oracle_ms"] = t_oracle; row["oracle_arg"] = min(times, key=times.get)
            row["arms"] = dict(
                gate=dict(choice=gname, t=times[gname], probes=0.0,
                          regret=(times[gname] - t_oracle) / t_oracle,
                          err=errs.get(gname, 0.0)),
                torch=dict(t=times["torch"], regret=(times["torch"] - t_oracle) / t_oracle),
                compile=dict(t=tc, iqr=iqc, compile_wall_s=cw, err=cerr,
                             regret=(None if tc is None else (tc - t_oracle) / t_oracle)))
            row["errs"] = errs; row["n_reps"] = 12
            finish(row)
            print("red K%d M%d: oracle %.4f gate %s fires=%s" %
                  (K, M, t_oracle, gname, fires), flush=True)

    # ---------------- conv (delivery arms) ----------------
    for N, H, C in CONV_POINTS:
        if ("conv", N, H, C, None) in done or ("conv", N, H, C) in done:
            continue
        torch.manual_seed(1234 + H * 100 + N * 10 + C)
        x = torch.randn(N, C, H, H, device=dev)
        w = torch.randn(C, C, 3, 3, device=dev)
        ref = F.conv2d(x, w, padding=1)
        M = N * H * H
        row = dict(family="conv", N=N, H=H, C=C, M=M,
                   domain_pred="FROZEN day-1: H<=14 and C<=64 -> imp_B32x32",
                   gate_fires=bool(H <= CONV_H0 and C <= CONV_C0), variants={})
        bufs = {}
        pool = {"torch": (lambda: F.conv2d(x, w, padding=1))}
        for name, BMc, BNc, BKc, nw in VARIANTS:
            out = torch.empty_like(ref); bufs[name] = out
            grid = (triton.cdiv(M, BMc) * triton.cdiv(C, BNc),)

            def go(BMc=BMc, BNc=BNc, BKc=BKc, nw=nw, out=out, grid=grid):
                conv_impgemm[grid](x, w, out, N, H, H, C, C,
                                   BM=BMc, BN=BNc, BK=BKc, num_warps=nw)
            pool[name] = go
        times, errs = {}, {}
        for nm, fn in pool.items():
            t, iq = evt(fn, 2, 12, flush)
            times[nm] = t
            if nm in bufs:
                bufs[nm].zero_(); fn(); torch.cuda.synchronize()
                errs[nm] = float((bufs[nm] - ref).abs().max())
        cw = None; cerr = None
        cfn = torch.compile(lambda: F.conv2d(x, w, padding=1), dynamic=False)
        t0 = time.perf_counter()
        try:
            _out = cfn(); torch.cuda.synchronize()
            cw = time.perf_counter() - t0
            _out2 = cfn(); torch.cuda.synchronize()
            cerr = float((_out2 - ref).abs().max())
            tc, iqc = evt(cfn, 0, 12, flush)
        except Exception as ex:
            cw = -1.0; tc, iqc, cerr = None, None, "compile-fail:%r" % (ex,)
        t_oracle = min(times.values())
        fires = H <= CONV_H0 and C <= CONV_C0
        gname = CONV_VARIANT if (fires and CONV_VARIANT in pool) else "torch"
        row["t_oracle_ms"] = t_oracle; row["oracle_arg"] = min(times, key=times.get)
        row["arms"] = dict(
            gate=dict(choice=gname, t=times[gname], probes=0.0,
                      regret=(times[gname] - t_oracle) / t_oracle,
                      err=errs.get(gname, 0.0)),
            torch=dict(t=times["torch"], regret=(times["torch"] - t_oracle) / t_oracle),
            compile=dict(t=tc, iqr=iqc, compile_wall_s=cw, err=cerr,
                         regret=(None if tc is None else (tc - t_oracle) / t_oracle)))
        row["errs"] = errs; row["n_reps"] = 12
        finish(row)
        print("conv N%d H%d C%d: oracle %.4f gate %s fires=%s" %
              (N, H, C, t_oracle, gname, fires), flush=True)

    # ---------------- pre-registered analysis ----------------
    import math
    def mean(v):
        v = [x for x in v if x is not None]
        return sum(v) / len(v) if v else None

    def paired_t(a, b):
        d = [x - y for x, y in zip(a, b) if x is not None and y is not None]
        if len(d) < 2:
            return None, None, 0
        m = sum(d) / len(d)
        var = sum((x - m) ** 2 for x in d) / (len(d) - 1)
        se = (var / len(d)) ** 0.5 if var > 0 else 0.0
        t = m / se if se > 0 else (0.0 if m == 0 else math.inf * (1 if m > 0 else -1))
        return m, t, len(d)

    summary = {"date": DATE, "pre_reg": "docstring 07:59Z 2026-10-19",
               "op_class": "reduction+conv", "rule": "FROZEN day-1 gate, day-2 shapes",
               "domain_predicate": [RED_VARIANT, "M>=16,bt>=1024", CONV_VARIANT, "H<=14,C<=64"],
               "derived_coords": ["M", "bt_deployed", "H", "C"],
               "boundary_ref": "ATLAS A2 day-2", "n_reps": 12, "seed": SEED,
               "grids": {"red_Ks": RED_KS, "red_Ms": RED_MS, "conv": CONV_POINTS},
               "families": {}}
    for fam in ("reduction", "conv"):
        rs = [r for r in rows if r["family"] == fam]
        if not rs:
            continue
        g = [r["arms"]["gate"]["regret"] for r in rs]
        c = [r["arms"]["compile"]["regret"] for r in rs]
        m, t, n = paired_t(g, c)  # gate - compile; day-1 conv +56.8pp favors gate
        fires = sum(1 for r in rs if r["gate_fires"]) / len(rs)
        cwalls = [r["arms"]["compile"]["compile_wall_s"] for r in rs
                  if r["arms"]["compile"]["compile_wall_s"] and r["arms"]["compile"]["compile_wall_s"] > 0]
        if fam == "conv":
            verdict = ("REPLICATES" if (m is not None and m > 0 and t is not None and t >= 2)
                       else "NO-REPLICATION")
        else:
            verdict = ("NOT-SEPARABLE-replicates" if (t is not None and abs(t) < 2)
                       else "SEPARABLE-REVISION-REQUIRED dir=%s" % ("gate" if (m or 0) > 0 else "compile"))
        summary["families"][fam] = dict(
            n=len(rs), gate_fires_frac=round(fires, 3),
            mean_gate_minus_compile= None if m is None else round(m, 5),
            paired_t=None if t is None else round(t, 3), verdict=verdict,
            gate_regret=mean(g), compile_regret=mean(c),
            compile_wall_s_mean=mean(cwalls),
            compile_wall_s_max=max(cwalls) if cwalls else None)
    json.dump({"meta": summary, "rows": rows}, open(OUT, "w"), indent=1)
    print(json.dumps(summary["families"], indent=1))
    print("wrote", OUT, len(rows), "rows")


if __name__ == "__main__":
    main()
