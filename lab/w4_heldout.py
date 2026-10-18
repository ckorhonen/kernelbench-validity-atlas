"""W4 held-out generalization lattice: frozen day-1 gate on fresh shapes.

Pre-registered BEFORE evaluation 2026-10-18 20:41Z (this docstring; DIARY same
tick, post reflection #13). This is A1's declared un-run axis: fresh-shape
generalization of the dispatcher, the program's load-bearing test.

GATE FROZEN from day-1 artifacts, hardcoded below and asserted at runtime
against results/m3_curve.json meta (P1 provenance); NO fitting, NO threshold
refit, NO LOO in this script -- that is the point:
  reduction: hand iff M>=16 and bt>=1024, bt = 2*bytes/(ctas*128),
             ctas = cdiv(M,32)*64; variant tsplit_S64_KT64 (M1-deployed).
  conv:      hand iff H<=14 and C<=64; variant imp_B32x32
             (day-1 full-data fit h0=14, c0=64, in-sample regret 0.054958).

FRESH GRIDS, disjoint as point-sets from every prior fitting/timing surface
(e2_splitk_v3 K=2^12..2^22 x M=2^0..2^9; e2b_floor K=2^8..2^17 x M 2..256;
w3/m3 conv H in 4..28, N in {1,4}, C in {16,64,256}):
  reduction: K in {3*2^13, 3*2^15, 3*2^17, 3*2^19} (never powers of two),
             M in {3, 12, 20, 48, 192, 384} (non-pow2; 12/20 straddle the
             M>=16 predicate edge; larger M enter via the bt term).
  conv:      N=2 with (H,C) in {5,8,12,24}x{32,96}; plus (3,5,32),(3,12,96).
             Inside and outside the frozen predicate; square W=H as day-1.

ARMS per point (fresh timing; no artifact timings reused for decisions):
  gate     0 probes, frozen rule, med-of-12.
  b1/b2/b4/b8  pool [torch] + seeded-shuffle(hand) seed 4321 (single-seed
           residue carried from A1); probes single-exec flush-charged,
           charged in full (m3 protocol); b caps at pool size (conv 6).
  torch    b=0 equivalent, med-of-12.
  oracle   med-of-12 min over {torch} + hand variants.
  compile  torch.compile(op), dynamic=False; delivery med-of-12; cold compile
           wall-clock recorded per point (compile_wall_s) but NOT charged to
           regret_total -- primary mirrors A1's probe-economics regime;
           compile amortization is a separate visits-axis, reported for it.

PRIMARY (pre-registered; supersedes truncated #12 phrasing): per family,
gate GENERALIZES iff mean regret_total(gate) <= min_b mean regret_total(b)
+ SE_paired; otherwise A1's adoption scope shrinks to the sampled lattice and
hybrid dispatcher+probe becomes the standing candidate (#12 reconsider-if).
SECONDARY (A1 killer objection (b), richer pool / compile space): paired
delivery regret compile-vs-gate with SE; compile_wall distribution.
HONESTY: fraction of points where gate delivery >1% worse than torch / oracle;
fresh-read max-abs errs for every delivered choice and compile.

FALSIFIER (P5, written now): if gate regret_total is indistinguishable from
b0-torch (|mean diff| < SE) AND the gate fires on <20% of a family's points,
record NO-GENERALIZATION-EVIDENCE (vacuous gate), not a win.

Run on spark: python3 w4_heldout.py -> results/w4_heldout.json(.jsonl incr);
rows present in the .jsonl are skipped on rerun (m3 resume pattern).
"""
import json, os, time, random
import torch
import torch.nn.functional as F
import triton
from e2_splitk_v3 import split_tiled, split_reduce
from w3_conv_gemm import conv_impgemm, VARIANTS
from m3_autotune_curve import evt, one_shot, RED_SKT, FLUSH_BYTES, BM

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "w4_heldout.json")
OUTL = OUT + ".jsonl"
M3 = os.path.join(RES, "m3_curve.json")
DATE = time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())
BUDGETS = [1, 2, 4, 8]
SEED = 4321

# ---- frozen day-1 gate (see docstring; asserted vs m3 meta below) ----
RED_VARIANT = "tsplit_S64_KT64"
CONV_VARIANT, CONV_H0, CONV_C0 = "imp_B32x32", 14, 64
CONV_MEAN_REGRET = 0.054958

RED_KS = [3 * 2 ** k for k in (13, 15, 17, 19)]
RED_MS = [3, 12, 20, 48, 192, 384]
CONV_POINTS = [(2, h, c) for h in (5, 8, 12, 24) for c in (32, 96)] + \
              [(3, 5, 32), (3, 12, 96)]


def red_gate(M, bytes_):
    ctas = ((M + BM - 1) // BM) * 64
    bt = 2 * bytes_ / (ctas * 128)
    return (M >= 16 and bt >= 1024), bt


def main():
    m3meta = json.load(open(M3))["meta"]
    cg = m3meta["conv_gate_fit"]
    assert (cg["h0"], cg["c0"], cg["variant"]) == (CONV_H0, CONV_C0, CONV_VARIANT), cg

    dev = torch.device("cuda:0")
    try:  # harness-bug fix: 34 fresh shapes > default 8 recompiles of the shared closure
        torch._dynamo.config.recompile_limit = 256
    except Exception:
        pass
    sm = torch.cuda.get_device_properties(0).multi_processor_count
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

    # ---------------- reduction ----------------
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
            ctas = ((M + BM - 1) // BM) * 64
            for name, S, KT in RED_SKT:
                if K // S < KT:
                    continue
                gm = triton.cdiv(M, BM)

                def run(S=S, KT=KT, gm=gm, x=x, P=P, K=K, M=M, o=o):
                    split_tiled[(gm, S)](x, P, K, M, S, BM=BM, KT=KT, num_warps=4)
                    split_reduce[(triton.cdiv(M, 256),)](P, o, M, S, BM=256, num_warps=4)
                pool["tsplit_" + name] = run
            times, iqrs, errs = {}, {}, {}
            for nm, fn in pool.items():
                t, iq = evt(fn, 2, 12, flush)
                times[nm], iqrs[nm] = t, iq
                if nm.startswith("tsplit"):
                    if nm == "torch":
                        continue
                    o.zero_(); fn(); torch.cuda.synchronize()
                    errs[nm] = float((o - ref).abs().max())
            # compile arm (delivery-space objection; wall clock recorded)
            cw = None; cerr = None
            cf = torch.compile(lambda: torch.sum(x, dim=0, out=o), dynamic=False)
            t0 = time.perf_counter()
            try:
                cf(); torch.cuda.synchronize()
                cw = time.perf_counter() - t0
                o.zero_(); cf(); torch.cuda.synchronize()
                cerr = float((o - ref).abs().max())
                tc, iqc = evt(cf, 0, 12, flush)
            except Exception as ex:  # recorded, never fatal
                cw = -1.0; tc, iqc, cerr = None, None, "compile-fail:%r" % (ex,)
            t_oracle = min(times.values())
            gname = RED_VARIANT if fires else "torch"
            arm = dict(gate=dict(choice=gname, t=times[gname], probes=0.0,
                                 regret=(times[gname] - t_oracle) / t_oracle,
                                 err=errs.get(gname, 0.0)))
            order = ["torch"] + ["tsplit_" + n for n in red_order]
            for b in BUDGETS:
                cand = [c for c in order if c in pool][:min(b, len(order))]
                if b == 0:
                    ch, pc = "torch", 0.0
                else:
                    pt = {c: one_shot(pool[c], flush) for c in cand}
                    ch = min(pt, key=pt.get); pc = sum(pt.values())
                arm["b%d" % b] = dict(choice=ch, t=times[ch], probes=pc,
                                      regret=(times[ch] - t_oracle) / t_oracle,
                                      regret_total=(times[ch] + pc - t_oracle) / t_oracle,
                                      iqr=iqrs[ch])
            arm["compile"] = dict(t=tc, iqr=iqc, compile_wall_s=cw, err=cerr,
                                  regret=(None if tc is None else (tc - t_oracle) / t_oracle))
            row["t_oracle_ms"] = t_oracle; row["oracle_arg"] = min(times, key=times.get)
            row["arms"] = arm; row["errs"] = errs; row["n_reps"] = 12
            finish(row)
            print("red K%d M%d: oracle %.4f gate %s fires=%s bt=%.0f" %
                  (K, M, t_oracle, gname, fires, bt), flush=True)

    # ---------------- conv ----------------
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
        times, iqrs, errs = {}, {}, {}
        for nm, fn in pool.items():
            t, iq = evt(fn, 2, 12, flush)
            times[nm], iqrs[nm] = t, iq
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
            tc, iqc = evt(cfn, 0, 12, flush)  # includes output alloc (pool arms use prealloc)
        except Exception as ex:
            cw = -1.0; tc, iqc, cerr = None, None, "compile-fail:%r" % (ex,)
        t_oracle = min(times.values())
        fires = H <= CONV_H0 and C <= CONV_C0
        gname = CONV_VARIANT if (fires and CONV_VARIANT in pool) else "torch"
        arm = dict(gate=dict(choice=gname, t=times[gname], probes=0.0,
                             regret=(times[gname] - t_oracle) / t_oracle,
                             err=errs.get(gname, 0.0)))
        order = ["torch"] + conv_order
        for b in BUDGETS:
            cand = [c for c in order if c in pool][:min(b, len(order))]
            if b == 0:
                ch, pc = "torch", 0.0
            else:
                pt = {c: one_shot(pool[c], flush) for c in cand}
                ch = min(pt, key=pt.get); pc = sum(pt.values())
            arm["b%d" % b] = dict(choice=ch, t=times[ch], probes=pc,
                                  regret=(times[ch] - t_oracle) / t_oracle,
                                  regret_total=(times[ch] + pc - t_oracle) / t_oracle,
                                  iqr=iqrs[ch])
        arm["compile"] = dict(t=tc, iqr=iqc, compile_wall_s=cw, err=cerr,
                              regret=(None if tc is None else (tc - t_oracle) / t_oracle))
        row["t_oracle_ms"] = t_oracle; row["oracle_arg"] = min(times, key=times.get)
        row["arms"] = arm; row["errs"] = errs; row["n_reps"] = 12
        finish(row)
        print("conv N%d H%d C%d: oracle %.4f gate %s fires=%s" %
              (N, H, C, t_oracle, gname, fires), flush=True)

    # ---------------- pre-registered analysis ----------------
    summary = {"date": DATE, "pre_reg": "docstring 20:41Z 2026-10-18",
               "op_class": "reduction+conv", "rule": "FROZEN day-1 gate, fresh shapes",
               "domain_predicate": [RED_VARIANT, "M>=16,bt>=1024", CONV_VARIANT, "H<=14,C<=64"],
               "derived_coords": ["M", "bt_deployed", "H", "C", "visits(compiles)"],
               "boundary_ref": "ATLAS A1", "n_reps": 12,
               "grids": {"red_Ks": RED_KS, "red_Ms": RED_MS, "conv": CONV_POINTS},
               "families": {}}
    for famname in ("reduction", "conv"):
        rs = [r for r in rows if r["family"] == famname]
        if not rs:
            continue
        import math
        def mean(v):
            v = [x for x in v if x is not None]
            return sum(v) / len(v) if v else None
        def paired_se(a, b):
            d = [x - y for x, y in zip(a, b) if x is not None and y is not None]
            if len(d) < 2:
                return None
            m = sum(d) / len(d)
            return math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d))
        gt = [r["arms"]["gate"]["regret"] for r in rs]  # gate probes=0: regret==regret_total
        bt_ = {b: [r["arms"]["b%d" % b]["regret_total"] for r in rs] for b in (1, 2, 4, 8)}
        gd = [r["arms"]["gate"]["regret"] for r in rs]
        cd = [r["arms"]["compile"]["regret"] for r in rs]
        best_b, best_bt = None, None
        for b in (1, 2, 4, 8):
            m = mean(bt_[b])
            if m is not None and (best_bt is None or m < best_bt):
                best_b, best_bt = b, m
        gm = mean(gt)
        se = paired_se(gt, bt_[best_b]) if best_b else None
        fires = sum(1 for r in rs if r["gate_fires"]) / len(rs)
        # b1 always probes torch first -> b1.t IS torch delivery (order fixed)
        torch_deliv = [(r["arms"]["b1"]["t"] - r["t_oracle_ms"]) / r["t_oracle_ms"] for r in rs]
        se_b0 = paired_se(gt, torch_deliv)
        worse_torch = mean([1.0 if r["arms"]["gate"]["t"] > 1.01 * r["arms"]["b1"]["t"] else 0.0
                            for r in rs])
        worse_oracle = mean([1.0 if r["arms"]["gate"]["t"] > 1.01 * r["t_oracle_ms"] else 0.0
                             for r in rs])
        cwalls = [r["arms"]["compile"]["compile_wall_s"] for r in rs
                  if r["arms"]["compile"]["compile_wall_s"] and r["arms"]["compile"]["compile_wall_s"] > 0]
        summary["families"][famname] = dict(
            n=len(rs), gate_fires_frac=round(fires, 3),
            gate_regret_total=gm, best_b=best_b, best_b_regret_total=best_bt,
            gate_vs_best_b_SE=se,
            verdict=("GENERALIZES" if (gm is not None and best_bt is not None and
                                       se is not None and gm <= best_bt + se)
                     else "NO-GENERALIZATION (scope shrinks; hybrid candidate)"),
            vacuity_check=dict(gate_vs_torch_delivery_SE=se_b0, fires_frac=round(fires, 3)),
            gate_delivery_worse1pct_vs_torch=worse_torch,
            gate_delivery_worse1pct_vs_oracle=worse_oracle,
            compile_delivery_regret=mean(cd), compile_wall_s_mean=mean(cwalls),
            compile_wall_s_max=max(cwalls) if cwalls else None,
            gate_delivery_regret=mean(gd))

    json.dump({"meta": summary, "rows": rows}, open(OUT, "w"), indent=1)
    print(json.dumps(summary["families"], indent=1))
    print("wrote", OUT, len(rows), "rows")


if __name__ == "__main__":
    main()
