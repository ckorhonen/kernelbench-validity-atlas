"""M3: dispatcher-vs-budgeted-autotune curve (revival plan, decisive public object).

Pre-registered BEFORE evaluation 2026-10-18 ~10:30Z (this docstring; DIARY same tick,
post-reflection-#10). Questions:
  Q1  Does the zero-probe family-local dispatcher match budgeted per-shape autotune
      at equal-or-better TOTAL time (delivery + probe cost) on both families?
  Q2  At what probe budget b does budgeted autotune reach parity with the gate?

Arms (per shape, fresh timing this run; no artifact timings reused for decisions):
  gate        zero probes. reduction: hand iff M>=16 AND bt>=1024 with
              bt = 2*bytes/(ctas*128), ctas from the DEPLOYED variant tsplit_S64_KT64
              (cdiv(M,32)*64) -- the M1-deployed-geometry rule, variant frozen from
              w4_v1/M1. conv: hand iff H<=h0 AND C<=c0, (h0,c0,variant) fit HERE on the
              DAY-1 artifact only (w3_conv_gemm.json, full-data fit, ties->smaller hand
              region then fewer features); declared in-sample-on-grid, NOT a fresh-shape
              generalization claim (that is W4's held-out axis, separate).
  autotune(b) candidate pool per shape: [torch] + fixed-seed shuffle(hand variants)
              (seed 4321). b in {0,1,2,4,8} capped at pool size; b=0 -> torch, no probe.
              probes: SINGLE timed exec each, same flush protocol, charged in full
              (torch probed like any candidate when b>=1). choice = argmin probe time.
  oracle      min med-of-12 over the full pool (torch_cmp EXCLUDED everywhere: context
              arm per IDENTITY#3, different cost structure).

Metrics (per point, both families, pooled and per-family):
  regret_delivery = (t_chosen_med12 - t_oracle)/t_oracle
  regret_total    = (t_chosen_med12 + sum(probe times) - t_oracle)/t_oracle   [PRIMARY:
                   fresh-shape deployment view; each shape visited once]
Decision rule (declared): gate WINS a family iff mean regret_total(gate) <=
  min_b mean regret_total(autotune b) - pooled SE; parity budget b* = smallest b with
  mean regret_total(autotune) <= gate + pooled SE (SE of per-point diffs / sqrt(n)).
Falsifier (P5, written now): if median probe cost << median delivery-time differences
  (probe accounting immaterial), the gate's edge is a probe-cost artifact of this
  deployment view -- report and reframe the object as warm-cache steady-state, where
  only regret_delivery matters and b=0 autotune(=torch) is the only competitor.
Honesty: b caps at pool size (reduction 6 hand variants, conv 5) -- b=8 saturates;
  reported as cap. Correctness of delivered choice read fresh (e2_debug2 lesson).
Schema: P2 fields in meta + per-row variance/n_reps. One rerun allowed for harness
  bugs only.

Run: python3 m3_autotune_curve.py (on spark) -> results/m3_curve.json(.jsonl incr)
"""
import json, os, time, random
import torch
import torch.nn.functional as F
import triton
from e2_splitk_v3 import split_tiled, split_reduce
from w3_conv_gemm import conv_impgemm, VARIANTS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "m3_curve.json")
OUTL = OUT + ".jsonl"
DAY1_CONV = os.path.join(HERE, "results", "w3_conv_gemm.json")
DATE = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
FLUSH_BYTES = 256 * 2**20
BM = 32
RED_SKT = [("S%d_KT%d" % (s, kt), s, kt) for s in (4, 16, 64) for kt in (16, 64)]
BUDGETS = [0, 1, 2, 4, 8]
SEED = 4321


def evt(fn, warmup, reps, flush):
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
    return ts[n // 2], [ts[int(.25 * n)], ts[min(n - 1, int(.75 * n))]]


def one_shot(fn, flush):
    if flush is not None:
        flush.zero_()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record(); fn(); e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)


def fit_conv_gate():
    """(h0,c0,variant) full-data fit on the DAY-1 artifact; ties->smaller hand region."""
    day1 = json.load(open(DAY1_CONV))["rows"]
    pts = []
    for r in day1:
        v = r["variants"]
        if "med_ms" not in v["torch"]:
            continue
        hand = {k: x["med_ms"] for k, x in v.items() if k != "torch_cmp" and "med_ms" in x and k != "torch"}
        t_or = min(list(hand.values()) + [v["torch"]["med_ms"]])
        pts.append(dict(H=r["H"], C=r["C"], t_torch=v["torch"]["med_ms"], hand=hand, t_or=t_or))
    best = None
    for h0 in [None] + sorted({p["H"] for p in pts}):        # None = never-hand
        for c0 in sorted({p["C"] for p in pts}):
            for u in {k for p in pts for k in p["hand"]}:
                ok = True
                regs = []
                for p in pts:
                    fire = h0 is not None and p["H"] <= h0 and p["C"] <= c0
                    if fire and u not in p["hand"]:
                        ok = False; break
                    t = p["hand"][u] if fire else p["t_torch"]
                    regs.append((t - p["t_or"]) / p["t_or"])
                if not ok:
                    continue
                nhand = sum(1 for p in pts if h0 is not None and p["H"] <= h0 and p["C"] <= c0)
                key = (round(sum(regs) / len(regs), 9), nhand)
                if best is None or key < best[0]:
                    best = (key, dict(h0=h0, c0=c0, variant=u, mean_regret=key[0]))
    return best[1], len(pts)


def main():
    dev = torch.device("cuda:0")
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.float32, device=dev)
    conv_gate, nfit = fit_conv_gate()
    print("conv gate from day-1 fit:", conv_gate, "n=", nfit)

    # reduction grid (v3): K x M
    Ks = [2 ** k for k in (12, 14, 16, 18, 20, 22)]
    Ms = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    rng = random.Random(SEED)
    red_order = [n for n, _, _ in RED_SKT]; rng.shuffle(red_order)
    conv_order = [v[0] for v in VARIANTS]; rng.shuffle(conv_order)

    rows = []
    done = set()
    if os.path.exists(OUTL):
        for line in open(OUTL):
            line = line.strip()
            if line:
                r = json.loads(line)
                k = (r["family"], r.get("K", r.get("N")), r.get("M", r.get("H")), r.get("C"))
                done.add(k); rows.append(r)
    outl = open(OUTL, "a")

    def finish_row(row):
        row["date"] = DATE
        rows.append(row); outl.write(json.dumps(row) + "\n"); outl.flush()

    # ---------------- reduction family ----------------
    Xmax = max(Ks) * max(Ms)
    X = torch.randn(Xmax, dtype=torch.float32, device=dev)
    P = torch.empty(64 * max(Ms) * 2, dtype=torch.float32, device=dev)
    for K in Ks:
        for M in Ms:
            if ("reduction", K, M, None) in done:
                continue
            x = X[:K * M].view(K, M).contiguous()
            o = torch.empty(M, dtype=torch.float32, device=dev)
            ref = torch.sum(x, dim=0)
            row = dict(family="reduction", K=K, M=M, bytes=K * M * 4 + M * 4,
                       domain_pred="M>=16 and bt(deployed S64KT64)>=1024", variants={})
            pool = {"torch": (lambda: torch.sum(x, dim=0, out=o))}
            ctas = ((M + BM - 1) // BM) * 64
            bt = 2 * row["bytes"] / (ctas * 128)
            gate_fires = M >= 16 and bt >= 1024
            for name, S, KT in RED_SKT:
                if K // S < KT:
                    continue
                gm = triton.cdiv(M, BM)

                def run(S=S, KT=KT, gm=gm, x=x, P=P, K=K, M=M, o=o):
                    split_tiled[(gm, S)](x, P, K, M, S, BM=BM, KT=KT, num_warps=4)
                    split_reduce[(triton.cdiv(M, 256),)](P, o, M, S, BM=256, num_warps=4)
                pool["tsplit_" + name] = run
            times, iqrs = {}, {}
            for nm, fn in pool.items():
                t, iq = evt(fn, 2, 12, flush)
                times[nm], iqrs[nm] = t, iq
            torch.cuda.synchronize()
            err_gate = None
            errs = {}
            for nm, fn in pool.items():
                if nm.startswith("tsplit"):
                    fn(); torch.cuda.synchronize()
                    e = float((o - ref).abs().max())
                    errs[nm] = e
                    if nm == "tsplit_S64_KT64":
                        err_gate = e
            row["errs"] = errs
            t_oracle = min(times.values())
            # gate
            gname = "tsplit_S64_KT64" if gate_fires else "torch"
            # autotune
            arm = dict(gate=dict(choice=gname, t=times[gname], probes=0,
                                 regret=(times[gname] - t_oracle) / t_oracle,
                                 err=err_gate if gate_fires else 0.0))
            order = ["torch"] + ["tsplit_" + n for n in red_order]
            for b in BUDGETS:
                if b == 0:
                    ch, pc = "torch", 0.0
                else:
                    cand = [c for c in order if c in pool][:min(b, len(order))]
                    pt = {c: one_shot(pool[c], flush) for c in cand}
                    ch = min(pt, key=pt.get); pc = sum(pt.values())
                arm["b%d" % b] = dict(choice=ch, t=times[ch], probes=pc,
                                      regret=(times[ch] - t_oracle) / t_oracle,
                                      regret_total=(times[ch] + pc - t_oracle) / t_oracle,
                                      iqr=iqrs[ch])
            row["t_oracle_ms"] = t_oracle; row["oracle_arg"] = min(times, key=times.get)
            row["bt_deployed"] = bt; row["gate_fires"] = gate_fires
            row["arms"] = arm; row["n_reps"] = 12
            finish_row(row)
            print("red K%d M%d: oracle %.4f gate %s" % (K, M, t_oracle, gname))

    # ---------------- conv family ----------------
    HS = [4, 7, 10, 14, 20, 28]; NS = [1, 4]; CS = [16, 64, 256]
    for H in HS:
        for N in NS:
            for C in CS:
                if ("conv", N, H, C) in done:
                    continue
                torch.manual_seed(1234 + H * 100 + N * 10 + C)
                x = torch.randn(N, C, H, H, device=dev)
                w = torch.randn(C, C, 3, 3, device=dev)
                ref = F.conv2d(x, w, padding=1)
                M = N * H * H
                row = dict(family="conv", N=N, H=H, C=C, M=M,
                           domain_pred="H<=h0 and C<=c0 (day-1 fit)", variants={})
                pool = {"torch": (lambda: F.conv2d(x, w, padding=1))}
                bufs = {}
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
                    if nm in bufs:  # fresh-read correctness (e2_debug2 lesson)
                        bufs[nm].zero_(); fn(); torch.cuda.synchronize()
                        errs[nm] = float((bufs[nm] - ref).abs().max())
                    else:
                        errs[nm] = 0.0
                gname = conv_gate["variant"]
                t_oracle = min(times.values())
                gate_fires = conv_gate["h0"] is not None and H <= conv_gate["h0"] and C <= conv_gate["c0"]
                gchoice = gname if (gate_fires and gname in pool) else "torch"
                arm = dict(gate=dict(choice=gchoice, t=times[gchoice], probes=0,
                                     regret=(times[gchoice] - t_oracle) / t_oracle))
                order = ["torch"] + conv_order
                for b in BUDGETS:
                    if b == 0:
                        ch, pc = "torch", 0.0
                    else:
                        cand = [c for c in order if c in pool][:min(b, len(order))]
                        pt = {c: one_shot(pool[c], flush) for c in cand}
                        ch = min(pt, key=pt.get); pc = sum(pt.values())
                    arm["b%d" % b] = dict(choice=ch, t=times[ch], probes=pc,
                                          regret=(times[ch] - t_oracle) / t_oracle,
                                          regret_total=(times[ch] + pc - t_oracle) / t_oracle,
                                          iqr=iqrs[ch])
                row["t_oracle_ms"] = t_oracle; row["oracle_arg"] = min(times, key=times.get)
                row["gate_fires"] = bool(gate_fires); row["arms"] = arm; row["n_reps"] = 12
                row["conv_gate"] = conv_gate
                finish_row(row)
                print("conv N%d H%d C%d: oracle %.4f gate %s" % (N, H, C, t_oracle, gchoice))

    json.dump({"meta": {"date": DATE, "sm": sm,
                        "pre_reg": "see m3_autotune_curve.py docstring 10:30Z 2026-10-18",
                        "budgets": BUDGETS, "seed": SEED, "conv_gate_fit": conv_gate,
                        "op_class": "reduction+conv", "rule": "family-local zero-probe gate",
                        "derived_coords": ["M", "bt_deployed", "H", "C"]}, "rows": rows},
              open(OUT, "w"), indent=1)
    print("wrote", OUT, len(rows))


if __name__ == "__main__":
    main()
