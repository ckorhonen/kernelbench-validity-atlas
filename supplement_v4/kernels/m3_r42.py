"""M3-R42: contract-enforced rerun of the m3 economics grid (rejection O4hnCx, R2').

Repair: m3_curve chose gate/autotune/oracle arms by timing with no correctness
contract; the repair showed the TF32 hand arms fail the 1e-4 contract on the
delivered reference, so the published crossover is not economics of
correctness-preserving deployment. This rerun repeats the EXACT m3 grids, seed,
budgets and decision rule (m3_autotune_curve.py, rev 423217d) with
validate-then-time arms (b1_gatezero_v2 / r1 pattern):

  - every arm is timed med-of-12 AND checked once, fresh buffer, read immediately,
    ALLCLOSE(ref, cand, rtol=1e-4, atol=1e-4) -- r1 CONTRACT verbatim; ref = the
    deployed reference op (F.conv2d default flags / torch.sum), the deployed
    reality per the repair;
  - conv hand arms are (variant, precision) pairs via r1.conv_impgemm_ip
    (ieee, tf32), exactly the r1 arm set; err_vs_fp32ref recorded secondary;
  - admissible set per row = hand arms whose check passes, plus torch (the
    reference op, admitted by construction -- the r42_analysis corrected-oracle
    convention, min(passing hand, torch));
  - gate: published structural rule unchanged (reduction M>=16 & bt>=1024 ->
    tsplit_S64_KT64; conv day-1 fit H<=h0 & C<=c0 -> fitted variant). Delivered
    arm = rule variant in its FIRST PASSING precision (ieee then tf32) when the
    rule fires, else torch; the (variant, precision) chain is recorded per row.
    The conv gate fit is the SAME day-1 artifact fit (declared in-sample);
  - autotune(b): candidate order [torch] + seed-4321 shuffle over hand arms
    (order fixed per family as in m3), probed ONLY if admissible --
    validate-then-time: a failing arm cannot be chosen. b in {0,1,2,4,8} capped
    at the row's admissible pool size; probes are single timed execs, charged
    in full;
  - oracle = min med12 over {torch} U passing hand arms.

Certification status, declared: offline on the deployed domain -- each grid
point's admissibility is certified by the recorded check BEFORE selection; the
deployed gate/autotuner reads the recorded verdict (the 'offline certification'
reading of review item 2; the alternative, charging validation at runtime, is
reported as the probe-cost sensitivity row adm_probe_ms). NO claim here extends
past the measured warm accounting; cold start stays a limitation (paper §4).

Prereg status: this is the declared replacement-probe run of triage disposition
R2' (rejection_triage_O4hnCx.md, 2026-10-22); grids/seed/budgets/decision rule
inherit m3 unchanged -- no new design, one rerun for the contract, P5-clean.

Run on spark: python3 m3_r42.py            # full grid, resumable via jsonl
               SMOKE=1 python3 m3_r42.py   # 1 row/family, harness check only
Output: results/m3_r42.jsonl; summary written by m3_r42_analysis.py.
"""
import json, os, random, time
import torch
import torch.nn.functional as F
import triton

import m3_autotune_curve as m3
import r1_conv_gated as r1
from r1_conv_gated import CONTRACT, PRECS, conv_impgemm_ip
from e2_splitk_v3 import split_tiled, split_reduce
from w3_conv_gemm import VARIANTS

HERE = os.path.dirname(os.path.abspath(__file__))
OUTL = os.path.join(HERE, "results", "m3_r42.jsonl")
DATE = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
SMOKE = os.environ.get("SMOKE") == "1"
FLUSH_BYTES = 256 * 2**20


def main():
    dev = torch.device("cuda:0")
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.float32, device=dev)
    conv_gate, nfit = m3.fit_conv_gate()
    print("conv gate from day-1 fit:", conv_gate, "n=", nfit)

    Ks = [2 ** k for k in (12, 14, 16, 18, 20, 22)]
    Ms = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    if SMOKE:
        Ks, Ms = Ks[:2], Ms[:2]
    rng = random.Random(m3.SEED)
    red_order = ["tsplit_" + n for n, _, _ in m3.RED_SKT]; rng.shuffle(red_order)
    conv_order = ["%s|%s" % (v[0], p) for v in VARIANTS for p in PRECS]
    rng.shuffle(conv_order)

    rows = []
    done = set()
    if os.path.exists(OUTL) and not SMOKE:
        for line in open(OUTL):
            line = line.strip()
            if line:
                r = json.loads(line)
                done.add((r["family"], r.get("K", r.get("N")), r.get("M", r.get("H")), r.get("C")))
        rows.extend(json.loads(l) for l in open(OUTL) if l.strip())
    outl = open(OUTL, "a")

    def finish_row(row):
        row["date"] = DATE
        rows.append(row); outl.write(json.dumps(row) + "\n"); outl.flush()

    def check(fn, out, ref):
        out.zero_(); fn(); torch.cuda.synchronize()
        err = float((out - ref).abs().max().item())
        return err, bool(torch.allclose(ref, out, **CONTRACT))

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
                       domain_pred="M>=16 and bt(deployed S64KT64)>=1024",
                       cert="offline-on-grid")
            pool, outs = {}, {}
            ctas = ((M + m3.BM - 1) // m3.BM) * 64
            bt = 2 * row["bytes"] / (ctas * 128)
            gate_fires = M >= 16 and bt >= 1024
            for name, S, KT in m3.RED_SKT:
                if K // S < KT:
                    continue
                gm = triton.cdiv(M, m3.BM)
                out = torch.empty_like(o); outs["tsplit_" + name] = out

                def run(S=S, KT=KT, gm=gm, x=x, P=P, K=K, M=M, out=out):
                    split_tiled[(gm, S)](x, P, K, M, S, BM=m3.BM, KT=KT, num_warps=4)
                    split_reduce[(triton.cdiv(M, 256),)](P, out, M, S, BM=256, num_warps=4)
                pool["tsplit_" + name] = run
            pool["torch"] = (lambda o=o: torch.sum(x, dim=0, out=o))
            outs["torch"] = o
            times, iqrs, adm = {}, {}, {}
            for nm, fn in pool.items():
                t, iq = m3.evt(fn, 2, 12, flush)
                times[nm], iqrs[nm] = t, iq
                e, ok = check(fn, outs[nm], ref)
                adm[nm] = dict(err=e, passed=ok)
            t_oracle = min([times[t] for t in ("torch",) + tuple(k for k in pool if k != "torch" and adm[k]["passed"])])
            o_arg = min((("torch", times["torch"]),) + tuple((k, times[k]) for k in pool if k != "torch" and adm[k]["passed"]), key=lambda kv: kv[1])
            gname = "tsplit_S64_KT64" if (gate_fires and adm["tsplit_S64_KT64"]["passed"]) else "torch"
            arm = dict(gate=dict(choice=gname, t=times[gname], probes=0,
                                 regret=(times[gname] - t_oracle) / t_oracle,
                                 err=adm.get(gname, {}).get("err"), passed=adm.get(gname, {}).get("passed"),
                                 chain=[gname] if gname != "torch" else []))
            order = ["torch"] + red_order
            for b in m3.BUDGETS:
                if b == 0:
                    ch, pc = "torch", 0.0
                else:
                    cand = [c for c in order if c in pool and (c == "torch" or adm[c]["passed"])][:b]
                    pt = {c: m3.one_shot(pool[c], flush) for c in cand}
                    ch = min(pt, key=pt.get); pc = sum(pt.values())
                arm["b%d" % b] = dict(choice=ch, t=times[ch], probes=pc,
                                      regret=(times[ch] - t_oracle) / t_oracle,
                                      regret_total=(times[ch] + pc - t_oracle) / t_oracle,
                                      iqr=iqrs[ch])
            row.update(t_oracle_ms=t_oracle, oracle_arg=o_arg, bt_deployed=bt,
                       gate_fires=bool(gate_fires), adm=adm, arms=arm, n_reps=12)
            finish_row(row)
            print("red K%d M%d: oracle %.4f (%s) gate %s adm %d/%d" %
                  (K, M, t_oracle, o_arg, gname, sum(1 for v in adm.values() if v["passed"]), len(adm)))

    # ---------------- conv family ----------------
    HS = [4, 7, 10, 14, 20, 28]; NS = [1, 4]; CS = [16, 64, 256]
    if SMOKE:
        HS, NS, CS = HS[:2], NS[:1], CS[:2]
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
                _sv = torch.backends.cudnn.allow_tf32
                torch.backends.cudnn.allow_tf32 = False
                ref32 = F.conv2d(x, w, padding=1)
                torch.backends.cudnn.allow_tf32 = _sv
                row = dict(family="conv", N=N, H=H, C=C, M=M,
                           domain_pred="H<=h0 and C<=c0 (day-1 fit)",
                           cert="offline-on-grid", conv_gate=conv_gate)
                pool, outs, meta = {}, {}, {}
                pool["torch"] = (lambda: F.conv2d(x, w, padding=1))
                outs["torch"] = torch.empty_like(ref)
                for name, BM, BN, BK, nw in VARIANTS:
                    grid = (triton.cdiv(M, BM) * triton.cdiv(C, BN),)
                    for prec in PRECS:
                        key = "%s|%s" % (name, prec)
                        out = torch.empty_like(ref); outs[key] = out

                        def go(BM=BM, BN=BN, BK=BK, nw=nw, prec=prec, out=out, grid=grid):
                            conv_impgemm_ip[grid](x, w, out, N, H, H, C, C,
                                                  BM=BM, BN=BN, BK=BK, IP=prec, num_warps=nw)
                        pool[key] = go
                        meta[key] = dict(variant=name, precision=prec, ctas=grid[0],
                                         waves=grid[0] / sm, warps=nw)
                times, iqrs, adm = {}, {}, {}
                for nm, fn in pool.items():
                    t, iq = m3.evt(fn, 2, 12, flush)
                    times[nm], iqrs[nm] = t, iq
                    e, ok = check(fn, outs[nm], ref)
                    adm[nm] = dict(err=e, passed=ok)
                    if nm != "torch":
                        adm[nm]["err_vs_fp32ref"] = float((outs[nm] - ref32).abs().max().item())
                passers = [k for k in pool if k != "torch" and adm[k]["passed"]]
                t_oracle = min([times["torch"]] + [times[k] for k in passers])
                o_arg = min([("torch", times["torch"])] + [(k, times[k]) for k in passers], key=lambda kv: kv[1])
                gf = conv_gate["h0"] is not None and H <= conv_gate["h0"] and C <= conv_gate["c0"]
                gkey = "%s|ieee" % conv_gate["variant"]
                if gf:
                    chain = []
                    gchoice = "torch"
                    for prec in PRECS:
                        k = "%s|%s" % (conv_gate["variant"], prec)
                        if k in pool and adm[k]["passed"]:
                            gchoice, chain = k, [k]; break
                    if not chain:
                        chain = ["torch(fallback)"]
                else:
                    gchoice, chain = "torch", []
                arm = dict(gate=dict(choice=gchoice, t=times[gchoice], probes=0,
                                     regret=(times[gchoice] - t_oracle) / t_oracle,
                                     err=adm[gchoice]["err"], passed=adm[gchoice]["passed"],
                                     chain=chain))
                order = ["torch"] + conv_order
                npass = len(passers)
                for b in m3.BUDGETS:
                    if b == 0:
                        ch, pc = "torch", 0.0
                    else:
                        cand = [c for c in order if c in pool and (c == "torch" or adm[c]["passed"])][:b]
                        pt = {c: m3.one_shot(pool[c], flush) for c in cand}
                        ch = min(pt, key=pt.get); pc = sum(pt.values())
                    arm["b%d" % b] = dict(choice=ch, t=times[ch], probes=pc,
                                          regret=(times[ch] - t_oracle) / t_oracle,
                                          regret_total=(times[ch] + pc - t_oracle) / t_oracle,
                                          iqr=iqrs[ch])
                row.update(t_oracle_ms=t_oracle, oracle_arg=o_arg, gate_fires=bool(gf),
                           adm=adm, arm_meta=meta, arms=arm, n_reps=12)
                finish_row(row)
                print("conv N%d H%d C%d: oracle %.4f (%s) gate %s adm %d/%d" %
                      (N, H, C, t_oracle, o_arg, gchoice, 1 + npass, len(pool)))

    json.dump({"meta": {"date": DATE, "sm": sm,
                        "pre_reg": "repair rerun of m3 grid per rejection_triage_O4hnCx R2'; grids/seed/budgets/decision rule = m3_autotune_curve.py 2026-10-18 unchanged",
                        "budgets": m3.BUDGETS, "seed": m3.SEED, "conv_gate_fit": conv_gate,
                        "contract": r1.CONTRACT, "cert": "offline-on-grid",
                        "op_class": "reduction+conv", "rule": "family-local zero-probe gate, delivered arm = first passing precision",
                        "derived_coords": ["M", "bt_deployed", "H", "C"]},
               "rows": rows},
              open(os.path.join(HERE, "results", "m3_r42.json"), "w"), indent=1)
    print("wrote m3_r42 rows:", len(rows))


if __name__ == "__main__":
    main()
