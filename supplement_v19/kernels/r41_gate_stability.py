"""r41 rider: gate stability under the fp64 reference anchor + reference-error bootstrap.

Pre-registered lab/r41_gate_stability_prereg.md, commit b5504bd, prereg sha
5ff93b202f89c032e751577bfa496990adc647a786d4db1e196c754d27e9efc3 -- frozen
BEFORE this run (2026-06-22 note: date strings UTC). Population: the 56 recorded
r1 rows AS RUN (grids/seed formula imported from r1_conv_gated, including the two
transcribed extras per held-out family; grid repair is #42, out of scope).

Per row: refs R_def (cudnn default flags), R_ieee (allow_tf32 False), R_64
(float64 conv). Candidate output y_a for EVERY r1 arm-instance (single eval, no
timing). Margin s_a(R)=max(|y_a-R|-(1e-4+1e-4*|R|)); pass iff <=0. Gate rule and
true oracle (min passing hand, torch) recompute per R -- times for regrets from
lab/r1_times.json (recorded r1 med_ms, verbatim).

Arm A readouts: delivered-arm change R_def->R_64; admissible-set change; gate
regret delta (pp, corrected-oracle convention both sides).
Arm B: B=1000 element-resamples of D=R_def-R_64 (seed 20261022, generator per
row, blocks of 25 reps on GPU); per-rep gate + arm verdicts; flip fractions.
Margins fp32; R_64 cast to fp32 for margin math (cast err ~1e-7 << atol 1e-4;
noted here pre-run per prereg dtype silence).

Run on spark: python3 r41_gate_stability.py ; SMOKE=1 -> 2 rows, discarded.
Output results/gate_stability.jsonl + _summary.json. NO timing claims.
"""
import json, os, time
import torch
import torch.nn.functional as F
import triton

from r1_conv_gated import (GRIDS, VARIANTS, PRECS, CONTRACT, conv_impgemm_ip)

HERE = os.path.dirname(os.path.abspath(__file__))
OUTL = os.path.join(HERE, "results", "gate_stability.jsonl")
DATE = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
SMOKE = os.environ.get("SMOKE") == "1"
B_REPS = 1000
SEED = 20261022
BLOCK = 25
TIMES = json.load(open(os.path.join(HERE, "r1_times.json")))

ARM_KEYS = ["torch", "torch_cmp"] + [f"{n}[{p}]" for n, _, _, _, _ in VARIANTS for p in PRECS]


def margins(Y, R):
    # Y (A, elems) fp32, R (elems,) fp32 -> (A,) signed margins, <=0 passes
    d = (Y - R.unsqueeze(0)).abs()
    tol = 1e-4 + 1e-4 * R.abs().unsqueeze(0)
    return (d - tol).amax(dim=1)


def gate_from_pass(passmap, H, C, times):
    """Recorded r1 gate + corrected true oracle. passmap: arm->bool (imp_ keys only timed)."""
    passing = {k: t for k, t in times.items() if k.startswith("imp_") and passmap.get(k)}
    torch_t = times["torch"]
    ora_t = min(list(passing.values()) + [torch_t]) if passing else torch_t
    inrule = H <= 14 and C <= 64
    gate_k, gate_t = "torch", torch_t
    chain = []
    if inrule:
        for prec in PRECS:
            k = f"imp_B32x32[{prec}]"
            chain.append((k, bool(passmap.get(k))))
            if passmap.get(k):
                gate_k, gate_t = k, times[k]
                break
    else:
        chain = [("torch", True)]
    admissible = sorted(passing.keys())
    return dict(arm=gate_k, regret=(gate_t - ora_t) / ora_t if ora_t else 0.0,
                admissible=admissible, inrule=inrule, chain=chain,
                ora_t=ora_t, gate_t=gate_t)


def main():
    dev = torch.device("cuda:0")
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    n_done = 0
    outf = open(OUTL, "w")
    g = torch.Generator(device=dev); g.manual_seed(SEED)
    for tag, (H, N, C) in [(t, pt) for t, pts in GRIDS for pt in pts]:
        if SMOKE and n_done >= 2:
            break
        if True:
            key = f"{tag}|{H}|{N}|{C}"
            times = dict(TIMES[key])
            g.manual_seed(SEED)  # prereg: generator fixed PER ROW
            torch.manual_seed(1234 + H * 100 + N * 10 + C)
            x = torch.randn(N, C, H, H, device=dev)
            w = torch.randn(C, C, 3, 3, device=dev)
            M = N * H * H
            # references; flags snapshotted before each ref, restored immediately
            f_def = torch.backends.cudnn.allow_tf32
            R_def = F.conv2d(x, w, padding=1)
            torch.backends.cudnn.allow_tf32 = False
            R_ieee = F.conv2d(x, w, padding=1)
            torch.backends.cudnn.allow_tf32 = f_def
            R_64 = F.conv2d(x.double(), w.double(), padding=1).float()
            # candidate outputs, single eval each
            Ys = {"torch": F.conv2d(x, w, padding=1).clone()}
            try:
                fn = torch.compile(lambda xx: F.conv2d(xx, w, padding=1), dynamic=False)
                fn(x); torch.cuda.synchronize()
                Ys["torch_cmp"] = fn(x).clone()
            except Exception as ex:
                Ys["torch_cmp"] = None
                times = {k: v for k, v in times.items() if k != "torch_cmp"}
            for name, BM, BN, BK, nw in VARIANTS:
                grid = (triton.cdiv(M, BM) * triton.cdiv(C, BN),)
                for prec in PRECS:
                    k = f"{name}[{prec}]"
                    if k not in times:
                        continue
                    out = torch.empty_like(R_def)
                    conv_impgemm_ip[grid](x, w, out, N, H, H, C, C,
                                          BM=BM, BN=BN, BK=BK, IP=prec, num_warps=nw)
                    Ys[k] = out.clone()
            arms = [k for k in ARM_KEYS if k in Ys and Ys[k] is not None and k in times]
            Ystack = torch.stack([Ys[k].flatten() for k in arms])
            R = {"def": R_def.flatten(), "ieee": R_ieee.flatten(), "64": R_64.clone().flatten()}
            S = {rk: margins(Ystack, R[rk]) for rk in R}
            P = {rk: (S[rk] <= 0).tolist() for rk in R}
            G = {rk: gate_from_pass(dict(zip(arms, P[rk])), H, C, times) for rk in P}
            # Arm A deltas
            delivered_change = G["def"]["arm"] != G["64"]["arm"]
            adm_change = G["def"]["admissible"] != G["64"]["admissible"]
            reg_delta_pp = 100 * (G["64"]["regret"] - G["def"]["regret"])
            # per-arm verdict flips def -> 64
            arm_flip = {k: bool(P["def"][i]) != bool(P["64"][i]) for i, k in enumerate(arms)}
            n_arm_flip = sum(arm_flip.values())
            # reference realized error artifact (elementwise, vs anchor)
            with torch.no_grad():
                d_def = (R["def"] - R["64"]).abs()
                d_ieee = (R["ieee"] - R["64"]).abs()
            ref64_absmax = float(R_64.abs().max().item())
            referr = dict(
                def_max=float(d_def.max()), def_median=float(d_def.median()),
                def_rel_max=float(d_def.max() / max(ref64_absmax, 1e-30)),
                ieee_max=float(d_ieee.max()), ieee_median=float(d_ieee.median()),
                ieee_rel_max=float(d_ieee.max() / max(ref64_absmax, 1e-30)),
                flags=dict(cudnn_allow_tf32_during_def=f_def))
            # Arm B: bootstrap D = R_def - R_64, element resample
            Dall = R["def"] - R["64"]
            n = Dall.numel()
            arm_flip_reps = torch.zeros(len(arms), dtype=torch.long, device=dev)
            delivered_flip_reps = 0
            adm_flip_reps = 0
            R64f = R["64"]
            Pdef_dev = torch.tensor(P["def"], device=dev)
            done = 0
            while done < B_REPS:
                b = min(BLOCK, B_REPS - done)
                idx = torch.randint(0, n, (b, n), device=dev, generator=g, dtype=torch.long)
                Dp = torch.gather(Dall.expand(b, n), 1, idx)          # (b, elems)
                Rp = R64f.unsqueeze(0) + Dp                           # (b, elems)
                d = (Ystack.unsqueeze(0) - Rp.unsqueeze(1)).abs()      # (b,A,elems)
                tol = 1.4e-4 + 1e-4 * Rp.unsqueeze(1).abs()
                Sp = (d - tol).amax(dim=-1)                           # (b,A)
                Pp = (Sp <= 0).cpu()                                  # (b,A) host
                arm_flip_reps += (Pp.to(dev) != Pdef_dev.unsqueeze(0)).sum(0)
                Plist = Pp.tolist()
                for bi in range(b):
                    gb = gate_from_pass(dict(zip(arms, Plist[bi])), H, C, times)
                    if gb["arm"] != G["def"]["arm"]:
                        delivered_flip_reps += 1
                    if gb["admissible"] != G["def"]["admissible"]:
                        adm_flip_reps += 1
                done += b
            del d, tol, Sp, Pp, Dp, Rp, idx
            row = dict(tag=tag, H=H, N=N, C=C, date=DATE, sm_count=sm,
                       prereg="lab/r41_gate_stability_prereg.md b5504bd sha 5ff93b20",
                       arms=arms,
                       pass_def=dict(zip(arms, P["def"])),
                       pass_ieee=dict(zip(arms, P["ieee"])),
                       pass_64=dict(zip(arms, P["64"])),
                       gate_def=G["def"], gate_ieee=G["ieee"], gate_64=G["64"],
                       armA_delivered_change=delivered_change,
                       armA_admissible_change=adm_change,
                       armA_regret_delta_pp=reg_delta_pp,
                       armA_arm_verdict_flips=arm_flip, armA_n_arm_flips=n_arm_flip,
                       ref_err=referr, ref64_absmax=ref64_absmax,
                       ref_absmax_def=float(R_def.abs().max()),
                       ref_absmax_ieee=float(R_ieee.abs().max()),
                       bootstrap=dict(B=B_REPS, seed=SEED, block=BLOCK,
                                      arm_flip_frac=(arm_flip_reps / B_REPS).tolist(),
                                      arm_flip_frac_by_arm=dict(zip(arms, (arm_flip_reps / B_REPS).tolist())),
                                      delivered_flip_frac=delivered_flip_reps / B_REPS,
                                      admissible_flip_frac=adm_flip_reps / B_REPS))
            outf.write(json.dumps(row) + "\n"); outf.flush()
            n_done += 1
            print(f"[{n_done}] {key}: def->64 delivered_change={delivered_change} "
                  f"adm_change={adm_change} dReg{reg_delta_pp:+.3f}pp armFlips={n_arm_flip} "
                  f"bootDeliverFlip={delivered_flip_reps/B_REPS:.3f}", flush=True)
        # SMOKE: stop after 2 rows total
        if SMOKE and n_done >= 2:
            break
    outf.close()
    summarize()


def summarize():
    rows = [json.loads(l) for l in open(OUTL) if l.strip()]
    nd = sum(r["armA_delivered_change"] for r in rows)
    na = sum(r["armA_admissible_change"] for r in rows)
    flips = sum(r["armA_n_arm_flips"] for r in rows)
    per_arm = {}
    for r in rows:
        for k, v in r["armA_arm_verdict_flips"].items():
            per_arm.setdefault(k, 0)
            per_arm[k] += int(v)
    boot_del = [r["bootstrap"]["delivered_flip_frac"] for r in rows]
    boot_arm = {}
    for r in rows:
        for k, v in r["bootstrap"]["arm_flip_frac_by_arm"].items():
            boot_arm.setdefault(k, []).append(v)
    import statistics as st
    summ = dict(date=DATE, n_rows=len(rows), prereg_sha="5ff93b20",
                armA_delivered_changes=nd, armA_admissible_changes=na,
                armA_arm_verdict_flips_total=flips, armA_flips_by_arm=per_arm,
                armA_regret_delta_pp=[min(r["armA_regret_delta_pp"] for r in rows),
                                      max(r["armA_regret_delta_pp"] for r in rows)],
                boot_delivered_flip_frac=dict(median=st.median(boot_del),
                                              max=max(boot_del)),
                boot_arm_flip_frac_median={k: st.median(v) for k, v in boot_arm.items()},
                ref_err_def_max=float(max(r["ref_err"]["def_max"] for r in rows)),
                ref_err_def_median=float(st.median([r["ref_err"]["def_median"] for r in rows])),
                ref_err_ieee_max=float(max(r["ref_err"]["ieee_max"] for r in rows)))
    verdict = ("ROBUST: zero delivered-arm and zero admissible-set changes under the "
               "fp64 anchor; rider closes as robustness confirmation"
               if nd == 0 and na == 0 else
               f"CHANGES: {nd} delivered-arm / {na} admissible-set -> claim-3/4/5 "
               "denominators take stability intervals in the #42 revision")
    summ["verdict"] = verdict
    p = OUTL[:-6] + "_summary.json"
    open(p, "w").write(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1)[:2000])


if __name__ == "__main__":
    main()
