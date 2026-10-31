"""s63 -- paired engine observation + dual-predicate certificate for the reference-swap
population. Repairs rejection findings F1 and F3 of YZfYBWJ5dIFgQZqg (2026-10-30T21:03Z).

PRE-REGISTRATION (frozen BEFORE first run; this docstring is the registration).

F1 objection: the paper attributes 11/56 delivered-arm changes to an "engine-class swap
alone" (default dispatch -> fp32-SIMT route), but r41 only flipped cudnn.allow_tf32 and
never profiled which engines ran. Repair options named by the reviewer: (a) report the
supported precision-configuration intervention only, or (b) supply paired engine
observations for the relevant rows and distinguish changed precision from changed
algorithm family. This script does (b): it profiles the reference conv call itself under
the three configurations on the same 56 rows, so the engine label is an observation, not
an inference from Table 1.

F3 objection: r41's swap checks use the reference-scaled predicate
max(|Y-R|-(1e-4+1e-4|R|))<=0 while the benchmark's evaluator scales rtol by the CANDIDATE
(eval.py:804, verified 10-21). This script records both margins per (row, arm, reference),
so the swap labels can be certified evaluator-exact or restated as a different predicate.

Declared reads (two-class, written before the run; P5 -- either outcome is a result):
  R1 (engine): for the rows where the delivered arm changes under R_def->R_ieee, is the
     reference's observed engine class different between R_def and R_ieee? If yes on a
     majority, the "engine-class swap" is a measured engine transition and the paper may
     say so for those rows; if no, the claim narrows to a precision-configuration
     intervention with engine class unchanged (and the 11/56 becomes a precision effect).
     Per-row labels are the result; no aggregate threshold is used to decide wording.
  R2 (predicate): does any (row, arm) pass label differ between the reference-scaled and
     candidate-scaled predicates, on any of the three references? Count reported per
     reference and per arm class; if zero, the swap labels are predicate-invariant on this
     population and the evaluator-exact claim holds for these rows; if nonzero, the swap
     numbers must be restated under the evaluator predicate and both sets published.

Population: the 56 r1 conv rows, grids verbatim from r1_conv_gated.GRIDS, seed formula
1234 + H*100 + N*10 + C (the same inputs r41 used), C=CO, 3x3 s1 p1, fp32.
Per row, three references: R_def (cudnn defaults), R_ieee (cudnn.allow_tf32=False),
R_64 (conv2d in float64, cast to fp32 for margin math, cast err ~1e-7 << atol).
Arms: the 12 r1 arm-instances (torch, torch_cmp, 5 variants x {ieee,tf32}).

Recorded per (row, ref, arm): maxabs err, |Y|max, |R|max, margin_ref (signed), margin_cand
(signed), pass_ref, pass_cand, ctas/waves/warps. Per (row, ref): raw top kernel names with
device times and the classified engine label. Per row: gate arm and admissible sets under
both predicates for each reference, delivered-arm change, delivery-status change.

Run on spark: python3 s63_engine_predicate.py            # full
               SMOKE=1 python3 s63_engine_predicate.py   # 2 rows, harness check
Output: results/s63_engine_predicate.jsonl + results/s63_engine_predicate_summary.json
No timing claims in this artifact (profiler-instrumented runs only).
"""
import json, os, re, time
import torch
import torch.nn.functional as F
import triton

from r1_conv_gated import GRIDS, VARIANTS, PRECS, CONTRACT, conv_impgemm_ip

HERE = os.path.dirname(os.path.abspath(__file__))
OUTL = os.path.join(HERE, "results", "s63_engine_predicate.jsonl")
SUML = os.path.join(HERE, "results", "s63_engine_predicate_summary.json")
DATE = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
SMOKE = os.environ.get("SMOKE") == "1"
ATOL = RTOL = 1e-4

ARM_KEYS = ["torch", "torch_cmp"] + [f"{n}[{p}]" for n, _, _, _, _ in VARIANTS for p in PRECS]

CLASS_PATTERNS = [
    ("WINOGRAD", re.compile(r"winograd", re.I)),
    ("TF32_TENSOROP", re.compile(r"tf32|tensorop|1688|16816|h1688", re.I)),
    ("FP64", re.compile(r"dgrad|dgemm|dconv|fp64|double|_d_", re.I)),
    ("FP32_SIMT", re.compile(r"fprop|implicit_gemm|sgemm|s1688|fp32|scudnn|simt", re.I)),
    ("CUTLASS", re.compile(r"cutlass", re.I)),
]


def classify(names):
    """Return (label, matched_name) from the recorded kernel names, first match in order."""
    for lab, pat in CLASS_PATTERNS:
        for n in names:
            if pat.search(n):
                return lab, n
    return ("OTHER", names[0] if names else None)


def profile_conv(fn):
    """Run fn once under the CUDA profiler; return [(kernel_name, device_time_us)] top 5."""
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    evts = []
    for e in prof.key_averages():
        if e.self_device_time_total > 0:
            evts.append((e.key, float(e.self_device_time_total)))
    evts.sort(key=lambda t: -t[1])
    return evts[:5]


def margin(Y, R):
    d = (Y - R).abs()
    m_ref = (d - (ATOL + RTOL * R.abs())).max().item()
    m_cand = (d - (ATOL + RTOL * Y.abs())).max().item()
    return m_ref, m_cand


def gate_from_pass(passmap, H, C):
    passing = {k for k, v in passmap.items() if k.startswith("imp_") and v}
    gk = "torch"
    if H <= 14 and C <= 64:
        for p in PRECS:
            k = f"imp_B32x32[{p}]"
            if passmap.get(k):
                gk = k
                break
    return gk, sorted(passing)


def main():
    dev = torch.device("cuda:0")
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    done = set()
    if os.path.exists(OUTL):
        for line in open(OUTL):
            r = json.loads(line)
            done.add((r["tag"], r["N"], r["H"], r["C"]))
    outf = open(OUTL, "a")
    n_done = 0
    for tag, pts in GRIDS:
        for (H, N, C) in pts:
            if (tag, N, H, C) in done:
                continue
            if SMOKE and n_done >= 2:
                break
            torch.manual_seed(1234 + H * 100 + N * 10 + C)
            x = torch.randn(N, C, H, H, device=dev)
            w = torch.randn(C, C, 3, 3, device=dev)

            # --- references, each with an engine observation -----------------
            refs = {}
            _sv = torch.backends.cudnn.allow_tf32
            torch.backends.cudnn.allow_tf32 = True
            f_def = lambda: F.conv2d(x, w, padding=1)
            R_def = f_def(); torch.cuda.synchronize()
            refs["def"] = dict(R=R_def, eng=profile_conv(f_def))
            torch.backends.cudnn.allow_tf32 = False
            f_ieee = lambda: F.conv2d(x, w, padding=1)
            R_ieee = f_ieee(); torch.cuda.synchronize()
            refs["ieee"] = dict(R=R_ieee, eng=profile_conv(f_ieee))
            torch.backends.cudnn.allow_tf32 = _sv
            x64, w64 = x.double(), w.double()
            f_64 = lambda: F.conv2d(x64, w64, padding=1)
            R_64 = f_64().float(); torch.cuda.synchronize()
            refs["64"] = dict(R=R_64, eng=profile_conv(f_64))

            row = dict(tag=tag, N=N, H=H, W=H, C=C, CO=C, date=DATE, M=N * H * H, K=C * 9,
                       sm_count=sm, op_class="conv2d_3x3_s1p1_fp32", contract=CONTRACT,
                       predicate_ref_scaled="max(|Y-R|-(1e-4+1e-4|R|))<=0",
                       predicate_candidate_scaled="max(|Y-R|-(1e-4+1e-4|Y|))<=0",
                       refs={}, arms={})
            for rk, rd in refs.items():
                names = [n for n, _ in rd["eng"]]
                lab, match = classify(names)
                row["refs"][rk] = dict(absmax=rd["R"].abs().max().item(),
                                       engine_class=lab, engine_match=match,
                                       kernels=[dict(name=n, us=u) for n, u in rd["eng"]])
            # ref-to-ref deviation (anchor-normalized diagnostic, unchanged definition)
            row["ref_err"] = dict(
                def_vs_64=(refs["def"]["R"] - refs["64"]["R"]).abs().max().item(),
                ieee_vs_64=(refs["ieee"]["R"] - refs["64"]["R"]).abs().max().item(),
                def_vs_ieee=(refs["def"]["R"] - refs["ieee"]["R"]).abs().max().item())

            # --- arms: outputs once, margins under both predicates ------------
            outs = {}
            outs["torch"] = refs["def"]["R"].clone()
            try:
                fn = torch.compile(lambda xx: F.conv2d(xx, w, padding=1), dynamic=False)
                outs["torch_cmp"] = fn(x).clone()
            except Exception as ex:
                outs["torch_cmp"] = None
                row["torch_cmp_error"] = repr(ex)[:200]
            for name, BM, BN, BK, nw in VARIANTS:
                grid = (triton.cdiv(N * H * H, BM) * triton.cdiv(C, BN),)
                for prec in PRECS:
                    out = torch.empty(N, C, H, H, device=dev)
                    conv_impgemm_ip[grid](x, w, out, N, H, H, C, C, BM=BM, BN=BN, BK=BK,
                                          IP=prec, num_warps=nw)
                    outs[f"{name}[{prec}]"] = out.clone()
                    row["arms"].setdefault(f"{name}[{prec}]", {})["ctas"] = grid[0]
                    row["arms"][f"{name}[{prec}]"]["waves"] = grid[0] / sm
            torch.cuda.synchronize()

            for ak, Y in outs.items():
                if Y is None:
                    continue
                a = row["arms"].setdefault(ak, {})
                a["absmax"] = Y.abs().max().item()
                a["ref"] = {}
                for rk, rd in refs.items():
                    R = rd["R"]
                    m_ref, m_cand = margin(Y, R)
                    a["ref"][rk] = dict(maxabs=(Y - R).abs().max().item(),
                                        margin_ref=m_ref, margin_cand=m_cand,
                                        pass_ref=bool(m_ref <= 0), pass_cand=bool(m_cand <= 0))

            # --- derived readouts per predicate ------------------------------
            row["readouts"] = {}
            for pk, field in (("ref_scaled", "pass_ref"), ("cand_scaled", "pass_cand")):
                gates, adms = {}, {}
                for rk in ("def", "ieee", "64"):
                    pm = {ak: row["arms"][ak]["ref"][rk][field] for ak in row["arms"]
                          if ak in ARM_KEYS and "ref" in row["arms"][ak]}
                    g, adm = gate_from_pass(pm, H, C)
                    gates[rk], adms[rk] = g, adm
                row["readouts"][pk] = dict(gate=gates, admissible=adms)

            # --- label disagreements between predicates ---------------------
            dis = []
            for ak in row["arms"]:
                for rk in ("def", "ieee", "64"):
                    rr = row["arms"][ak].get("ref", {}).get(rk)
                    if rr and rr["pass_ref"] != rr["pass_cand"]:
                        dis.append(dict(arm=ak, ref=rk, pass_ref=rr["pass_ref"],
                                        pass_cand=rr["pass_cand"],
                                        margin_ref=rr["margin_ref"], margin_cand=rr["margin_cand"]))
            row["predicate_disagreements"] = dis
            outf.write(json.dumps(row) + "\n")
            outf.flush()
            n_done += 1
            print(f"[{n_done}] {tag} H{H} N{N} C{C} eng def/ieee/64 = "
                  f"{row['refs']['def']['engine_class']}/{row['refs']['ieee']['engine_class']}/"
                  f"{row['refs']['64']['engine_class']} pred-disagree={len(dis)}", flush=True)
    outf.close()

    rows = [json.loads(l) for l in open(OUTL)]
    # ---- summary (counts only; no thresholds) --------------------------------
    eng = {}
    for r in rows:
        for rk in ("def", "ieee", "64"):
            eng.setdefault(rk, {}).setdefault(r["refs"][rk]["engine_class"], 0)
            eng[rk][r["refs"][rk]["engine_class"]] += 1
    trans = {}
    for r in rows:
        k = (r["refs"]["def"]["engine_class"], r["refs"]["ieee"]["engine_class"])
        trans[str(k)] = trans.get(str(k), 0) + 1
    dis_by_ref = {}
    for r in rows:
        for d in r["predicate_disagreements"]:
            dis_by_ref[d["ref"]] = dis_by_ref.get(d["ref"], 0) + 1
    # delivered-arm change under each predicate, and whether engine class moved on those rows
    sw = {}
    for pk in ("ref_scaled", "cand_scaled"):
        for a, b in (("def", "ieee"), ("def", "64")):
            sel = [r for r in rows if r["readouts"][pk]["gate"][a] != r["readouts"][pk]["gate"][b]]
            engmoved = [r for r in sel if r["refs"][a]["engine_class"] != r["refs"][b]["engine_class"]]
            sw[f"{pk}:{a}->{b}"] = dict(n_sel=len(sel), n_eng_moved=len(engmoved),
                                        n_eng_same=len(sel) - len(engmoved))
    summary = dict(date=DATE, n_rows=len(rows), sm_count=sm,
                   engine_class_counts=eng, def_to_ieee_engine_transitions=trans,
                   predicate_disagreements_by_ref=dis_by_ref,
                   predicate_disagreements_total=sum(dis_by_ref.values()),
                   swap_counts=sw,
                   note="engine labels are observations from torch.profiler on the reference "
                        "conv call itself; counts are descriptive, no threshold applied")
    json.dump(summary, open(SUML, "w"), indent=1)
    print(json.dumps(summary, indent=1)[:2500])


if __name__ == "__main__":
    main()
