#!/usr/bin/env python3
"""e59b — per-element budget-ratio recorder (instrument repair for e59).

WHY: e59's prereg rebuilt admissibility from recorded maxima (max_abs_err <= eps*(1+ref_absmax)) and
that reconstruction FAILED its own validation leg (agreement 0.8735 < 0.95 bar, 85/672 mismatches, all
in the over-admitting direction: atol-dominated elements make the max-based threshold far too generous).
Per the prereg, no eps-ordering claim is made from it.

This leg records the quantity the prereg actually named: the per-element budget ratio at the benchmark's
1e-4 denominator, max over elements, for every arm of every one of the 56 conv shapes in
lab/results/r1_conv_gated.jsonl. Admissibility at any eps is then a recomputation (ratio <= eps*1e4), and
the eps-ordering / frontier / kill-clause objects of lab/e59_epsilon_sweep_prereg.md apply unchanged.

Frozen here BEFORE the run:
- Population, seeds, grids, arms, contract: verbatim r1_conv_gated.py (imported, not re-typed).
- Denominators: BOTH recorded per arm -- ref-scaled (|ref|) and candidate-scaled (|cand|). The evaluator
  direction is candidate-scaled (torch.allclose(ref, cand) scales rtol by the second argument), so the
  candidate-scaled ratio is the primary; the ref-scaled one is descriptive.
- Instrument validation: primary ratio <= 1.0 must reproduce the recorded pass_1e4 label on all 672
  arm-checks; bar is exact agreement (>=99.5% accepted as instrument-valid, mismatches enumerated).
- Timings are NOT re-measured: frontier times are inherited from the committed r1_conv_gated.jsonl,
  matched by (tag,N,H,C,arm). Declared; this leg measures error only.
- torch.compile arm runs as in r1 (dynamic=False, warm inductor cache), its ratio recorded the same way.
- No service touched; GPU forward passes only, single pass per arm, no reps.
"""
import json, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r1_conv_gated as R

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "e59b_budget_ratios.jsonl")
DENOM = 1e-4

def ratios(diff, ref, cand):
    # per-element budget ratio at the 1e-4 contract, max over elements
    r_ref = (diff.abs() / (DENOM + DENOM * ref.abs())).max().item()
    r_cand = (diff.abs() / (DENOM + DENOM * cand.abs())).max().item()
    return r_ref, r_cand

def main():
    dev = torch.device("cuda:0")
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    recorded = {}
    for line in open(os.path.join(os.path.dirname(OUT), "r1_conv_gated.jsonl")):
        if line.strip():
            r = json.loads(line)
            recorded[(r["tag"], r["N"], r["H"], r["C"])] = r
    outf = open(OUT, "w")
    t_start = time.time()
    n_rows = 0
    for tag, pts in R.GRIDS:
        for (H, N, C) in pts:
            key = (tag, N, H, C)
            if key not in recorded:
                continue
            torch.manual_seed(1234 + H * 100 + N * 10 + C)
            x = torch.randn(N, C, H, H, device=dev)
            w = torch.randn(C, C, 3, 3, device=dev)
            ref = F.conv2d(x, w, padding=1)
            M = N * H * H
            row = dict(tag=tag, N=N, H=H, C=C, M=M, date=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       ref_absmax=ref.abs().max().item(), sm_count=sm, arms={})
            A = row["arms"]
            # torch eager == the reference op; ratio is 0 by construction
            A["torch"] = dict(ratio_ref=0.0, ratio_cand=0.0, pass_1e4=True)
            # compiled arm
            try:
                fn = torch.compile(lambda xx: F.conv2d(xx, w, padding=1), dynamic=False)
                fn(x); torch.cuda.synchronize()
                out_c = fn(x)
                rr, rc = ratios(out_c - ref, ref, out_c)
                A["torch_cmp"] = dict(ratio_ref=rr, ratio_cand=rc,
                                      pass_1e4=bool(torch.allclose(ref, out_c, **R.CONTRACT)))
            except Exception as ex:
                A["torch_cmp"] = dict(error=repr(ex)[:200])
            for name, BM, BN, BK, nw in R.VARIANTS:
                grid = (torch.cdiv(torch.tensor(M), BM).item() * torch.cdiv(torch.tensor(C), BN).item(),)
                for prec in R.PRECS:
                    out = torch.empty_like(ref)
                    R.conv_impgemm_ip[grid](x, w, out, N, H, H, C, C, BM=BM, BN=BN, BK=BK,
                                            IP=prec, num_warps=nw)
                    torch.cuda.synchronize()
                    rr, rc = ratios(out - ref, ref, out)
                    A[f"{name}[{prec}]"] = dict(ratio_ref=rr, ratio_cand=rc,
                                                pass_1e4=bool(torch.allclose(ref, out, **R.CONTRACT)))
            # validation leg against the committed labels
            mism = []
            for arm, rec in recorded[key]["variants"].items():
                if arm not in A or "ratio_cand" not in A[arm]: continue
                got = A[arm]["ratio_cand"] <= 1.0
                want = bool(rec.get("pass_1e4"))
                if got != want:
                    mism.append({"arm": arm, "recon": got, "recorded": want,
                                 "ratio_cand": A[arm]["ratio_cand"]})
            row["validation"] = dict(n=sum(1 for a in A if "ratio_cand" in A[a]), mismatch=len(mism),
                                     mismatches=mism)
            outf.write(json.dumps(row) + "\n"); outf.flush()
            n_rows += 1
            print(f"{tag} H{H} N{N} C{C}: arms={len(A)} mism={len(mism)} t={time.time()-t_start:.0f}s", flush=True)
    outf.close()
    print(f"DONE rows={n_rows} wall={time.time()-t_start:.0f}s -> {OUT}")

if __name__ == "__main__":
    main()
