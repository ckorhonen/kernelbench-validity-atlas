# conv_err_backfill.py - R2 residue burn, tick #36 (2026-10-21, thermodynamic_mind)
#
# Backfills correctness fields for conv rows, NOT timings:
#  (a) m3_curve conv rows: errs=null, variants={} (harness bug: conv-loop errs never
#      attached to row; reduction path only, m3_autotune_curve.py L186 vs L270ff).
#  (b) delivered-gate conv max-abs errs (w4 0.0566 / a2-day2 0.0844 class) lack the
#      reference magnitude needed to judge them against tolerance.
# Inputs are seeded deterministically per harness (m3/w4/a2: 1234+H*100+N*10+C;
# m2: row seed_base+H*100+N*10+C), so on the recorded stack the tensors and the
# fresh-read max-abs errs are reproducible; published latencies/regrets stand untouched.
# Per request row emits: ref_absmax/absmean/rms, per-variant fresh-read errs +
# evaluator-rule pass (|ref-cand| <= ATOL + RTOL*|cand|, ATOL=RTOL=1e-4, R1c direction),
# delivered-arm errs, and tie-out vs any recorded gate err.
# Failure convention (R2): any variant exception -> status string in errs + failures[];
# summary counts; runner exits 3 if any delivered arm lacks a numeric err.
import json, sys
import torch, torch.nn.functional as F
import triton
from w3_conv_gemm import conv_impgemm, VARIANTS

ATOL = RTOL = 1e-4
REQ_IN, OUT = "conv_rows.json", "conv_err_backfill.jsonl"

def run_variant(fn, out, ref):
    out.zero_()
    res = fn()
    o = res if isinstance(res, torch.Tensor) else out
    torch.cuda.synchronize()
    err = float((ref - o).abs().max())
    ok = bool(((ref - o).abs() <= ATOL + RTOL * o.abs()).all().item())
    return err, ok

def main():
    dev = "cuda"
    reqs = json.load(open(REQ_IN))
    cache = {}
    out_rows, failures = [], []
    for req in reqs:
        N, H, C = req["N"], req["H"], req["C"]
        sb = req.get("seed_base", 1234)
        key = (N, H, C, sb)
        if key not in cache:
            torch.manual_seed(sb + H * 100 + N * 10 + C)
            x = torch.randn(N, C, H, H, device=dev)
            w = torch.randn(C, C, 3, 3, device=dev)
            ref = F.conv2d(x, w, padding=1)
            M = N * H * H
            rec = dict(ref_absmax=float(ref.abs().max()),
                       ref_absmean=float(ref.abs().mean()),
                       ref_rms=float(ref.pow(2).mean().sqrt()), errs={})
            pool = {"torch": (lambda: F.conv2d(x, w, padding=1),
                              torch.empty_like(ref))}
            for name, BMc, BNc, BKc, nw in VARIANTS:
                out = torch.empty_like(ref)
                grid = (triton.cdiv(M, BMc) * triton.cdiv(C, BNc),)
                def go(BMc=BMc, BNc=BNc, BKc=BKc, nw=nw, out=out, grid=grid):
                    conv_impgemm[grid](x, w, out, N, H, H, C, C,
                                       BM=BMc, BN=BNc, BK=BKc, num_warps=nw)
                pool[name] = (go, out)
            for nm, (fn, out) in pool.items():
                try:
                    err, ok = run_variant(fn, out, ref)
                    rec["errs"][nm] = dict(max_abs_err=err, pass_1e4=ok)
                except Exception as ex:
                    rec["errs"][nm] = dict(status="error:%r" % (ex,))
                    failures.append("%s N%dH%dC%d" % (nm, N, H, C))
            cache[key] = rec
        rec = cache[key]
        gate = dict(choice=req["gate_choice"])
        e = rec["errs"].get(req["gate_choice"], {})
        if "max_abs_err" in e:
            gate.update(err=e["max_abs_err"], pass_1e4=e["pass_1e4"])
        else:
            gate["status"] = e.get("status", "missing")
        rec_err = req.get("gate_err_recorded")
        if rec_err is not None and "err" in gate:
            gate["recorded_err"] = rec_err
            gate["tieout_absdelta"] = abs(gate["err"] - rec_err)
        delivered = {}
        for ch in req["delivered_choices"]:
            e = rec["errs"].get(ch, {})
            delivered[ch] = (dict(max_abs_err=e["max_abs_err"], pass_1e4=e["pass_1e4"])
                             if "max_abs_err" in e else dict(status=e.get("status", "missing")))
        out_rows.append(dict(src=req["src"], N=N, H=H, C=C, seed_base=sb,
                             ref_absmax=rec["ref_absmax"], ref_absmean=rec["ref_absmean"],
                             ref_rms=rec["ref_rms"], errs=rec["errs"],
                             gate=gate, delivered=delivered))
    n_del = sum(len(r["delivered"]) + 1 for r in out_rows)
    n_pass = sum(1 for r in out_rows
                 if r["gate"].get("pass_1e4") and all(v.get("pass_1e4") for v in r["delivered"].values()))
    ties = [r["gate"]["tieout_absdelta"] for r in out_rows if "tieout_absdelta" in r["gate"]]
    summary = dict(n_rows=len(out_rows), n_delivered_checks=n_del, n_pass_all=n_pass,
                   n_fail=n_del - n_pass, tieout_max=max(ties) if ties else None,
                   failures=failures, atol=ATOL, rtol=RTOL)
    with open(OUT, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    json.dump(summary, open("conv_err_backfill_summary.json", "w"), indent=1)
    print(json.dumps(summary))
    if summary["n_fail"]:
        sys.exit(3)

if __name__ == "__main__":
    main()
