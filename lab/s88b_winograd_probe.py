#!/usr/bin/env python
"""s88b_winograd_probe.py — POST-HOC descriptive probe (NOT part of the s88 prereg).

s88 found all 19 r1 winograd tuples reference-clean (frac_over exactly 0.0, rho <= 0.300), which contradicts
A7's winograd row (H=W=5, C=CO=256, N=2: rho 10.195, frac_over 0.0052). This probe asks whether that
exceedance replicates at A7's exact tuple and whether it is a narrow shape cell.

No kill clauses, no threshold; three declared tuples: A7's exact (2,5,5,256,256) and two brackets
(2,6,6,256,256), (2,8,8,256,256). Same arms and predicate as s88. Any reading is descriptive, labeled post-hoc.
"""
import json, os, sys, time
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from s88_class_share import (classify, profile_call, set_flags, wilson, ATOL, RTOL)

OUT = os.path.join(HERE, "results", "s88b_winograd_probe.json")
TUPLES = [(2, 5, 5, 256, 256), (2, 6, 6, 256, 256), (2, 8, 8, 256, 256)]
DATE = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run(N, H, W, C, CO, i):
    torch.manual_seed(8900 + i)
    conv = torch.nn.Conv2d(C, CO, 3, 1, 1, bias=True).cuda()
    x = torch.randn(N, C, H, W, device="cuda")
    set_flags(True)
    with torch.no_grad():
        A = F.conv2d(x, conv.weight, conv.bias, 1, 1)
    engA = classify([n for n, _ in profile_call(lambda: F.conv2d(x, conv.weight, conv.bias, 1, 1))])
    with torch.no_grad():
        X = F.conv2d(x.double(), conv.weight.double(), conv.bias.double(), 1, 1)
    Ad, dref = A.double(), (A.double() - X).abs()
    frac_over = float((dref > ATOL + RTOL * Ad.abs()).sum().item()) / dref.numel()
    rho = float((dref / (ATOL + RTOL * X.abs())).max().item())
    set_flags(False)
    with torch.no_grad():
        C2 = F.conv2d(x, conv.weight, conv.bias, 1, 1)
    engC2 = classify([n for n, _ in profile_call(lambda: F.conv2d(x, conv.weight, conv.bias, 1, 1))])
    set_flags(True)
    cm = torch.compile(conv, dynamic=False)
    with torch.no_grad():
        C1 = cm(x)
    engC1 = classify([n for n, _ in profile_call(lambda: cm(x))])
    out = {"date": DATE, "N": N, "H": H, "W": W, "C": C, "CO": CO, "n_elem": int(dref.numel()),
           "engine_A": engA[0], "engine_A_name": engA[1], "engine_C1": engC1[0], "engine_C2": engC2[0],
           "frac_over_ref": frac_over, "rho_max": rho, "cands": {}}
    for name, cand in (("C1", C1), ("C2", C2)):
        Cd = cand.double()
        be = ATOL + RTOL * Cd.abs()
        fail = (Cd - Ad).abs() > be
        nf = int(fail.sum().item())
        ri = int((fail & (dref > be)).sum().item())
        out["cands"][name] = {"n_fail": nf, "n_ref_ind": ri, "n_cand_ind": nf - ri,
                              "share": (ri / nf) if nf else None, "bit_identical": bool(torch.equal(cand, A))}
    return out


if __name__ == "__main__":
    res = [run(*t, i) for i, t in enumerate(TUPLES)]
    json.dump({"post_hoc": True, "note": "descriptive probe; no prereg, no kill", "rows": res}, open(OUT, "w"), indent=1)
    for r in res:
        print(r["N"], r["H"], r["W"], r["C"], r["CO"], r["engine_A"], "frac_over %.5f" % r["frac_over_ref"],
              "rho %.3f" % r["rho_max"], {k: (v["n_fail"], v["share"]) for k, v in r["cands"].items()})
