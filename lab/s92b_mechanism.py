#!/usr/bin/env python
"""s92b_mechanism.py — POST-HOC mechanism probe for the s92 plateau (declared after s92 ran; no frozen branch).

Question the plateau raises: frac_over stops rising at ~1.8% while rho keeps growing. If the predicate at large
output scale is effectively relative (atol negligible against rtol*|X|), then the plateau is not a gate artifact but
the tail of the reference's RELATIVE-error distribution: frac_over -> P(|A-X|/|X| > rtol). This probe measures that
directly on the two s92 tuples at the top of the grid (g=4096, the point where s92 recorded frac 0.01842/0.01831).

Design (same harness, same seeds, same arms; no thresholds beyond the frozen 1e-2):
  - A default fp32 reference, X fp64 anchor on the exact widening, chunked.
  - frac_over with the evaluator predicate (atol=rtol=1e-2) and with atol=0 (pure relative), so the atol share is
    measurable rather than assumed.
  - relative-error quantiles p50/p90/p99/p99.9/max, and the cancellation depth of the argmax element
    (|X| at argmax / median |X|) as the mechanism diagnostic.
Declared limit: one scale point per tuple; the claim is scoped to "at this scale the predicate is relative", not a
general statement about all scales.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s91_scale_boundary as s91

import torch

LAB = os.path.expanduser("~/kb-lab")
OUT = os.path.join(LAB, "results", "s92b_mechanism.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
G = 4096.0
TOL = 1e-2
CHUNK = 1 << 22


def measure(td):
    torch.manual_seed(td["seed"])
    x, w = s91.build(td, torch.float32)
    fan_in = w.numel() // w.shape[0]
    w.div_(fan_in ** 0.5).mul_(G)
    s91.set_cfg(True, False)
    a = s91.conv_fwd(x, w, td["args"])
    s91.set_cfg(False, False)
    xa = s91.conv_fwd(x.double(), w.double(), td["args"])
    af, xf = a.reshape(-1), xa.reshape(-1)
    n = af.numel()
    n_ev = n_rel = 0
    rels = []
    i = 0
    maxrel, maxrel_x, maxrel_d = 0.0, 0.0, 0.0
    while i < n:
        j = min(n, i + CHUNK)
        d = (af[i:j] - xf[i:j]).abs()
        xv = xf[i:j].abs()
        r_ev = d / (TOL + TOL * xv)
        r_rel = d / xv.clamp_min(1e-30)
        n_ev += int((r_ev > 1).sum().item())
        n_rel += int((r_rel > TOL).sum().item())
        rels.append(r_rel.cpu())
        mr, am = torch.max(r_rel, 0)
        if float(mr.item()) > maxrel:
            maxrel = float(mr.item())
            k = int(am.item())
            maxrel_x, maxrel_d = float(xv[k].item()), float(d[k].item())
        del d, xv, r_ev, r_rel
        i = j
    allr = torch.cat(rels)
    q = {p: float(torch.quantile(allr, p).item()) for p in (0.5, 0.9, 0.99, 0.999)}
    med_x = float(xf.abs().median().item())
    rec = dict(tid=td["tid"], g=G, n=n, frac_evaluator=n_ev / n, frac_atol0=n_rel / n,
               rel_q=q, max_rel=maxrel, argmax_abs_x=maxrel_x, argmax_delta=maxrel_d,
               median_abs_x=med_x, argmax_over_median=(maxrel_x / med_x) if med_x else None,
               maxabs_out=float(a.abs().max().item()))
    del a, xa, af, xf, allr, rels
    torch.cuda.empty_cache()
    return rec


def main():
    torch.cuda.init()
    out = dict(prereg="post-hoc, declared after s92", g=G, rows=[measure(td) for td in s91.TUPLE_DEFS])
    for r in out["rows"]:
        r["plateau_matches_relative_tail"] = abs(r["frac_evaluator"] - r["frac_atol0"]) <= 0.002
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
