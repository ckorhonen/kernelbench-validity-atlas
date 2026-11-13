#!/usr/bin/env python
"""s93b_threshold_model.py — POST-HOC mechanism test, declared after s93's run (no new boundary claim).

Why post-hoc. s93 measured that the reference's relative-error distribution is unchanged across scale (p50
2.9243e-4, p99 1.862e-2 bit-identical at g=16/1024/4096) while the evaluator-predicate failure fraction falls from
1.86% to 0.027% at g=16. That pattern implies a specific mechanism rather than a fitted curve: the evaluator's
atol term raises the effective relative threshold to rtol + atol/|X_e|, and the failure fraction is the tail
probability above that per-element threshold. This script tests that model directly on the same tensors. It was
declared AFTER seeing s93's numbers, so it is labelled post-hoc and cannot create a new boundary; it can only
confirm or falsify the mechanism sentence attached to the measured plateau.

Model under test (no free parameters; rtol=atol=1e-2, X = fp64 anchor):
  pred_frac(g) = mean_e [ |d_e| > atol + rtol*|X_e| ]  computed from the measured |X_e| and rel_e = |d_e|/|X_e|
  equivalently pred_frac = P(rel_e > rtol + atol/|X_e|) plus the |X_e| = 0 case (fail iff |d_e| > atol).
The model is exact by construction if the evaluator predicate is what s91 assumed; the test is therefore an
instrument/definition audit rather than a discovery. What it adds is the *decomposition*: how much of the tail is
absorbed by the atol term at each scale, i.e. the quantitative link between the scale coordinate u and the
population statistic. A disagreement means the predicate direction or the anchor widening differs from the one
s93/s91 recorded, and the plateau sentence must be withdrawn.

Points: T1/T2 at g in {16, 1024, 1e5}; T3 at g in {1024, 1e5}. Same seeds, same arms, same normalization.
Verdict clauses:
  MODEL_EXACT          |pred - measured| <= 1e-9 absolute on every point (predicate reconstructed bit-exactly)
  MODEL_AGREES         <= 2e-4 absolute on every point (reconstruction correct to fp32 rounding of the comparison)
  MODEL_DISAGREES      otherwise => the plateau sentence is withdrawn and the row is reported as unexplained.
Recorded per point: pred_frac, measured frac_over, absorbed_mass = frac_atol0 - frac_over, the threshold at the
median |X|, and the |X| quantiles, so the absorption is readable as a function of scale rather than asserted.
"""
import hashlib
import json
import os
import signal
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s91_scale_boundary as s91
import s93_plateau_domain as s93

import torch

LAB = os.path.expanduser("~/kb-lab")
OUT = os.path.join(LAB, "results", "s93b_threshold_model.jsonl")
SUMOUT = os.path.join(LAB, "results", "s93b_threshold_model_summary.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

SHIPPED = s91.SHIPPED
CFG_ALARM_S = 240
POINTS = [("T1", 16.0), ("T1", 1024.0), ("T1", 100000.0),
          ("T2", 16.0), ("T2", 1024.0), ("T2", 100000.0),
          ("T3", 1024.0), ("T3", 100000.0)]
TD = {td["tid"]: td for td in list(s91.TUPLE_DEFS) + [s93.T3]}


def run_model_point(td, g):
    rec = dict(tid=td["tid"], g=g, status="ok")
    t0 = time.time()
    signal.alarm(CFG_ALARM_S)
    try:
        s91.set_cfg(True, False)
        torch.manual_seed(td["seed"])
        x, w = s91.build(td, torch.float32)
        fan_in = w.numel() // w.shape[0]
        w.div_(fan_in ** 0.5)
        w.mul_(g)
        a = s91.conv_fwd(x, w, td["args"])
        s91.set_cfg(False, False)
        xa = s91.conv_fwd(x.double(), w.double(), td["args"])
        del x, w
        af = a.detach().reshape(-1)
        xf = xa.detach().reshape(-1)
        n = int(af.numel())
        d = (af - xf).abs()
        ax = xf.abs()
        measured = d > (SHIPPED + SHIPPED * ax)
        n_meas = int(measured.sum().item())
        frac_meas = n_meas / n
        # model: rel threshold rtol + atol/|X|; exact zero-|X| branch
        nz = ax > 0
        rel = torch.zeros_like(d)
        rel[nz] = d[nz] / ax[nz]
        thr = torch.full_like(d, float("inf"))
        thr[nz] = SHIPPED + SHIPPED / ax[nz]
        pred = (rel > thr)
        pred[~nz] = d[~nz] > SHIPPED
        n_pred = int(pred.sum().item())
        rec.update(n_elems=n, frac_over=frac_meas, frac_model=n_pred / n, n_fail_meas=n_meas, n_fail_model=n_pred,
                   abs_delta=abs(n_pred - n_meas) / n,
                   absorbed_mass=float(int((d > SHIPPED * ax).sum().item())) / n - frac_meas,
                   n_zero_x=int((~nz).sum().item()),
                   x_quantiles={str(q): float(torch.quantile(ax, q).item()) for q in (0.01, 0.5, 0.99)},
                   thr_at_median_x=float(SHIPPED + SHIPPED / torch.quantile(ax, 0.5).item()))
        rec["maxabs_out"] = float(af.abs().max().item())
        rec["t_s"] = round(time.time() - t0, 2)
        del a, xa, d, ax, rel, thr, measured, pred
        torch.cuda.empty_cache()
    except Exception as e:
        rec["status"] = "error"
        rec["err"] = repr(e)[:300]
        torch.cuda.empty_cache()
    finally:
        signal.alarm(0)
        rec.setdefault("t_s", round(time.time() - t0, 2))
    return rec


def main():
    torch.cuda.init()
    open(OUT, "w").close()
    sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    print("s93b sha", sha[:12], flush=True)
    rows = []
    for tid, g in POINTS:
        r = run_model_point(TD[tid], g)
        rows.append(r)
        with open(OUT, "a") as f:
            f.write(json.dumps(r) + "\n")
        print("model", tid, "g", g, r["status"], "meas", r.get("frac_over"), "pred", r.get("frac_model"),
              "absdelta", r.get("abs_delta"), "absorbed", r.get("absorbed_mass"),
              "thr_med", r.get("thr_at_median_x"), "t", r.get("t_s"), flush=True)
    ok = [r for r in rows if r.get("status") == "ok"]
    deltas = [r["abs_delta"] for r in ok]
    summary = dict(sha=sha, n_attempted=len(rows), n_ok=len(ok), points=POINTS,
                   max_abs_delta=max(deltas) if deltas else None,
                   rows=[{k: r.get(k) for k in ("tid", "g", "frac_over", "frac_model", "abs_delta",
                                                "absorbed_mass", "thr_at_median_x", "x_quantiles")} for r in ok])
    if not ok:
        verdict = "INSTRUMENT_FAILED"
    elif max(deltas) <= 1e-9:
        verdict = "MODEL_EXACT"
    elif max(deltas) <= 2e-4:
        verdict = "MODEL_AGREES"
    else:
        verdict = "MODEL_DISAGREES"
    summary["verdict"] = verdict
    with open(SUMOUT, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1)[:4000], flush=True)


if __name__ == "__main__":
    main()
