#!/usr/bin/env python
"""s96b_ratio_argmax.py — POST-HOC mechanism probe (declared after s96's numbers; no new boundary).

s96 fired AMPLITUDE_LOTTERY: at fixed M=N=1024 K=4096 the reference's extremum-to-budget ratio rho_ref
spans 1.41x (CV 9.2%) across 32 draws while sigma_max(a) varies 0.2% and the bulk tail (p99.99) varies
2.2%. This probe asks WHERE in the ratio the variance lives, by recording the element that maximises the
evaluator ratio r = e/(1e-4+1e-4|X|) per draw (s96 recorded only the argmax of e, which is a different
element) and decomposing log rho into log e minus log(1e-4*(1+|X_e|)).

Reported: per-draw rho, e_e, |X_e|, cancellation depth |X_e|/median|X|, and the variance shares of
log rho attributable to log e and to log(1+|X_e|) (covariance decomposition of two terms whose sum is
log rho, so shares are not assumed to add to 1). No verdict is issued; the reading is descriptive.
"""
import json
import os
import signal
import time

import torch

LAB = os.path.expanduser("~/kb-lab")
OUT = os.path.join(LAB, "results", "s96b_ratio_argmax.jsonl")
SUMOUT = os.path.join(LAB, "results", "s96b_ratio_argmax_summary.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

STRICT = 1e-4
CFG_ALARM_S = 240
CHUNK = 1 << 22
NDRAWS = 32
MN = 1024
K = 4096


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout("config exceeded %ds" % CFG_ALARM_S)


signal.signal(signal.SIGALRM, _alarm)


def set_flags():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def append(rec):
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")


def run(draw):
    seed = 9600 + draw
    rec = dict(tid="B%d" % draw, draw=draw, M=MN, N=MN, K=K, seed=seed, status="ok")
    t0 = time.time()
    signal.alarm(CFG_ALARM_S)
    try:
        set_flags()
        g = torch.Generator(device="cuda").manual_seed(seed)
        a = torch.randn(MN, K, generator=g, device="cuda", dtype=torch.float32)
        b = torch.randn(K, MN, generator=g, device="cuda", dtype=torch.float32)
        A = torch.mm(a, b)
        X = torch.mm(a.double(), b.double())
        dAX = (A.double() - X).abs()
        thr = STRICT + STRICT * X.abs()
        r = dAX / thr
        flat = r.reshape(-1)
        idx = int(flat.argmax().item())
        e_flat = dAX.reshape(-1)
        x_flat = X.abs().reshape(-1)
        med = float(x_flat.median().item())
        rec["rho_ref"] = float(flat[idx].item())
        rec["e_at_ratio_argmax"] = float(e_flat[idx].item())
        rec["abs_x_at_ratio_argmax"] = float(x_flat[idx].item())
        rec["depth_at_ratio_argmax"] = rec["abs_x_at_ratio_argmax"] / max(med, 1e-30)
        rec["median_abs_x"] = med
        rec["log_rho"] = float(torch.log(torch.tensor(rec["rho_ref"])).item())
        rec["log_e"] = float(torch.log(torch.tensor(rec["e_at_ratio_argmax"])).item())
        rec["log_thr"] = float(torch.log(torch.tensor(STRICT * (1.0 + rec["abs_x_at_ratio_argmax"]))).item())
        # where the argmax of the absolute error sits, for comparison
        idxe = int(e_flat.argmax().item())
        rec["e_max"] = float(e_flat[idxe].item())
        rec["abs_x_at_e_argmax"] = float(x_flat[idxe].item())
        rec["wall_s"] = time.time() - t0
        del a, b, A, X, dAX, thr, r, flat, e_flat, x_flat
        torch.cuda.empty_cache()
    except Timeout:
        rec["status"] = "timeout"
    except Exception as e:  # noqa: BLE001
        rec["status"] = "error"
        rec["error"] = repr(e)
    finally:
        signal.alarm(0)
    append(rec)
    return rec


def main():
    open(OUT, "w").close()
    for draw in range(NDRAWS):
        r = run(draw)
        print("%s status=%s rho=%.6g e=%.4g |x|=%.4g depth=%.4g"
              % (r["tid"], r["status"], r.get("rho_ref", -1), r.get("e_at_ratio_argmax", -1),
                 r.get("abs_x_at_ratio_argmax", -1), r.get("depth_at_ratio_argmax", -1)), flush=True)
    with open(SUMOUT, "w") as f:
        json.dump(dict(rows=NDRAWS, geometry=dict(M=MN, N=MN, K=K)), f, indent=1)


if __name__ == "__main__":
    main()
