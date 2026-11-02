"""s82_strided_median.py — is the 0.0 median reference budget ratio on the five FALSY_ZERO rows a
prefix-sampling artifact? GPU, spark. Same arms, seeds and predicates as s73/s73b.

PREREG (frozen at commit BEFORE the first run; no threshold, population or read changed after results)

Motivation: s81 established that five exceed rows carry ratio_ref_median == 0.0 exactly, excluded from
s74/s75b by a falsy test. s73 takes its ratio sample as the FIRST RATIO_CAP (20,000) elements in chunk order
(contiguous prefix). A 0.0 median therefore says the prefix lands where the reference equals the fp64 anchor
bit-exactly, not that the tensor's typical element is exact. This producer measures the same quantity under
a STRIDED sample spanning the whole tensor, on the same inputs, to decide which reading is correct.

Population: exactly the five rows s81 classified FALSY_ZERO — (1,81), (2,1), (2,2), (2,79), (2,93), all
resolved by (level, problem_id, name) against s71_full_audit.jsonl.

Arms per row, one model instance, seed 0 (identical to s73/s73b): A default fp32 reference (cudnn tf32 on,
matmul tf32 off), X fp64 anchor from the same fp32 weights, C fp32 tf32 off. Predicate evaluator-exact,
thr = 1e-4 + 1e-4*|C|.

Quantities at 1e-4: prefix median (first 20,000 elements, must reproduce s73's recorded value), strided
median/q90/q99 over a uniform stride sample of ~200,000 elements spanning the tensor, frac over budget,
fraction of elements with ratio <= 1e-3 (near-exact mass).

TIE-OUT GATE (per row): prefix median must equal s73c's recorded ratio_ref_median exactly AND the chunked
ref_over_budget count must equal s73c's recorded count exactly; otherwise the row is excluded and no new
number from it is used.

FROZEN VERDICT (per row, then pooled):
  PREFIX_ARTIFACT   prefix median == 0.0 and strided median > 0
  PREFIX_REPRESENTATIVE otherwise
Pooled: ZERO_MEDIAN_IS_PREFIX_ARTIFACT iff PREFIX_ARTIFACT on >= 4/5 rows; else PREFIX_MEDIAN_REPRESENTATIVE.
Kill clause: if the tie-out gate fails on >= 2 rows, verdict is UNRESOLVED (report nulls, no claim).

Limits declared: one input draw per row (s79d/s80b measured draw invariance <=1.17x at fixed tuples, so a
draw is not the object here); strided sampling estimates quantiles to ~1/sqrt(n) relative precision on the
near-zero mass, which is enough to separate 0.0 from any positive median.
"""
import hashlib
import json
import os
import signal
import time

import torch

import s73b_oom_rerun as base  # same directory; reuses load_task/set_cfg/run_model/_rows_cols

LAB = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(LAB, "results")
KB = os.path.expanduser("~/kernelbench")
S71 = os.path.join(RES, "s71_full_audit.jsonl")
S73C = os.path.join(RES, "s73c_merged_56.jsonl")
OUT = os.path.join(RES, "s82_strided_median.json")

POP = [(1, 81), (2, 1), (2, 2), (2, 79), (2, 93)]
PREFIX_CAP = 20000
STRIDE_CAP = 200000
CFG_ALARM_S = 900


def _alarm(signum, frame):
    raise TimeoutError("task exceeded wall alarm")


def sample_strided(a, x, c, tol):
    """Uniform-stride sample of r_ref = |A-X|/(tol + tol*|C|) across the whole tensor, plus exact counts."""
    rows, cols = base._rows_cols(a)
    af = a.reshape(rows, cols)
    xf = x.reshape(rows, cols)
    cf = c.reshape(rows, cols)
    total = int(rows * cols)
    stride = max(1, total // STRIDE_CAP)
    step = max(1, base.ELEM_CHUNK // max(1, cols))
    prefix, strided = [], []
    over = 0
    near_exact = 0
    seen = 0
    for i in range(0, rows, step):
        ac = af[i:i + step].double()
        xc = xf[i:i + step]
        cc = cf[i:i + step].double()
        d_ax = (ac - xc).abs()
        thr = tol + tol * cc.abs()
        r = (d_ax / thr).reshape(-1)
        over += int((d_ax > thr).sum())
        if len(prefix) < PREFIX_CAP:
            prefix.extend(r[:PREFIX_CAP - len(prefix)].tolist())
        # global-index stride: local offsets continue the global phase across chunks
        idx = range((-seen) % stride, r.numel(), stride)
        if len(strided) < STRIDE_CAP:
            take = list(idx)[:STRIDE_CAP - len(strided)]
            if take:
                strided.extend(r[torch.tensor(take, device=r.device)].tolist())
        near_exact += int((r <= 1e-3).sum())
        seen += r.numel()
        del ac, cc, d_ax, thr, r
    return {"prefix": prefix, "strided": strided, "over": over, "total": total, "near_exact": near_exact}


def q(smp, p):
    s = sorted(smp)
    return s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))] if s else None


def main():
    torch.cuda.init()
    dev = "cuda"
    signal.signal(signal.SIGALRM, _alarm)
    prereg_sha = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()

    rows71 = {(r["level"], r["problem_id"]): r for r in
              (json.loads(l) for l in open(S71))}
    rows73 = {(r["level"], r["problem_id"]): r for r in
              (json.loads(l) for l in open(S73C))}

    out = {"prereg_sha": prereg_sha, "population": [], "rows": [], "tie_out_fail": 0}
    for level, pid in POP:
        r71 = rows71.get((level, pid))
        r73 = rows73.get((level, pid))
        rec = {"level": level, "problem_id": pid, "name": r71["name"], "op_class": r71.get("op_class")}
        recorded = (r73.get("decomp_c2_tf32off") or {}).get("1e-4", {})
        A = X = C = None
        try:
            torch.manual_seed(0)
            torch.cuda.empty_cache()
            mod = base.load_task(os.path.join(KB, "KernelBench", "level%d" % level, r71["name"]))
            init_args = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
            inputs = mod.get_inputs()
            torch.manual_seed(0)
            model = mod.Model(*init_args)

            signal.alarm(CFG_ALARM_S)
            base.set_cfg(True, False)
            A = base.run_model(model, inputs, torch.float32, dev)
            base.set_cfg(False, False)
            X = base.run_model(model, inputs, torch.float64, dev)
            base.set_cfg(False, True)
            C = base.run_model(model, inputs, torch.float32, dev)
            signal.alarm(0)
            if A is None or X is None or C is None:
                raise RuntimeError("missing arm output")

            smp = sample_strided(A, X, C, 1e-4)
            rec["prefix_median"] = q(smp["prefix"], 0.5)
            rec["strided_median"] = q(smp["strided"], 0.5)
            rec["strided_q90"] = q(smp["strided"], 0.9)
            rec["strided_q99"] = q(smp["strided"], 0.99)
            rec["n_prefix"] = len(smp["prefix"])
            rec["n_strided"] = len(smp["strided"])
            rec["frac_over"] = smp["over"] / smp["total"]
            rec["near_exact_frac"] = smp["near_exact"] / smp["total"]
            rec["ref_over_budget"] = smp["over"]
            rec["recorded_prefix_median"] = recorded.get("ratio_ref_median")
            rec["recorded_over"] = recorded.get("ref_over_budget")
            rec["tie_out_ok"] = (rec["prefix_median"] == recorded.get("ratio_ref_median")
                                 and rec["ref_over_budget"] == recorded.get("ref_over_budget"))
            if rec["prefix_median"] == 0.0 and (rec["strided_median"] or 0) > 0:
                rec["verdict"] = "PREFIX_ARTIFACT"
            else:
                rec["verdict"] = "PREFIX_REPRESENTATIVE"
            if not rec["tie_out_ok"]:
                out["tie_out_fail"] += 1
                rec["verdict"] = "TIE_OUT_MISMATCH"
        except Exception as e:  # noqa: BLE001
            rec["error"] = repr(e)
            rec["verdict"] = "ERROR"
        out["rows"].append(rec)
        print(json.dumps(rec, default=str), flush=True)
        del A, X, C
        torch.cuda.empty_cache()

    verdicts = [r.get("verdict") for r in out["rows"]]
    n_art = sum(1 for v in verdicts if v == "PREFIX_ARTIFACT")
    if out["tie_out_fail"] >= 2:
        out["verdict"] = "UNRESOLVED"
    elif n_art >= 4:
        out["verdict"] = "ZERO_MEDIAN_IS_PREFIX_ARTIFACT"
    else:
        out["verdict"] = "PREFIX_MEDIAN_REPRESENTATIVE"
    out["n_prefix_artifact"] = n_art
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("VERDICT", out["verdict"], "n_art", n_art, "tie_out_fail", out["tie_out_fail"], flush=True)


if __name__ == "__main__":
    main()
