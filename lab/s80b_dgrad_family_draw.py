#!/usr/bin/env python
"""s80b_dgrad_family_draw.py — draw invariance on the 14 dgrad-family tuples that the DECLARED R1 rule selects.

PREREG (frozen before the run; do not edit after).

s80 re-profiled the 54 exceed rows under two declared ranking rules and found that s75's recorded kernel_top1
reproduces the ALL-DEVICE-EVENT rule (51/54), not the compute-kernel rule its docstring declares: 11/54 labels
are Memcpy/Memset. Under the declared compute-kernel rule (R1) the dgrad family is 14 tuples, not the five that
s79c derived from the recorded labels, with an across-shape bias span of 8.963x. s80's frozen P4 clause flags
DRAW_RERUN_NEEDED for exactly this case.

Population (rule, frozen): the 14 (level, problem_id) pairs in results/s80_instance_rule_summary.json
["P4"]["tuples"]. No exclusions by outcome.

Per tuple: model weights fixed once (init seed 0); N_DRAWS=6 input draws, seeds 300..305. One profiler pass on
the reference call before the draws records the live top-1 kernel and whether it contains "dgrad"; the live
top-1 is compared with s80's R1 label for the same tuple (within-tick reproducibility of the ranking rule).
Arms per draw exactly s73/s79d's: A = default fp32 reference (cudnn tf32 on, matmul tf32 off); X = fp64 anchor
from the same fp32 weights. Predicate thr = atol + rtol*|X| at 1e-4 and 1e-2, element-chunked.

FROZEN VERDICT RULE (declared before the run):
  DRAW_INVARIANT_ALL  if every tuple with >=4 ok draws has max(rho)/min(rho) <= 1.5;
  DRAW_DEPENDENT      if any such tuple exceeds 2.0;
  DRAW_PARTIAL        otherwise.
FROZEN SECONDARY READS:
  S1 shape_to_draw_ratio = 8.963044 (s80 P4 across-shape span) / max observed draw spread. SHAPE_DOMINATES_
     AT_FAMILY_SCOPE if >= 4.0, else FAMILY_SCOPE_AMBIGUOUS.
  S2 live label agreement: fraction of tuples whose live profiler top-1 (self device time, compute kernels only)
     contains "dgrad". If < 1.0 the R1 label is not reproducible on a second pass and is reported as such.

Limits declared: six seeds per tuple, one draw generator per task, one profiler pass per tuple, GB10 sm_120,
torch 2.14.1+cu130, one run.
"""
import glob
import hashlib
import importlib.util
import json
import os
import re
import signal
import time

import torch

KB = os.path.expanduser("~/kernelbench/KernelBench")
LAB = os.path.expanduser("~/kb-lab")
OUT = os.path.join(LAB, "results", "s80b_dgrad_family_draw.jsonl")
SUMOUT = os.path.join(LAB, "results", "s80b_dgrad_family_draw_summary.json")
S80 = os.path.join(LAB, "results", "s80_instance_rule_summary.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

NEEDLE = "dgrad"
MEM_RE = re.compile(r"memcpy|memset", re.I)
ELEM_CHUNK = 1 << 22
ATOL_RTOL = 1e-4
ATOL_RTOL_BENCH = 1e-2
N_DRAWS = 6
SEED_BASE = 300
CFG_ALARM_S = 300
ACROSS_SPAN = 8.963044122021856


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout("config exceeded %ds" % CFG_ALARM_S)


signal.signal(signal.SIGALRM, _alarm)


def load_task(path):
    name = "kb_" + os.path.basename(path).replace(".py", "").replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def set_cfg(tf32_cudnn, tf32_matmul):
    torch.backends.cudnn.allow_tf32 = tf32_cudnn
    torch.backends.cuda.matmul.allow_tf32 = tf32_matmul


def cast_inputs(x, dtype, device):
    if isinstance(x, torch.Tensor):
        if x.is_floating_point():
            return x.to(device=device, dtype=dtype)
        return x.to(device=device)
    if isinstance(x, (list, tuple)):
        return type(x)(cast_inputs(v, dtype, device) for v in x)
    if isinstance(x, dict):
        return {k: cast_inputs(v, dtype, device) for k, v in x.items()}
    return x


def first_tensor(o):
    if isinstance(o, torch.Tensor):
        return o
    if isinstance(o, (list, tuple)):
        for v in o:
            t = first_tensor(v)
            if t is not None:
                return t
    return None


def run_model(model, inputs, dtype, device):
    model = model.to(device=device, dtype=dtype)
    model.eval()
    with torch.no_grad():
        out = model(*[cast_inputs(x, dtype, device) for x in inputs])
    return first_tensor(out)


def _rows_cols(t):
    if t.ndim == 0:
        return 1, 1
    return t.reshape(-1, t.shape[-1]).shape


def measure(a, x):
    rows, cols = _rows_cols(a)
    af = a.reshape(rows, cols) if a.ndim else a.reshape(1, 1)
    xf = x.reshape(rows, cols) if x.ndim else x.reshape(1, 1)
    res = {"total": int(rows * cols)}
    for tag, tol in (("1e-4", ATOL_RTOL), ("1e-2", ATOL_RTOL_BENCH)):
        res[tag] = {"over": 0, "rho_max": 0.0, "maxabs": 0.0}
    step = max(1, ELEM_CHUNK // max(1, cols))
    for i in range(0, rows, step):
        ac = af[i:i + step].double()
        xc = xf[i:i + step]
        d = (ac - xc).abs()
        for tag, tol in (("1e-4", ATOL_RTOL), ("1e-2", ATOL_RTOL_BENCH)):
            s = res[tag]
            thr = tol + tol * xc.abs()
            s["over"] += int((d > thr).sum())
            s["maxabs"] = max(s["maxabs"], float(d.max()))
            s["rho_max"] = max(s["rho_max"], float((d / thr).max()))
        del ac, d
    for tag in ("1e-4", "1e-2"):
        s = res[tag]
        s["frac_over"] = s["over"] / float(res["total"]) if res["total"] else None
    return res


def profile_kernels(model, inputs):
    model = model.to(device="cuda", dtype=torch.float32)
    model.eval()
    ci = [cast_inputs(x, torch.float32, "cuda") for x in inputs]
    with torch.no_grad():
        model(*ci)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            model(*ci)
            torch.cuda.synchronize()
    names = []
    for evt in prof.key_averages():
        if evt.self_device_time_total > 0 and not MEM_RE.search(evt.key):
            names.append((evt.key, evt.self_device_time_total))
    names.sort(key=lambda t: -t[1])
    return [n for n, _ in names[:6]]


def resolve(level, pid):
    pat = os.path.join(KB, "KernelBench", "level%d" % level, "%d_*.py" % pid)
    hits = sorted(glob.glob(pat))
    if len(hits) != 1:
        raise RuntimeError("resolve %d/%d -> %d hits" % (level, pid, len(hits)))
    return hits[0]


def main():
    torch.cuda.init()
    dev = "cuda"
    t_start = time.time()
    prereg_sha = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    s80 = json.load(open(S80))
    targets = [(int(a), int(b)) for a, b in s80["P4"]["tuples"]]
    s80_label = {}
    for r in (json.loads(l) for l in open(os.path.join(LAB, "results", "s80_instance_rule.jsonl"))):
        if r.get("status") == "ok":
            s80_label[(r["level"], r["problem_id"])] = r.get("r1_top1")
    print("targets", len(targets), targets, flush=True)
    rows = []
    for level, pid in targets:
        path = resolve(level, pid)
        torch.manual_seed(0)
        torch.cuda.empty_cache()
        mod = load_task(path)
        init_args = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
        torch.manual_seed(0)
        model = mod.Model(*init_args)
        torch.manual_seed(SEED_BASE)
        probe_inputs = mod.get_inputs()
        kernels = profile_kernels(model, probe_inputs)
        live_top1 = kernels[0] if kernels else None
        verified = bool(live_top1 and NEEDLE in live_top1.lower())
        agree = bool(live_top1 and s80_label.get((level, pid)) == live_top1)
        print("tuple L%d/%d verified=%s agree_s80=%s top=%s" % (level, pid, verified, agree,
                                                                 (live_top1 or "")[:90]), flush=True)
        del probe_inputs
        torch.cuda.empty_cache()
        for i in range(N_DRAWS):
            seed = SEED_BASE + i
            torch.manual_seed(seed)
            inputs = mod.get_inputs()
            rec = {"level": level, "problem_id": pid, "name": os.path.basename(path),
                   "draw_seed": seed, "engine_verified": verified,
                   "live_top1": live_top1, "s80_r1_top1": s80_label.get((level, pid)),
                   "label_agrees_s80": agree,
                   "prereg_sha": prereg_sha, "producer": "s80b_dgrad_family_draw"}
            try:
                signal.alarm(CFG_ALARM_S)
                set_cfg(True, False)
                A = run_model(model, inputs, torch.float32, dev)
                if A is None:
                    raise RuntimeError("no tensor output")
                set_cfg(False, False)
                X = run_model(model, inputs, torch.float64, dev)
                signal.alarm(0)
                rec["out_shape"] = list(A.shape)
                rec.update(measure(A, X))
                rec["status"] = "ok"
                del A, X
            except Exception as e:
                signal.alarm(0)
                rec["status"] = "error:%s" % type(e).__name__
                rec["error"] = str(e)[:300]
                print("  draw %d ERROR %s" % (seed, e), flush=True)
            torch.cuda.empty_cache()
            rows.append(rec)
            print("  L%d/%d draw %d rho=%s frac=%s" % (level, pid, seed,
                  rec.get("1e-4", {}).get("rho_max"), rec.get("1e-4", {}).get("frac_over")), flush=True)

    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    per = {}
    for level, pid in targets:
        key = "L%d/%d" % (level, pid)
        rs = [r for r in rows if r["level"] == level and r["project_id"] is None] if False else \
             [r for r in rows if r["level"] == level and r["problem_id"] == pid and r["status"] == "ok"]
        if len(rs) < 4:
            per[key] = {"n_ok": len(rs)}
            continue
        rhos = [r["1e-4"]["rho_max"] for r in rs]
        fr = [r["1e-4"]["frac_over"] for r in rs]
        lo, hi = min(rhos), max(rhos)
        per[key] = {
            "n_ok": len(rs), "engine_verified": rs[0]["engine_verified"],
            "label_agrees_s80": rs[0]["label_agrees_s80"], "live_top1": rs[0]["live_top1"],
            "rho_min": lo, "rho_max": hi, "rho_median": sorted(rhos)[len(rhos) // 2],
            "draw_spread_max_over_min": (hi / lo) if lo > 0 else None,
            "frac_over_1e-4_min": min(fr), "frac_over_1e-4_max": max(fr),
            "frac_over_1e-2_max": max(r["1e-2"]["frac_over"] for r in rs),
        }
    spreads = [v["draw_spread_max_over_min"] for v in per.values()
               if v.get("draw_spread_max_over_min") and v.get("n_ok", 0) >= 4]
    if not spreads:
        verdict = "UNDERPOWERED"
    elif all(s <= 1.5 for s in spreads):
        verdict = "DRAW_INVARIANT_ALL"
    elif any(s > 2.0 for s in spreads):
        verdict = "DRAW_DEPENDENT"
    else:
        verdict = "DRAW_PARTIAL"
    max_spread = max(spreads) if spreads else None
    ratio = (ACROSS_SPAN / max_spread) if max_spread else None
    s1 = "SHAPE_DOMINATES_AT_FAMILY_SCOPE" if (ratio is not None and ratio >= 4.0) else "FAMILY_SCOPE_AMBIGUOUS"
    verified_n = sum(1 for v in per.values() if v.get("engine_verified"))
    agree_n = sum(1 for v in per.values() if v.get("label_agrees_s80"))
    summary = {"prereg_sha": prereg_sha, "n_draws": N_DRAWS, "n_targets": len(targets),
               "per_tuple": per, "verdict": verdict,
               "S1": {"across_span_s80": ACROSS_SPAN, "max_draw_spread": max_spread,
                      "shape_to_draw_ratio": ratio, "read": s1},
               "S2": {"tuples_verified_dgrad": verified_n, "tuples_agreeing_with_s80_label": agree_n,
                      "frac": (verified_n / len(targets)) if targets else None},
               "wall_s": round(time.time() - t_start, 1),
               "limits": ["6 seeds per tuple", "one draw generator per task", "one profiler pass per tuple",
                          "GB10 sm_120", "torch 2.14.1+cu130"]}
    json.dump(summary, open(SUMOUT, "w"), indent=1)
    print("VERDICT", verdict, "S1", s1, ratio, "S2", verified_n, agree_n, flush=True)
    print(json.dumps(per, indent=1), flush=True)


if __name__ == "__main__":
    main()
