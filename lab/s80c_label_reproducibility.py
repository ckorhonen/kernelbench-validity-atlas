#!/usr/bin/env python
"""s80c_label_reproducibility.py — is the R1 engine-instance label reproducible across profiler passes?

PREREG (frozen before the run; do not edit after).

s80b's declared secondary read S2 came out 0/14 tuples agreeing with s80's R1 label. That number is not usable
as measured: s80b's profile_kernels() never set the benchmark flags, so every pass after the first inherited
tf32-off from the preceding draw loop, while s80 profiled with the benchmark defaults (cudnn tf32 on). The
disagreement is therefore confounded by flag state. The draw measurements in s80b are unaffected (each draw set
the flags explicitly before arm A).

This script isolates the question the label needs answered: under ONE fixed flag state (the benchmark defaults:
cudnn allow_tf32 True, matmul allow_tf32 False), is the compute-kernel top-1 of the reference call the same on
two consecutive profiler passes, and does it match s80's recorded label?

Population (rule, frozen): the 14 (level, problem_id) pairs in results/s80_instance_rule_summary.json
["P4"]["tuples"]. Per tuple: model weights fixed once (init seed 0), one input draw (seed 300), two consecutive
torch.profiler passes on the same tensors with CUDA activity only, compute kernels ranked by self device time
(memcpy/memset excluded, identical to s80's R1).

FROZEN READS:
  L1 pass-to-pass stability: fraction of tuples whose pass-A top-1 equals pass-B top-1.
  L2 agreement with s80: fraction whose pass-A top-1 equals s80's r1_top1 for that tuple.
  L3 flag sensitivity: fraction of tuples whose pass-A top-1 equals s80b's recorded live_top1 (which was taken
     under tf32-off for 13/14 tuples). Reported as the confound's size, not as a defect of either producer.
FROZEN VERDICT:
  LABEL_STABLE      if L1 = 1.0 and L2 >= 0.90;
  LABEL_PASS_DEPENDENT if L1 < 1.0 but L2 >= 0.90;
  LABEL_UNREPRODUCIBLE otherwise.
Consequence written before the run: under LABEL_STABLE the instance table may be cited as a dispatch observation
with its rule named; under LABEL_PASS_DEPENDENT it must be cited as "one-pass observation, ranking unstable
within a pass pair"; under LABEL_UNREPRODUCIBLE instance grouping is dropped and only the per-shape budget
column is cited.

Limits declared: one input draw per tuple, two passes, GB10 sm_120, torch 2.14.1+cu130, one run.
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
OUT = os.path.join(LAB, "results", "s80c_label_reproducibility.json")
S80 = os.path.join(LAB, "results", "s80_instance_rule_summary.json")
S80B = os.path.join(LAB, "results", "s80b_dgrad_family_draw_summary.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

MEM_RE = re.compile(r"memcpy|memset", re.I)
CFG_ALARM_S = 300


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


def profile_top(model, inputs):
    ci = [cast_inputs(x, torch.float32, "cuda") for x in inputs]
    model.eval()
    with torch.no_grad():
        model(*ci)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            model(*ci)
            torch.cuda.synchronize()
    evts = [(e.key, float(e.self_device_time_total)) for e in prof.key_averages()
            if e.self_device_time_total > 0 and not MEM_RE.search(e.key)]
    evts.sort(key=lambda t: -t[1])
    return [{"name": n, "us": round(u, 3)} for n, u in evts[:3]]


def resolve(level, pid):
    pat = os.path.join(KB, "KernelBench", "level%d" % level, "%d_*.py" % pid)
    hits = sorted(glob.glob(pat))
    if len(hits) != 1:
        raise RuntimeError("resolve %d/%d -> %d hits" % (level, pid, len(hits)))
    return hits[0]


def main():
    torch.cuda.init()
    t_start = time.time()
    prereg_sha = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    s80 = json.load(open(S80))
    s80b = json.load(open(S80B))
    targets = [(int(a), int(b)) for a, b in s80["P4"]["tuples"]]
    label80 = {}
    for r in (json.loads(l) for l in open(os.path.join(LAB, "results", "s80_instance_rule.jsonl"))):
        if r.get("status") == "ok":
            label80[(r["level"], r["problem_id"])] = r.get("r1_top1")
    live80b = {k: v.get("live_top1") for k, v in s80b["per_tuple"].items()}

    rows = []
    for level, pid in targets:
        path = resolve(level, pid)
        torch.manual_seed(0)
        torch.cuda.empty_cache()
        mod = load_task(path)
        init_args = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
        torch.manual_seed(0)
        model = mod.Model(*init_args).to(device="cuda", dtype=torch.float32)
        torch.manual_seed(300)
        inputs = mod.get_inputs()
        rec = {"level": level, "problem_id": pid, "name": os.path.basename(path),
               "s80_r1_top1": label80.get((level, pid)),
               "s80b_live_top1": live80b.get("L%d/%d" % (level, pid)),
               "prereg_sha": prereg_sha}
        try:
            signal.alarm(CFG_ALARM_S)
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = False
            a = profile_top(model, inputs)
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = False
            b = profile_top(model, inputs)
            signal.alarm(0)
            rec["pass_a"] = a
            rec["pass_b"] = b
            rec["a_equals_b"] = bool(a and b and a[0]["name"] == b[0]["name"])
            rec["a_equals_s80"] = bool(a and label80.get((level, pid)) == a[0]["name"])
            rec["a_equals_s80b"] = bool(a and live80b.get("L%d/%d" % (level, pid)) == a[0]["name"])
            rec["status"] = "ok"
        except Exception as e:
            signal.alarm(0)
            rec["status"] = "error:%s" % type(e).__name__
            rec["error"] = str(e)[:300]
        del inputs, model
        torch.cuda.empty_cache()
        rows.append(rec)
        print(level, pid, rec.get("status"), rec.get("a_equals_b"), rec.get("a_equals_s80"),
              (rec.get("pass_a") or [{}])[0].get("name", "")[:70], flush=True)

    ok = [r for r in rows if r.get("status") == "ok"]
    l1 = sum(1 for r in ok if r["a_equals_b"]) / len(ok) if ok else None
    l2 = sum(1 for r in ok if r["a_equals_s80"]) / len(ok) if ok else None
    l3 = sum(1 for r in ok if r["a_equals_s80b"]) / len(ok) if ok else None
    if l1 == 1.0 and l2 is not None and l2 >= 0.90:
        verdict = "LABEL_STABLE"
    elif l1 is not None and l1 < 1.0 and l2 is not None and l2 >= 0.90:
        verdict = "LABEL_PASS_DEPENDENT"
    else:
        verdict = "LABEL_UNREPRODUCIBLE"
    summary = {"prereg_sha": prereg_sha, "n_targets": len(targets), "n_ok": len(ok),
               "L1_pass_to_pass": l1, "L2_agreement_with_s80": l2, "L3_agreement_with_s80b": l3,
               "verdict": verdict, "rows": rows,
               "wall_s": round(time.time() - t_start, 1),
               "limits": ["one input draw", "two passes", "GB10 sm_120", "torch 2.14.1+cu130"]}
    json.dump(summary, open(OUT, "w"), indent=1)
    print("VERDICT", verdict, "L1", l1, "L2", l2, "L3", l3, flush=True)


if __name__ == "__main__":
    main()
