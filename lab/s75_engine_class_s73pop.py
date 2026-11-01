#!/usr/bin/env python
"""s75_engine_class_s73pop.py — engine-class labels for the population where the reference-deviation
spread was measured (s73/s74), plus the tree-identity descriptors available from the same profiler run.

PREREG (frozen before the run; do not edit after).

Motivation: s74 measured a 301x spread in per-task reference budget consumption across 42 joined rows
inside op classes conv/conv_transpose, with within-class geometry explaining none of it (LOO R2 -0.99 /
-0.19). Its declared limit: "no engine-class label for this population (s63 covers a different 56
rows), so op_class is the class proxy." The reflection EVB-Vd2GyUZ-0cNi named the missing bridge: what
sets the magnitude of the reference's own deviation INSIDE an engine class, and does a verdict follow
class membership or accumulation-tree identity. This producer supplies the label set that s74 lacked.

Population (rule, not outcome): every row of results/s73c_merged_56.jsonl (54 kind=exceed + 8
kind=control), i.e. the exact population whose bias ratios s74 analysed. No sampling, no exclusions by
outcome.

Per task: instantiate the model once with torch.manual_seed(0) (s70 lesson), generate inputs with
mod.get_inputs(), set the benchmark's shipped default flags (cudnn allow_tf32 True, matmul allow_tf32
False), run the reference forward once under torch.profiler (CUDA activity only), and record the top-5
CUDA kernels by self device time.

Labels:
  engine_class_primary = s63's ordered patterns applied verbatim to the recorded kernel names (frozen,
      comparable with s63/s66/s72).
  engine_class_refined = descriptive secondary: a name carrying dgrad/dgemm/dconv is FP64 only when it
      also carries a double/fp64 marker, otherwise it is classified by its precision signature
      (tf32|tensorop|1688 => TF32_TENSOROP, winograd => WINOGRAD, else FP32_SIMT). Declared because
      conv_transpose forward maps to a dgrad kernel whose name can contain "dgrad" without being fp64.
  kernel_top1 = highest-self-time kernel name; kernel_names = full top-5 list with device microseconds.

Predeclared reads (the analysis script s75b performs them; no thresholds invented after the run):
  R1 class distribution over the 54 exceed rows and the 8 controls, by op_class.
  R2 if >=80% of the exceed rows carry ONE primary class => class membership cannot partition the
     spread; the magnitude question narrows to within-class tree/shape idiosyncrasy (the reflection's
     reconsider-if).
  R3 count distinct top-1 kernel names among the exceed rows. If >=2 distinct names exist inside one
     class, report whether bias clusters by name; if names are unique per shape, report that the tree
     instance varies per shape and that identity is a per-shape property (hypothesis, not a law).
  R4 controls must show reference budget ratio <= 0.10 at 1e-4 in the s73 records (instrument check,
     same bar as s73/s74).

Kill clauses (predeclared):
  K1 if fewer than 80% of the 62 tasks yield a profiler label, the population is narrowed to the
     labelled tasks and stated as such; no claim beyond them.
  K2 if every exceed row collapses to a single primary class, the class-explained test is vacuous by
     construction and is reported as the null (not as a failure).

Limits declared: one seed per task, single profiler pass per task, no timing claims from this artifact
(profiler-instrumented runs only), engine labels are observations of this GB10/cudnn stack at rev
423217d.
"""
import importlib.util
import json
import os
import re
import signal
import time

import torch

KB = os.path.expanduser("~/kernelbench/KernelBench")
LAB = os.path.expanduser("~/kb-lab")
OUT = os.path.join(LAB, "results", "s75_engine_class_s73pop.jsonl")
SUMOUT = os.path.join(LAB, "results", "s75_engine_class_s73pop_summary.json")
S73C = os.path.join(LAB, "results", "s73c_merged_56.jsonl")
S71 = os.path.join(LAB, "results", "s71_full_audit.jsonl")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

CFG_ALARM_S = 240
GLOBAL_DEADLINE_S = 2100

# s63's classifier, verbatim and frozen.
CLASS_PATTERNS = [
    ("WINOGRAD", re.compile(r"winograd", re.I)),
    ("TF32_TENSOROP", re.compile(r"tf32|tensorop|1688|16816|h1688", re.I)),
    ("FP64", re.compile(r"dgrad|dgemm|dconv|fp64|double|_d_", re.I)),
    ("FP32_SIMT", re.compile(r"fprop|implicit_gemm|sgemm|s1688|fp32|scudnn|simt", re.I)),
    ("CUTLASS", re.compile(r"cutlass", re.I)),
]
FP64_MARK = re.compile(r"fp64|double|_d_|dgemm|dconv", re.I)
TF32_MARK = re.compile(r"tf32|tensorop|1688|16816|h1688", re.I)
WINO_MARK = re.compile(r"winograd", re.I)


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


def classify_primary(names):
    for lab, pat in CLASS_PATTERNS:
        for n in names:
            if pat.search(n):
                return lab, n
    return ("OTHER", names[0] if names else None)


def classify_refined(names):
    for n in names:
        if FP64_MARK.search(n):
            return "FP64", n
    for n in names:
        if TF32_MARK.search(n):
            return "TF32_TENSOROP", n
    for n in names:
        if WINO_MARK.search(n):
            return "WINOGRAD", n
    return ("FP32_SIMT_OR_OTHER", names[0] if names else None)


def profile_call(fn):
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


def main():
    torch.cuda.init()
    dev = "cuda"
    t_start = time.time()

    rows73 = [json.loads(l) for l in open(S73C)]
    rows71 = {(r["level"], r["problem_id"]): r for r in (json.loads(l) for l in open(S71))}
    pop = [r for r in rows73 if r.get("kind") in ("exceed", "control")]
    print("population", len(pop), "exceed", sum(1 for r in pop if r["kind"] == "exceed"),
          "controls", sum(1 for r in pop if r["kind"] == "control"), flush=True)

    out, excluded = [], []
    for r in pop:
        if time.time() - t_start > GLOBAL_DEADLINE_S:
            excluded.append({"level": r["level"], "problem_id": r["problem_id"], "reason": "global deadline"})
            continue
        level, pid, fname = r["level"], r["problem_id"], r["name"]
        path = os.path.join(KB, "KernelBench", "level%d" % level, fname)
        r71 = rows71.get((level, pid), {})
        rec = {
            "kind": r["kind"], "level": level, "problem_id": pid, "name": fname,
            "op_class": r.get("op_class") or r71.get("op_class"),
            "s73_bias_1e4": (r.get("decomp_c2_tf32off") or {}).get("1e-4", {}).get("ratio_ref_median"),
            "s73_ref_over_budget_frac": None,
            "a_vs_b_bit_identical": r71.get("A_vs_B_bit_identical"),
            "output_shape": r71.get("output_shape"),
            "input_shapes": r71.get("input_shapes"),
            "n_params": r71.get("n_params"),
        }
        d = (r.get("decomp_c2_tf32off") or {}).get("1e-4", {})
        if d.get("total"):
            rec["s73_ref_over_budget_frac"] = d["ref_over_budget"] / d["total"]
        t_task = time.time()
        try:
            torch.manual_seed(0)
            torch.cuda.empty_cache()
            mod = load_task(path)
            init_args = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
            inputs = mod.get_inputs()
            torch.manual_seed(0)
            model = mod.Model(*init_args)
            model = model.to(device=dev, dtype=torch.float32)
            model.eval()

            def fn():
                with torch.no_grad():
                    return model(*[cast_inputs(x, torch.float32, dev) for x in inputs])

            signal.alarm(CFG_ALARM_S)
            evts = profile_call(fn)
            signal.alarm(0)
            names = [n for n, _ in evts]
            prim, prim_name = classify_primary(names)
            ref, ref_name = classify_refined(names)
            rec["engine_class_primary"] = prim
            rec["engine_class_refined"] = ref
            rec["kernel_top1"] = names[0] if names else None
            rec["kernel_top1_primary"] = prim_name
            rec["kernel_top1_refined"] = ref_name
            rec["kernels"] = [{"name": n, "us": round(u, 3)} for n, u in evts]
            rec["n_kernels_profiled"] = len(names)
            rec["status"] = "ok"
        except Timeout:
            rec["status"] = "timeout"
        except Exception as e:
            rec["status"] = "error"
            rec["error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
        rec["t_task_s"] = round(time.time() - t_task, 2)
        out.append(rec)
        with open(OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(rec["kind"], level, pid, rec.get("status"), rec.get("engine_class_primary"),
              rec.get("engine_class_refined"), rec.get("t_task_s"), flush=True)

    ok = [r for r in out if r.get("status") == "ok"]
    exc = [r for r in ok if r["kind"] == "exceed"]
    ctl = [r for r in ok if r["kind"] == "control"]
    summary = {
        "prereg": "docstring of this file, frozen pre-run",
        "population_attempted": len(pop), "rows_written": len(out), "ok": len(ok),
        "exceed_ok": len(exc), "control_ok": len(ctl), "excluded": excluded,
        "label_coverage": (len(ok) / len(pop)) if pop else None,
        "K1_fired": bool(len(ok) < 0.8 * len(pop)),
        "class_counts_primary_exceed": {}, "class_counts_refined_exceed": {},
        "class_counts_primary_control": {}, "class_counts_refined_control": {},
        "distinct_top1_names_exceed": None,
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_s": round(time.time() - t_start, 1),
    }
    for key, field, grp in (("class_counts_primary_exceed", "engine_class_primary", exc),
                            ("class_counts_refined_exceed", "engine_class_refined", exc),
                            ("class_counts_primary_control", "engine_class_primary", ctl),
                            ("class_counts_refined_control", "engine_class_refined", ctl)):
        for r in grp:
            v = r.get(field) or "UNLABELLED"
            summary[key][v] = summary[key].get(v, 0) + 1
    summary["distinct_top1_names_exceed"] = len({r.get("kernel_top1") for r in exc})
    summary["distinct_top1_names_control"] = len({r.get("kernel_top1") for r in ctl})
    with open(SUMOUT, "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
