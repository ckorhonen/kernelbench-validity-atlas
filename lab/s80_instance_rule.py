#!/usr/bin/env python
"""s80_instance_rule.py — re-emit the engine-instance table under ONE declared ranking rule.

PREREG (frozen before the run; do not edit after).

Why: s79d found that s75's recorded kernel_top1 does not reproduce as a self-time top-1 on 4 of the 5 dgrad
tuples it re-profiled live. s75's docstring declares "top-5 CUDA kernels by self device time"; its producer
(profile_call) in fact ranks every profiler key_averages entry with self_device_time_total > 0, which includes
device Memcpy/Memset entries. For short reference calls the input HtoD copy dominates, so the label becomes a
data-movement event. s79c then regrouped 54 rows by that label and published the top-1 instance table. This
script re-measures the population under two declared rules in the same pass so the table is emitted once, with
the rule stated, and the size of the defect is measured rather than asserted.

Population (rule, not outcome): every kind=exceed row of results/s73c_merged_56.jsonl (54 rows). Controls are
not needed here; s75 already established the instrument check.

Per task: torch.manual_seed(0); instantiate mod.Model(*get_init_inputs()); mod.get_inputs(); benchmark default
flags (cudnn allow_tf32 True, matmul allow_tf32 False); one reference forward under torch.profiler with CUDA
activity; collect every key_averages entry with self_device_time_total > 0 (name, self device us).

DECLARED RULES (both computed from the same pass):
  R0_all_events  = rank ALL device entries by self device time, descending. This is the rule s75's producer
                   actually implemented; it is recomputed here to test whether the recorded field reproduces.
  R1_compute     = rank only entries that are compute kernels: exclude any key matching
                   ^(Memcpy|Memset) (case-insensitive) or containing "Memcpy"/"Memset". Top-1 under R1 is the
                   engine instance that performs the reference computation.

Recorded per row: top-5 names and us under R1; top-1 under R0; whether a memcpy/memset is present and its total
us; the recorded kernel_top1 from s75; the s75 bias ratio; op_class; level/problem_id; output shape.

PREDECLARED READS (analysis in this script, after the run):
  P1 rule identification: fraction of rows where s75's recorded kernel_top1 equals the R0 top-1. If >= 0.90 the
     producer's rule is identified as all-device-event ranking, i.e. a documentation defect in s75, and s79c's
     re-read is correct in method but inherited the wrong ranking.
  P2 defect size: fraction of rows whose R1 top-1 differs from the recorded kernel_top1, and the count whose
     recorded top-1 is a non-compute event (Memcpy/Memset) or a layout/elementwise kernel. If > 25% differ, the
     instance table published as rOZ7mxOvxdfVUWIf and corrected as gaJoW3UP01V_ISRO is not reproducible under a
     declared compute-kernel rule and is retracted to label-dependent grouping.
  P3 instance compression: under R1, count instances, multi-row instances, and the fraction of multi-row
     instances whose bias span is <= 5x; report the largest multi-row span. Frozen verdict:
       COMPRESSION_SURVIVES  if >= 70% of multi-row instances are <= 5x AND the largest multi-row span <= 2608x
                             (one decade below the 26,083x within-class span s75b measured);
       COMPRESSION_WEAK      if the 70% clause holds but the largest span exceeds 2608x, or 50-70% are <= 5x;
       NO_COMPRESSION        otherwise.
     Rationale: s75b's "six of seven multi-row instances reproduce within 5x" was computed on containment
     grouping, so it must be recomputed on a declared rule before it is cited again.
  P4 dgrad group: the rows whose R1 top-1 contains "dgrad"; n and bias span. If this set differs from s79c's
     five tuples (level1/58, 1/68, 1/73, 2/16, 2/77), print DRAW_RERUN_NEEDED with the added and removed tuples.

KILL CLAUSES (predeclared):
  K1 if fewer than 80% of the 54 tasks yield a labelled pass, the population is narrowed and stated as such.
  K2 if P1 < 0.50 (the recorded labels are not reproducible under either rule), then no rule explains s75 and
     the kernel_top1 field is declared UNREPRODUCIBLE; instance grouping is then unusable and only the
     per-shape budget column is cited downstream.

Limits declared: one seed per task, one profiler pass per task, profiler-instrumented runs only (no timing
claims from this artifact), GB10 sm_120 / torch 2.14.1+cu130 / rev 423217d.
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
OUT = os.path.join(LAB, "results", "s80_instance_rule.jsonl")
SUMOUT = os.path.join(LAB, "results", "s80_instance_rule_summary.json")
S73C = os.path.join(LAB, "results", "s73c_merged_56.jsonl")
S75 = os.path.join(LAB, "results", "s75_engine_class_s73pop.jsonl")
S79C = os.path.join(LAB, "results", "s79c_instance_regroup.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

CFG_ALARM_S = 300
GLOBAL_DEADLINE_S = 2400
MEM_RE = re.compile(r"memcpy|memset", re.I)
NEEDLE_DGRAD = "dgrad"
S79C_ROWS = [(1, 58), (1, 68), (1, 73), (2, 16), (2, 77)]


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
    return evts


def main():
    torch.cuda.init()
    dev = "cuda"
    t_start = time.time()

    rows73 = [json.loads(l) for l in open(S73C)]
    pop = [r for r in rows73 if r.get("kind") == "exceed"]
    s75 = {(r["level"], r["problem_id"]): r for r in (json.loads(l) for l in open(S75))}
    print("population", len(pop), flush=True)

    out, excluded = [], []
    for r in pop:
        if time.time() - t_start > GLOBAL_DEADLINE_S:
            excluded.append({"level": r["level"], "problem_id": r["problem_id"], "reason": "global deadline"})
            continue
        level, pid, fname = r["level"], r["problem_id"], r["name"]
        path = os.path.join(KB, "KernelBench", "level%d" % level, fname)
        r75 = s75.get((level, pid), {})
        rec = {
            "level": level, "problem_id": pid, "name": fname,
            "op_class": r.get("op_class") or r75.get("op_class"),
            "output_shape": r.get("s71_output_shape") or r75.get("output_shape"),
            "s75_kernel_top1": r75.get("kernel_top1"),
            "s75_bias_1e4": r75.get("s73_bias_1e4"),
        }
        t_task = time.time()
        try:
            torch.manual_seed(0)
            torch.cuda.empty_cache()
            mod = load_task(path)
            init_args = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
            inputs = mod.get_inputs()
            torch.manual_seed(0)
            model = mod.Model(*init_args).to(device=dev, dtype=torch.float32)
            model.eval()

            def fn():
                with torch.no_grad():
                    return model(*[cast_inputs(x, torch.float32, dev) for x in inputs])

            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = False
            signal.alarm(CFG_ALARM_S)
            evts = profile_call(fn)
            signal.alarm(0)

            r0 = evts[0] if evts else (None, 0.0)
            compute = [(n, u) for n, u in evts if not MEM_RE.search(n)]
            mem = [(n, u) for n, u in evts if MEM_RE.search(n)]
            rec["r0_top1"] = r0[0]
            rec["r0_top1_us"] = round(r0[1], 3)
            rec["r1_top1"] = compute[0][0] if compute else None
            rec["r1_top1_us"] = round(compute[0][1], 3) if compute else None
            rec["r1_top5"] = [{"name": n, "us": round(u, 3)} for n, u in compute[:5]]
            rec["memcpy_us_total"] = round(sum(u for _, u in mem), 3)
            rec["n_device_events"] = len(evts)
            rec["n_compute_events"] = len(compute)
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
        print(level, pid, rec.get("status"), (rec.get("r1_top1") or "")[:60],
              rec.get("r1_top1_us"), rec.get("t_task_s"), flush=True)

    ok = [r for r in out if r.get("status") == "ok"]
    n_pop = len(pop)
    summary = {
        "prereg": "docstring of this file, frozen pre-run",
        "population_attempted": n_pop, "rows_written": len(out), "ok": len(ok),
        "excluded": excluded,
        "coverage": (len(ok) / n_pop) if n_pop else None,
        "K1_fired": bool(len(ok) < 0.8 * n_pop),
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_s": round(time.time() - t_start, 1),
    }

    # P1: does the recorded field reproduce under the all-device-event rule?
    match_r0 = [r for r in ok if r.get("s75_kernel_top1") and r.get("s75_kernel_top1") == r.get("r0_top1")]
    summary["P1_recorded_equals_R0"] = len(match_r0)
    summary["P1_frac"] = len(match_r0) / len(ok) if ok else None
    # P2: defect size
    differ = [r for r in ok if r.get("s75_kernel_top1") and r.get("r1_top1")
              and r["s75_kernel_top1"] != r["r1_top1"]]
    memlab = [r for r in ok if r.get("s75_kernel_top1") and MEM_RE.search(r["s75_kernel_top1"])]
    summary["P2_recorded_differs_from_R1"] = len(differ)
    summary["P2_frac_differs"] = len(differ) / len(ok) if ok else None
    summary["P2_recorded_is_memcpy_or_memset"] = len(memlab)
    summary["P2_moved_rows"] = [{"level": r["level"], "problem_id": r["problem_id"],
                                 "recorded": r.get("s75_kernel_top1"),
                                 "r1": r.get("r1_top1")} for r in differ]

    # P3: instance table under R1
    groups = {}
    for r in ok:
        k = r.get("r1_top1")
        if not k:
            continue
        groups.setdefault(k, []).append(r)
    inst = []
    for k, rs in groups.items():
        biases = [r["s75_bias_1e4"] for r in rs if r.get("s75_bias_1e4")]
        span = (max(biases) / min(biases)) if len(biases) >= 2 and min(biases) > 0 else None
        inst.append({"name": k, "n": len(rs), "span": span,
                     "median": sorted(biases)[len(biases) // 2] if biases else None,
                     "rows": [{"level": r["level"], "problem_id": r["problem_id"],
                               "op_class": r.get("op_class"), "bias": r.get("s75_bias_1e4")} for r in rs]})
    inst.sort(key=lambda d: -d["n"])
    multi = [d for d in inst if d["n"] >= 2]
    within5 = [d for d in multi if d["span"] is not None and d["span"] <= 5.0]
    largest = max([d["span"] for d in multi if d["span"] is not None], default=None)
    frac5 = (len(within5) / len(multi)) if multi else None
    if frac5 is not None and frac5 >= 0.70 and largest is not None and largest <= 2608.0:
        verdict = "COMPRESSION_SURVIVES"
    elif frac5 is not None and (frac5 >= 0.50):
        verdict = "COMPRESSION_WEAK"
    else:
        verdict = "NO_COMPRESSION"
    summary["P3"] = {"instances_total": len(inst), "multi_row_instances": len(multi),
                     "multi_within_5x": len(within5), "frac_within_5x": frac5,
                     "largest_multi_span": largest, "verdict": verdict,
                     "table": inst}
    # P4: dgrad group under R1
    dg = [r for r in ok if r.get("r1_top1") and NEEDLE_DGRAD in r["r1_top1"].lower()]
    dg_bias = [r["s75_bias_1e4"] for r in dg if r.get("s75_bias_1e4")]
    dg_span = (max(dg_bias) / min(dg_bias)) if len(dg_bias) >= 2 and min(dg_bias) > 0 else None
    dg_set = sorted({(r["level"], r["problem_id"]) for r in dg})
    added = sorted(set(dg_set) - set(S79C_ROWS))
    removed = sorted(set(S79C_ROWS) - set(dg_set))
    summary["P4"] = {"n": len(dg), "span": dg_span, "tuples": dg_set,
                     "added_vs_s79c": added, "removed_vs_s79c": removed,
                     "flag": "DRAW_RERUN_NEEDED" if (added or removed) else "MEMBERSHIP_UNCHANGED"}
    summary["K2_fired"] = bool(summary["P1_frac"] is not None and summary["P1_frac"] < 0.50)
    if summary["K2_fired"]:
        summary["K2_consequence"] = "kernel_top1 UNREPRODUCIBLE; instance grouping unusable; per-shape column stands alone"
    with open(SUMOUT, "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "P3"}, indent=1))
    print("P3", {k: v for k, v in summary["P3"].items() if k != "table"})


if __name__ == "__main__":
    main()
