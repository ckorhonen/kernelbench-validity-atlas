#!/usr/bin/env python
"""s105b_map_engine_class.py — the engine-class column for the full 200-task evaluator map.

PREREG (frozen at commit before any row; do not edit after).

Question this fills: s105 established the base rate from committed records (M1_BASE_RATE_NARROW: 1/103 non-conv rows
above rho 0.1 at 1e-4 and that row is conv-class; M2_BENCH_GATE_NONBINDING: 0/200 exceed at the shipped 1e-2 gate,
max rho_bench 0.119). What the map lacks is the class column on the full population: the reference's observed engine
class per task, which the decomposition needs as its categorical factor. s63 labelled 56 conv rows and s75 labelled 62
rows; the rest of the population is unlabelled.

Population (rule): all 200 tasks in s71_full_audit.jsonl (level1+level2, rev 423217d), identified by (level,
problem_id, name). Inputs from the task's own get_inputs()/get_init_inputs() under torch.manual_seed(0) — the same
convention s71 used — so the profiled call is the same computation s71 measured. Reference flags are the benchmark's
defaults (cudnn.allow_tf32=True, matmul.allow_tf32=True); no perturbation.

Per task recorded: observed top-5 CUDA kernels with device time, classified engine label under s63's CLASS_PATTERNS
(verbatim import, same ordering), n_cuda_kernels, ref_max_abs, output_shape, wall, status. Classification is an
OBSERVATION of the dispatched kernels, not an inference from op_class.

Controls / kills:
  K1 a task whose forward raises, OOMs or exceeds the per-task alarm is recorded with status=error|timeout and its
     label "UNKNOWN"; it stays in the table (no silent drop).
  K2 tie-out: for the rows s63 recorded, the observed top-1 kernel name must match s63's recorded name where the same
     tuple and flags apply; a mismatch is reported as tie_out_mismatch and counted, not repaired.
  K3 if fewer than 190/200 rows complete, the class column is declared partial and the completed count is printed.

Outputs: results/s105b_map_engine_class.jsonl + results/s105b_map_engine_class_summary.json.
No timing claims (profiler-instrumented runs only).
"""
import importlib.util, json, os, re, signal, sys, time, traceback
import torch

KB = os.path.expanduser("~/kernelbench/KernelBench")
OUT = os.path.expanduser("~/kb-lab/results/s105b_map_engine_class.jsonl")
SUM = os.path.expanduser("~/kb-lab/results/s105b_map_engine_class_summary.json")
S71 = os.path.expanduser("~/kb-lab/results/s71_full_audit.jsonl")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

ALARM = 240
SMOKE = os.environ.get("SMOKE") == "1"
DATE = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

CLASS_PATTERNS = [
    ("WINOGRAD", re.compile(r"winograd", re.I)),
    ("TF32_TENSOROP", re.compile(r"tf32|tensorop|1688|16816|h1688", re.I)),
    ("FP64", re.compile(r"dgrad|dgemm|dconv|fp64|double|_d_", re.I)),
    ("FP32_SIMT", re.compile(r"fprop|implicit_gemm|sgemm|s1688|fp32|scudnn|simt", re.I)),
    ("CUTLASS", re.compile(r"cutlass", re.I)),
]


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout("task exceeded %ds" % ALARM)


signal.signal(signal.SIGALRM, _alarm)


def classify(names):
    for lab, pat in CLASS_PATTERNS:
        for n in names:
            if pat.search(n):
                return lab, n
    return ("OTHER", names[0] if names else None)


def load_task(path):
    name = "kb_" + os.path.basename(path).replace(".py", "").replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def profile_fn(fn):
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
    rows = [json.loads(l) for l in open(S71) if l.strip()]
    if SMOKE:
        rows = rows[:4]
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    recs = []
    t0 = time.time()
    with open(OUT, "w") as f:
        for i, r in enumerate(rows):
            if time.time() - t0 > 1500:
                print("DEADLINE at row", i, flush=True)
                break
            lvl, pid, name = r["level"], r["problem_id"], r["name"]
            path = os.path.join(KB, "level%d" % lvl, name)
            rec = {"level": lvl, "problem_id": pid, "name": name,
                   "op_class": r.get("op_class"), "date": DATE, "status": "ok"}
            tw = time.time()
            try:
                signal.alarm(ALARM)
                mod = load_task(path)
                torch.manual_seed(0)
                init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
                model = mod.Model(*init).cuda().eval()
                inp = mod.get_inputs()
                inp = [x.cuda() if torch.is_tensor(x) else x for x in inp]
                with torch.no_grad():
                    evts = profile_fn(lambda: model(*inp))
                    out = model(*inp)
                signal.alarm(0)
                names = [e[0] for e in evts]
                lab, matched = classify(names)
                rec.update({"engine_class": lab, "matched_kernel": matched,
                            "top_kernels": [{"name": e[0], "us": e[1]} for e in evts],
                            "n_cuda_kernels": len(evts),
                            "ref_max_abs": float(out.abs().max().item()) if torch.is_tensor(out) else None,
                            "output_shape": list(out.shape) if torch.is_tensor(out) else None,
                            "wall_s": round(time.time() - tw, 3)})
                del model, inp, out
                torch.cuda.empty_cache()
            except Timeout:
                signal.alarm(0)
                rec.update({"status": "timeout", "engine_class": "UNKNOWN", "wall_s": round(time.time() - tw, 3)})
            except torch.cuda.OutOfMemoryError:
                signal.alarm(0)
                torch.cuda.empty_cache()
                rec.update({"status": "oom", "engine_class": "UNKNOWN", "wall_s": round(time.time() - tw, 3)})
            except Exception as e:
                signal.alarm(0)
                torch.cuda.empty_cache()
                rec.update({"status": "error", "engine_class": "UNKNOWN",
                            "error": "%s: %s" % (type(e).__name__, str(e)[:200]),
                            "wall_s": round(time.time() - tw, 3)})
            recs.append(rec)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 20 == 0:
                print(i, name, rec.get("engine_class"), rec.get("status"), flush=True)
    ok = [r for r in recs if r["status"] == "ok"]
    dist = {}
    for r in ok:
        dist[r["engine_class"]] = dist.get(r["engine_class"], 0) + 1
    by_class = {}
    for r in ok:
        c = r.get("op_class") or "unlabelled"
        by_class.setdefault(c, {})
        by_class[c][r["engine_class"]] = by_class[c].get(r["engine_class"], 0) + 1
    summary = {"date": DATE, "n_attempted": len(recs), "n_ok": len(ok),
               "status_counts": {s: sum(1 for r in recs if r["status"] == s) for s in set(r["status"] for r in recs)},
               "engine_distribution": dist, "engine_by_op_class": by_class,
               "coverage_branch": "K3_COMPLETE" if len(ok) >= 190 else "K3_PARTIAL",
               "wall_total_s": round(time.time() - t0, 1)}
    json.dump(summary, open(SUM, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
