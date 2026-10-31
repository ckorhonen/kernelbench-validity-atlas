"""s72 -- fresh-process engine-class replication: is the reference realization a function of
(tuple, flags) or of process state?

PRE-REGISTRATION (frozen BEFORE the first run; this docstring is the registration).

Why: the r19 manuscript says the reference realization is a categorical dispatch decision,
deterministic over the declared tuple (Table 1). s63 observed the engine classes by profiling
the reference call, and s66 showed they are stable across invocation contexts (warm/cold/
benchmark/deterministic/both/tf32-off) inside ONE process lifetime. Neither test crossed a
process boundary. cuDNN's heuristics, its autotune cache and the workspace budget it is
offered are process-scoped state, so "deterministic" is currently a within-process claim.
This probe tests the missing axis with fresh interpreters, one cell per process.

Population (fixed by rule, not by outcome): the same 17 rows s66 used -- the 11 rows of
results/gate_stability.jsonl with armA_delivered_change=true, plus 6 controls taken in
ascending (H,N,C) order at every 7th position of the no-change rows. Reusing s66's row set
makes every cell directly comparable to s63's recorded default class and s66's context A.

Cells: per row, three independent fresh OS processes (subprocess, new interpreter) each
performing, in order: (1) allocate and zero a 256 MiB device buffer (cold L2), (2) seed
1234 + H*100 + N*10 + C (r1/s63/s66 formula), (3) build x and w, (4) one profiled reference
conv2d call with allow_tf32=True, cudnn.benchmark=False, deterministic=False. A fourth cell
per row runs in a fresh process with torch.cuda.set_per_process_memory_fraction(0.10)
(~12 GB of the 121 GB unified pool) and otherwise identical flags -- the constrained-budget
arm. 17 rows x (3 + 1) = 68 cells.

Recorded per cell: classified engine class (s63's CLASS_PATTERNS verbatim), raw top-5
kernels with device times, reference output absmax (input-identity check), the flags, the
process pid and a per-process nonce, and the s63/s66 comparison labels.

Declared reads (two-class, written before the run; P5 -- either outcome is a result):
  R1 (process identity): if on every row all three fresh-process cells carry the same class
     as s63's recorded default class and as each other, the engine class is process-invariant
     on this population and "deterministic over the declared tuple" survives at process
     scope. Kill clause: any row whose three fresh-process classes disagree, or any row that
     disagrees with s63's recorded class, falsifies process invariance; then the published
     word "deterministic" must be scoped to "within a process lifetime" and the per-row labels
     are the result. No aggregate threshold decides wording.
  R2 (budget): if the constrained-allocator cell matches the default cell on all 17 rows, the
     engine choice is budget-invariant here and Table 1 needs no budget term. Any flip makes
     the class budget-conditional and Table 1 must print the budget beside the flags.

No timing claims: profiler-instrumented single calls only. Mechanism is not attributed; if a
flip appears, the per-cell kernel names are the evidence and the cause stays labeled hypothesis.

Run on spark (from ~/kb-lab): ~/kernelbench/.venv/bin/python s72_fresh_process.py
Output: results/s72_fresh_process.jsonl + results/s72_fresh_process_summary.json
"""
import json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
OUTL = os.path.join(HERE, "results", "s72_fresh_process.jsonl")
SUML = os.path.join(HERE, "results", "s72_fresh_process_summary.json")
GATE = os.path.join(HERE, "results", "gate_stability.jsonl")
S63 = os.path.join(HERE, "results", "s63_engine_predicate.jsonl")
S66 = os.path.join(HERE, "results", "s66_foreign_warm.jsonl")
DATE = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
REPS = 3
BUDGET_FRACTION = 0.10

CLASS_PATTERNS = [
    ("WINOGRAD", re.compile(r"winograd", re.I)),
    ("TF32_TENSOROP", re.compile(r"tf32|tensorop|1688|16816|h1688", re.I)),
    ("FP64", re.compile(r"dgrad|dgemm|dconv|fp64|double|_d_", re.I)),
    ("FP32_SIMT", re.compile(r"fprop|implicit_gemm|sgemm|s1688|fp32|scudnn|simt", re.I)),
    ("CUTLASS", re.compile(r"cutlass", re.I)),
]


def classify(names):
    for label, pat in CLASS_PATTERNS:
        for n in names:
            if pat.search(n):
                return label
    return "OTHER"


def pick_rows():
    gate = [json.loads(l) for l in open(GATE)]
    change = [r for r in gate if r.get("armA_delivered_change")]
    rest = [r for r in gate if not r.get("armA_delivered_change")]
    rest.sort(key=lambda r: (r["H"], r["N"], r["C"]))
    controls = rest[::7][:6]
    return change + controls


def worker(h, n, c, budget, rep):
    import torch
    import torch.nn.functional as F
    from torch.profiler import profile, ProfilerActivity
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)
    if budget == "constrained":
        torch.cuda.set_per_process_memory_fraction(BUDGET_FRACTION)
    flush = torch.zeros(256 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    torch.manual_seed(1234 + h * 100 + n * 10 + c)
    x = torch.randn(n, c, h, h, device="cuda")
    w = torch.randn(c, c, 3, 3, device="cuda")
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        out = F.conv2d(x, w, padding=1)
        torch.cuda.synchronize()
    ev = []
    for e in prof.key_averages():
        if e.device_time_total > 0:
            ev.append((e.key, e.device_time_total))
    ev.sort(key=lambda t: -t[1])
    top = [(nm, round(us, 2)) for nm, us in ev[:5]]
    rec = {"pid": os.getpid(), "nonce": os.urandom(6).hex(), "H": h, "N": n, "C": c,
           "budget": budget, "rep": rep, "date": DATE,
           "flags": {"allow_tf32": True, "benchmark": False, "deterministic": False,
                     "memory_fraction": BUDGET_FRACTION if budget == "constrained" else None},
           "engine_class": classify([nm for nm, _ in top]),
           "top_kernels": top, "ref_absmax": float(out.abs().max().item())}
    print("S72CELL " + json.dumps(rec), flush=True)


def main():
    rows = pick_rows()
    s63 = {}
    if os.path.exists(S63):
        for l in open(S63):
            r = json.loads(l)
            s63[(r["H"], r["N"], r["C"])] = r["refs"]["def"]["engine_class"]
    s66 = {}
    if os.path.exists(S66):
        for l in open(S66):
            r = json.loads(l)
            s66[(r["H"], r["N"], r["C"])] = r["contexts"]["A"]["engine_class"]
    out = open(OUTL, "w")
    summary = {"date": DATE, "prereg": "docstring of lab/s72_fresh_process.py",
               "n_rows": len(rows), "reps": REPS, "budget_fraction": BUDGET_FRACTION,
               "cells": 0, "rows": [], "r1_process_identity": {"rows_agree_all_reps": 0,
               "rows_agree_s63": 0, "rows_disagree_reps": [], "rows_disagree_s63": []},
               "r2_budget": {"rows_budget_invariant": 0, "rows_flip": []}}
    for r in rows:
        H, N, C = r["H"], r["N"], r["C"]
        cells = []
        for rep in range(REPS):
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker",
                                str(H), str(N), str(C), "default", str(rep)],
                               capture_output=True, text=True, timeout=300)
            line = [l for l in p.stdout.splitlines() if l.startswith("S72CELL ")]
            if not line:
                print("WORKER FAIL", H, N, C, rep, p.stderr[-400:], flush=True)
                cells.append({"error": p.stderr[-400:]})
                continue
            rec = json.loads(line[0][len("S72CELL "):])
            out.write(json.dumps(rec) + "\n")
            cells.append(rec)
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker",
                            str(H), str(N), str(C), "constrained", "0"],
                           capture_output=True, text=True, timeout=300)
        line = [l for l in p.stdout.splitlines() if l.startswith("S72CELL ")]
        bcell = json.loads(line[0][len("S72CELL "):]) if line else {"error": p.stderr[-400:]}
        if line:
            out.write(json.dumps(bcell) + "\n")
        out.flush()
        classes = [c.get("engine_class") for c in cells]
        s63c, s66c = s63.get((H, N, C)), s66.get((H, N, C))
        agree_reps = len(set(classes)) == 1 and None not in classes
        agree_s63 = agree_reps and classes[0] == s63c
        agree_s66 = agree_reps and classes[0] == s66c
        if agree_reps:
            summary["r1_process_identity"]["rows_agree_all_reps"] += 1
        else:
            summary["r1_process_identity"]["rows_disagree_reps"].append(
                {"H": H, "N": N, "C": C, "classes": classes, "s63": s63c, "s66": s66c})
        if agree_s63:
            summary["r1_process_identity"]["rows_agree_s63"] += 1
        else:
            summary["r1_process_identity"]["rows_disagree_s63"].append(
                {"H": H, "N": N, "C": C, "classes": classes, "s63": s63c})
        bclass = bcell.get("engine_class")
        budget_ok = bclass == classes[0] and None not in classes
        if budget_ok:
            summary["r2_budget"]["rows_budget_invariant"] += 1
        else:
            summary["r2_budget"]["rows_flip"].append(
                {"H": H, "N": N, "C": C, "default": classes, "constrained": bclass,
                 "constrained_top": bcell.get("top_kernels")})
        row = {"H": H, "N": N, "C": C, "classes_fresh": classes, "s63": s63c, "s66": s66c,
               "agree_reps": agree_reps, "agree_s63": agree_s63, "agree_s66": agree_s66,
               "budget_class": bclass, "budget_matches": budget_ok,
               "absmax": [c.get("ref_absmax") for c in cells] + [bcell.get("ref_absmax")],
               "armA_delivered_change": bool(r.get("armA_delivered_change"))}
        summary["rows"].append(row)
        summary["cells"] += len(cells) + (1 if line else 0)
        print("ROW", json.dumps(row), flush=True)
    summary["r1_process_identity"]["pass"] = (
        summary["r1_process_identity"]["rows_agree_all_reps"] == len(rows)
        and summary["r1_process_identity"]["rows_agree_s63"] == len(rows))
    summary["r2_budget"]["pass"] = summary["r2_budget"]["rows_budget_invariant"] == len(rows)
    json.dump(summary, open(SUML, "w"), indent=1)
    out.close()
    print("SUMMARY", json.dumps({k: summary[k] for k in ("cells", "r1_process_identity", "r2_budget")})[:1500], flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], int(sys.argv[6]))
    else:
        main()
