#!/usr/bin/env python
"""s71c_csv.py — joinable per-task table from the s71 full-population audit. No measurement: reads
results/s71_full_audit.jsonl and writes results/s71_full_audit.csv with one row per task."""
import csv, json, os

RES = os.path.expanduser("~/rstation/kernelbench/lab/results")
rows = [json.loads(l) for l in open(os.path.join(RES, "s71_full_audit.jsonl"))]
cols = ["level", "problem_id", "name", "op_class", "anchor", "status", "output_shape", "ref_max_abs",
        "ew_count_bench_default", "ew_frac_bench_default", "rho_global_bench_default",
        "ew_count_strict_default", "ew_frac_strict_default", "rho_global_strict_default",
        "ew_count_bench_tf32off", "ew_frac_bench_tf32off", "rho_global_bench_tf32off",
        "ew_count_strict_tf32off", "ew_frac_strict_tf32off", "rho_global_strict_tf32off",
        "ab_ew_count_bench", "ab_ew_count_strict", "ab_bit_identical", "t_default_s", "t_fp64_s"]
out = []
for r in rows:
    a = r.get("A_vs_anchor", {})
    b = r.get("B_vs_anchor", {})
    ab = r.get("A_vs_B", {})
    out.append({
        "level": r["level"], "problem_id": r["problem_id"], "name": r["name"], "op_class": r["op_class"],
        "anchor": r.get("anchor"), "status": r.get("status") or "ok",
        "output_shape": json.dumps(r.get("output_shape")), "ref_max_abs": r.get("ref_max_abs"),
        "ew_count_bench_default": a.get("ew_1e-2", {}).get("count"),
        "ew_frac_bench_default": a.get("ew_1e-2", {}).get("frac"),
        "rho_global_bench_default": a.get("ew_1e-2", {}).get("rho_global"),
        "ew_count_strict_default": a.get("ew_1e-4", {}).get("count"),
        "ew_frac_strict_default": a.get("ew_1e-4", {}).get("frac"),
        "rho_global_strict_default": a.get("ew_1e-4", {}).get("rho_global"),
        "ew_count_bench_tf32off": b.get("ew_1e-2", {}).get("count"),
        "ew_frac_bench_tf32off": b.get("ew_1e-2", {}).get("frac"),
        "rho_global_bench_tf32off": b.get("ew_1e-2", {}).get("rho_global"),
        "ew_count_strict_tf32off": b.get("ew_1e-4", {}).get("count"),
        "ew_frac_strict_tf32off": b.get("ew_1e-4", {}).get("frac"),
        "rho_global_strict_tf32off": b.get("ew_1e-4", {}).get("rho_global"),
        "ab_ew_count_bench": ab.get("ew_1e-2", {}).get("count"),
        "ab_ew_count_strict": ab.get("ew_1e-4", {}).get("count"),
        "ab_bit_identical": r.get("A_vs_B_bit_identical"),
        "t_default_s": r.get("t_default_s"), "t_fp64_s": r.get("t_fp64_s"),
    })
with open(os.path.join(RES, "s71_full_audit.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for o in out:
        w.writerow(o)
print("rows", len(out), "cols", len(cols))
