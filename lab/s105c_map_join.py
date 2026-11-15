#!/usr/bin/env python
"""s105c_map_join.py — join the base-rate rows to the engine-class column and repair one classifier defect.

DECLARED POST-HOC (after s105b's rows existed; no GPU, no re-measurement; relabelling from the recorded kernel names
only). Prereg discipline: the join rules below are fixed before this script's first run.

Join: s105_map_base_rate.json rows (200 tasks, rho/frac at both gates from s71/s76) x s105b_map_engine_class.jsonl
(200 tasks, observed top-5 kernels). Key (level, problem_id, name). Every row must join; a miss is reported.

Classifier defect found and repaired here (recorded, not threshold-repaired): s63's CLASS_PATTERNS maps any name
containing "dgrad" to FP64, but cuDNN names the conv_TRANSPOSE forward kernel dgrad*, and the reference here is fp32 —
so 10 conv_transpose rows came out labelled FP64. Repair rule (name-based, no GPU): a matched name is FP64 only if it
matches dgemm|fp64|double and does NOT match dgrad; a dgrad-named kernel is labelled TF32_TENSOROP when it carries
tf32|tensorop|1688, else FP32_SIMT. All other labels are kept verbatim from s105b. The original labels are retained in
the output as engine_class_s63 so the defect is visible, not erased.

Tie-out (K2): for tasks that s75 profiled (results/s75_engine_class.jsonl), compare this script's relabelled class to
s75's recorded label; disagreements are printed and counted.

Reported: the class-conditioned map — per (op_class, engine_class): n rows, n above rho 0.1 at 1e-4, n exceeding at
1e-4, median/max rho_strict, median frac_strict; the 55 strict-exceeding rows with their class and rho; and the
bench-gate column (max rho_bench, n exceed).

Outputs: results/s105c_map_join.json + results/s105c_map_join.csv.
"""
import json, os, re, csv, statistics

RES = os.path.expanduser("~/rstation/kernelbench/lab/results")
S105 = os.path.join(RES, "s105_map_base_rate.json")
S105B = os.path.join(RES, "s105b_map_engine_class.jsonl")
S75 = os.path.join(RES, "s75_engine_class.jsonl")
OUT = os.path.join(RES, "s105c_map_join.json")
CSV = os.path.join(RES, "s105c_map_join.csv")

FP64_RE = re.compile(r"dgemm|fp64|double", re.I)
DGRAD_RE = re.compile(r"dgrad", re.I)
TF32_RE = re.compile(r"tf32|tensorop|1688|16816|h1688", re.I)


def relabel(label, matched):
    if label != "FP64" or not matched:
        return label
    if FP64_RE.search(matched) and not DGRAD_RE.search(matched):
        return "FP64"
    if DGRAD_RE.search(matched):
        return "TF32_TENSOROP" if TF32_RE.search(matched) else "FP32_SIMT"
    return label


def main():
    base = json.load(open(S105))
    rows = base["rows"]
    cls = {}
    for l in open(S105B):
        r = json.loads(l)
        cls[(r["level"], r["problem_id"], r["name"])] = r
    out, misses = [], []
    for r in rows:
        k = (r["level"], r["problem_id"], r["name"])
        c = cls.get(k)
        if not c:
            misses.append(r["name"])
            continue
        out.append({
            "level": r["level"], "problem_id": r["problem_id"], "name": r["name"],
            "op_class": r["op_class"], "rho_strict": r["rho_strict"], "frac_strict": r["frac_strict"],
            "rho_bench": r["rho_bench"], "frac_bench": r["frac_bench"],
            "ref_max_abs": r["ref_max_abs"], "u_strict": r["u_strict"], "u_bench": r["u_bench"],
            "exceed_strict": r["exceed_strict"], "exceed_bench": r["exceed_bench"],
            "engine_class_s63": c.get("engine_class"),
            "matched_kernel": c.get("matched_kernel"),
            "engine_class": relabel(c.get("engine_class"), c.get("matched_kernel")),
            "n_cuda_kernels": c.get("n_cuda_kernels"), "status": c.get("status"),
        })
    n = len(out)
    n_relabelled = sum(1 for r in out if r["engine_class"] != r["engine_class_s63"])

    tie = {"compared": 0, "agree": 0, "disagree": []}
    if os.path.exists(S75):
        s75 = {}
        for l in open(S75):
            r = json.loads(l)
            key = r.get("name") or r.get("task")
            if key:
                s75[key] = r.get("engine_class") or r.get("class")
        for r in out:
            if r["name"] in s75 and s75[r["name"]]:
                tie["compared"] += 1
                if s75[r["name"]] == r["engine_class"]:
                    tie["agree"] += 1
                else:
                    tie["disagree"].append({"name": r["name"], "s75": s75[r["name"]],
                                            "s105c": r["engine_class"], "kernel": r["matched_kernel"]})

    grid = {}
    for r in out:
        key = "%s|%s" % (r["op_class"], r["engine_class"])
        g = grid.setdefault(key, {"n": 0, "n_above_0.1": 0, "n_exceed_strict": 0,
                                  "rho_strict": [], "frac_strict": [], "n_exceed_bench": 0})
        g["n"] += 1
        if r["rho_strict"] is not None:
            g["rho_strict"].append(r["rho_strict"])
            if r["rho_strict"] > 0.1:
                g["n_above_0.1"] += 1
        if r["frac_strict"] is not None:
            g["frac_strict"].append(r["frac_strict"])
        g["n_exceed_strict"] += 1 if r["exceed_strict"] else 0
        g["n_exceed_bench"] += 1 if r["exceed_bench"] else 0
    for k, g in grid.items():
        g["median_rho_strict"] = statistics.median(g["rho_strict"]) if g["rho_strict"] else None
        g["max_rho_strict"] = max(g["rho_strict"]) if g["rho_strict"] else None
        g["median_frac_strict"] = statistics.median(g["frac_strict"]) if g["frac_strict"] else None
        del g["rho_strict"], g["frac_strict"]

    exceed_rows = [{"name": r["name"], "op_class": r["op_class"], "engine_class": r["engine_class"],
                    "rho_strict": r["rho_strict"], "frac_strict": r["frac_strict"],
                    "u_strict": r["u_strict"], "matched_kernel": r["matched_kernel"]}
                   for r in out if r["exceed_strict"]]
    summary = {
        "n_joined": n, "misses": misses, "n_relabelled_fp64_dgrad": n_relabelled,
        "tie_out_s75": {"compared": tie["compared"], "agree": tie["agree"],
                        "n_disagree": len(tie["disagree"]), "disagreements": tie["disagree"][:10]},
        "class_conditioned_map": grid,
        "n_exceed_strict": len(exceed_rows),
        "n_exceed_bench": sum(1 for r in out if r["exceed_bench"]),
        "max_rho_bench": max([r["rho_bench"] for r in out if r["rho_bench"] is not None]),
        "exceed_rows": exceed_rows,
        "date": "2026-11-15",
    }
    json.dump({"summary": summary, "rows": out}, open(OUT, "w"), indent=1)
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        for r in out:
            w.writerow(r)
    print(json.dumps(summary, indent=1)[:6000])


if __name__ == "__main__":
    main()
