#!/usr/bin/env python
"""s109c_schema_check.py — hard-fail validator for MAP_ROW_SCHEMA.md over a map row file.
Usage: python s109c_schema_check.py results/s109_rows.json  (default: results/s109_rows.json next to this
script; '-' reads stdin). Writes results/s109c_schema_check.json beside this script and exits 1 if any row fails.
Portability note: the output directory is resolved relative to this file, so the check runs from inside an
unpacked clone or package, not only from the author's home directory."""
import json, os, sys, time

REQ = ["task", "op_class", "procedure", "reference", "deviation", "verdict", "scope"]


def check(row):
    fails = []
    for k in REQ:
        if k not in row:
            fails.append("missing:" + k)
    if fails:
        return fails
    proc = row["procedure"]
    for k in ("id", "cudnn_tf32", "matmul_tf32", "dtype", "arch", "repo_rev", "commit"):
        if k not in proc:
            fails.append("procedure.missing:" + k)
    ref = row["reference"]
    if ref.get("label_basis") != "profiler_top1_dominance":
        fails.append("reference.label_basis")
    if not ref.get("top1_kernel"):
        fails.append("reference.top1_kernel_empty")
    dev = row["deviation"]
    if dev.get("statistic") != "evaluator_exact_rho":
        fails.append("deviation.statistic")
    if dev.get("anchor") != "chunked_fp64":
        fails.append("deviation.anchor")
    ta = row.get("type_a")
    v = row["verdict"]
    if v.get("kind") == "rate":
        if not ta:
            fails.append("type_a.required_for_rate")
        elif not (0 < ta.get("exceed_rate", -1) < 1):
            fails.append("type_a.exceed_rate_inconsistent_with_rate")
    if v.get("kind") == "label":
        if not ta:
            fails.append("type_a.required_for_label")
        else:
            if ta.get("exceed_rate") != 1 and ta.get("exceed_rate") != 0:
                fails.append("label_requires_exceed_rate_0_or_1")
            rho = dev.get("rho_ew", 0.0)
            if 0.85 <= rho <= 1.30 and ta.get("n_draws", 0) < 16:
                fails.append("boundary_row_needs_16_draws")
    if v.get("kind") not in ("label", "rate", "unmeasured_rate", "diagnostic"):
        fails.append("verdict.kind")
    return fails


def _res_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, "results")
    if os.path.isdir(cand):
        return cand
    return os.path.join(os.path.expanduser("~/rstation/kernelbench/lab/results"))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "results", "s109_rows.json")
    rows = json.load(sys.stdin if src == "-" else open(src))
    if isinstance(rows, dict):
        rows = rows.get("rows", [])
    out = []
    for i, r in enumerate(rows):
        f = check(r)
        out.append({"i": i, "task": r.get("task"), "verdict": (r.get("verdict") or {}).get("kind"),
                    "fails": f, "pass": not f})
    summary = dict(date=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   n_rows=len(rows), n_pass=sum(1 for o in out if o["pass"]),
                   n_fail=sum(1 for o in out if not o["pass"]), results=out)
    path = os.path.join(_res_dir(), "s109c_schema_check.json")
    json.dump(summary, open(path, "w"), indent=1)
    print(json.dumps({k: summary[k] for k in ("n_rows", "n_pass", "n_fail")}))
    for o in out:
        if not o["pass"]:
            print("FAIL", o["task"], o["fails"])
    sys.exit(0 if summary["n_fail"] == 0 else 1)


if __name__ == "__main__":
    main()
