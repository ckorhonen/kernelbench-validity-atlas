#!/usr/bin/env python
"""s105_map_base_rate.py — the benchmark-wide evaluator map, base-rate leg, from committed records.

PREREG (frozen at commit before any output row; do not edit after).

Question: does the reference-deviation map extend beyond conv at benchmark scale? The 11-13 reflection committed a
stratified 24-task non-conv pilot with a predeclared split (>=25% above rho 0.1 => WIDE; >=90% below => NARROW).
This script answers that question from the committed s71_full_audit records (200 tasks, both gates, anchor fp64)
plus s76's 8 anchor-coverage rows, BEFORE spending any GPU time, because the population was already measured and the
pilot would have been a re-measurement (the same class of error as the eleven "deferred" flip-table citations).

Population (rule): every row in results/s71_full_audit.jsonl with anchor == "admissible" (192 rows) plus every row in
results/s76_anchor_coverage.jsonl (8 rows, coverage closure). op_class is the field the benchmark's own task list
carries; no re-labelling by me.

Quantities per row (no thresholds invented): rho_strict = A_vs_anchor.ew_1e-4.rho_global, frac_strict =
A_vs_anchor.ew_1e-4.frac, rho_bench = A_vs_anchor.ew_1e-2.rho_global, frac_bench = ew_1e-2.frac, ref_max_abs,
exceed flags at both gates, input_bytes, output_bytes, op_class, and the derived coordinate
u = 2.2e-4 * ref_max_abs / atol for each gate (atol = rtol = 1e-2 and 1e-4 respectively), which is s91's constant.

Predeclared branch rules (computed by this script, printed verbatim):
  M1_BASE_RATE_WIDE   if the fraction of NON-CONV rows with rho_strict > 0.1 is >= 0.25
  M1_BASE_RATE_NARROW if that fraction is <= 0.10  (>=90% below 0.1)
  M1_BASE_RATE_PARTIAL otherwise
  M2_BENCH_GATE       reported separately: count of rows with exceed_bench_default (the shipped gate), and the
                      max rho_bench. If 0 rows exceed and max rho_bench < 0.25 the shipped gate is declared
                      non-binding on this population, conv included.
  M3_CLASS_SPREAD     per-class rho_strict distributions (median, p90, max) and the count above 0.1, printed with the
                      sampling weights (class counts) so the split cannot be read as a population fraction.

Kill clauses (K0) a row whose rho is missing or whose anchor is not admissible is reported as excluded, not dropped.
(K1) if the s76 rows disagree with s71's tie-out where both exist, the disagreement is named.
No GPU, no re-measurement: this is a join over committed artifacts (P6 bootstrap-from-artifacts).

Outputs: results/s105_map_base_rate.json (per-row table + summary), results/s105_map_base_rate.csv.
"""
import json, os, csv, statistics

RES = os.path.expanduser("~/rstation/kernelbench/lab/results")
S71 = os.path.join(RES, "s71_full_audit.jsonl")
S76 = os.path.join(RES, "s76_anchor_coverage.jsonl")
OUT = os.path.join(RES, "s105_map_base_rate.json")
CSV = os.path.join(RES, "s105_map_base_rate.csv")

CONV_CLASSES = {"conv", "conv_transpose"}


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_rho(row, gate):
    a = row.get("A_vs_anchor") or {}
    k = "ew_%s" % gate
    d = a.get(k) or {}
    return d.get("rho_global"), d.get("frac")


def main():
    rows71 = load(S71)
    rows76 = load(S76)
    table = []
    excluded = []
    for r in rows71:
        if r.get("anchor") != "admissible":
            excluded.append({"level": r.get("level"), "problem_id": r.get("problem_id"),
                             "name": r.get("name"), "reason": r.get("anchor")})
            continue
        rs, fs = get_rho(r, "1e-4")
        rb, fb = get_rho(r, "1e-2")
        table.append({
            "level": r["level"], "problem_id": r["problem_id"], "name": r["name"],
            "op_class": r.get("op_class"), "source": "s71",
            "rho_strict": rs, "frac_strict": fs, "rho_bench": rb, "frac_bench": fb,
            "ref_max_abs": r.get("ref_max_abs"),
            "exceed_strict": bool(r.get("exceed_strict_default")),
            "exceed_bench": bool(r.get("exceed_bench_default")),
            "u_strict": (2.2e-4 * r["ref_max_abs"] / 1e-4) if r.get("ref_max_abs") else None,
            "u_bench": (2.2e-4 * r["ref_max_abs"] / 1e-2) if r.get("ref_max_abs") else None,
            "input_bytes": r.get("input_bytes"), "output_bytes": r.get("output_bytes"),
            "output_shape": r.get("output_shape"),
        })
    for r in rows76:
        rs, fs = get_rho(r, "1e-4")
        rb, fb = get_rho(r, "1e-2")
        table.append({
            "level": r["level"], "problem_id": r["problem_id"], "name": r.get("name"),
            "op_class": r.get("op_class"), "source": "s76",
            "rho_strict": rs, "frac_strict": fs, "rho_bench": rb, "frac_bench": fb,
            "ref_max_abs": r.get("ref_max_abs"),
            "exceed_strict": bool(r.get("exceed_strict_default")),
            "exceed_bench": bool(r.get("exceed_bench_default")),
            "u_strict": (2.2e-4 * r["ref_max_abs"] / 1e-4) if r.get("ref_max_abs") else None,
            "u_bench": (2.2e-4 * r["ref_max_abs"] / 1e-2) if r.get("ref_max_abs") else None,
            "input_bytes": r.get("input_bytes"), "output_bytes": r.get("output_bytes"),
            "output_shape": r.get("output_shape"),
        })

    n = len(table)
    nonconv = [t for t in table if t["op_class"] not in CONV_CLASSES]
    conv = [t for t in table if t["op_class"] in CONV_CLASSES]
    nc_above = [t for t in nonconv if t["rho_strict"] is not None and t["rho_strict"] > 0.1]
    frac_above = len(nc_above) / len(nonconv) if nonconv else None

    if frac_above is not None and frac_above >= 0.25:
        m1 = "M1_BASE_RATE_WIDE"
    elif frac_above is not None and frac_above <= 0.10:
        m1 = "M1_BASE_RATE_NARROW"
    else:
        m1 = "M1_BASE_RATE_PARTIAL"

    bench_exceed = [t for t in table if t["exceed_bench"]]
    bench_rhos = [t["rho_bench"] for t in table if t["rho_bench"] is not None]
    strict_exceed = [t for t in table if t["exceed_strict"]]
    m2 = "M2_BENCH_GATE_NONBINDING" if (len(bench_exceed) == 0 and max(bench_rhos) < 0.25) else "M2_BENCH_GATE_BINDING"

    by_class = {}
    for t in table:
        c = t["op_class"] or "unlabelled"
        by_class.setdefault(c, []).append(t)
    class_stats = {}
    for c, ts in by_class.items():
        rhos = sorted([t["rho_strict"] for t in ts if t["rho_strict"] is not None])
        if not rhos:
            continue
        p90 = rhos[min(len(rhos) - 1, int(round(0.9 * (len(rhos) - 1))))]
        class_stats[c] = {
            "n": len(ts), "n_above_0.1": sum(1 for x in rhos if x > 0.1),
            "median": statistics.median(rhos), "p90": p90, "max": rhos[-1],
            "n_exceed_strict": sum(1 for t in ts if t["exceed_strict"]),
            "n_exceed_bench": sum(1 for t in ts if t["exceed_bench"]),
            "frac_strict_nonzero": sum(1 for t in ts if (t["frac_strict"] or 0) > 0),
        }

    summary = {
        "population_rule": "s71 admissible rows + s76 coverage rows; op_class from the benchmark task list",
        "n_rows": n, "n_conv": len(conv), "n_nonconv": len(nonconv), "excluded_rows": excluded,
        "branch_M1": m1, "nonconv_fraction_above_0.1": frac_above,
        "nonconv_n_above_0.1": len(nc_above),
        "nonconv_above_rows": [{"name": t["name"], "op_class": t["op_class"],
                                "rho_strict": t["rho_strict"], "frac_strict": t["frac_strict"]}
                               for t in nc_above],
        "branch_M2": m2, "n_exceed_bench": len(bench_exceed),
        "max_rho_bench": max(bench_rhos) if bench_rhos else None,
        "n_exceed_strict": len(strict_exceed),
        "strict_exceed_by_class": {},
        "class_stats": class_stats,
        "max_rho_strict": max([t["rho_strict"] for t in table if t["rho_strict"] is not None]),
        "frac_strict_max": max([t["frac_strict"] for t in table if t["frac_strict"] is not None]),
        "u_strict_range": [min([t["u_strict"] for t in table if t["u_strict"] is not None]),
                           max([t["u_strict"] for t in table if t["u_strict"] is not None])],
        "date": "2026-11-15",
    }
    for t in strict_exceed:
        c = t["op_class"] or "unlabelled"
        summary["strict_exceed_by_class"][c] = summary["strict_exceed_by_class"].get(c, 0) + 1

    os.makedirs(RES, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"summary": summary, "rows": table}, f, indent=1)
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        for t in table:
            w.writerow(t)
    print(json.dumps(summary, indent=1))
    print("rows ->", OUT, "csv ->", CSV)


if __name__ == "__main__":
    main()
