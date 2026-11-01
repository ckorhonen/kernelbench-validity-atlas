#!/usr/bin/env python
"""s73c_csv.py — emit the s73 decomposition table as CSV from the committed jsonl. Stdlib only."""
import csv, json, os
LAB = os.path.expanduser("~/kb-lab")
if not os.path.isdir(LAB):
    LAB = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(LAB, "results", "s73_cand_ref_decomp.jsonl")
DST = os.path.join(LAB, "results", "s73_cand_ref_decomp.csv")
rows = [json.loads(l) for l in open(SRC)]
cols = ["kind", "level", "problem_id", "name", "op_class", "s71_strict_count", "tie_out_ok",
        "t1e4_c1_verdict_default", "t1e4_c1_fail_default", "t1e4_c1_verdict_anchor", "t1e4_c1_bit_identical",
        "t1e4_c2_verdict_default", "t1e4_c2_fail_default", "t1e4_c2_ref_induced", "t1e4_c2_cand_induced",
        "t1e4_c2_ref_share_of_fails", "t1e4_c2_verdict_anchor", "t1e4_c2_fail_anchor",
        "t1e4_ratio_ref_median", "t1e4_ratio_ref_q90", "t1e4_max_ratio_ref",
        "t1e2_c1_fail_default", "t1e2_c2_fail_default", "t1e2_c2_ref_induced", "t1e2_c2_cand_induced",
        "total_elements", "status"]
with open(DST, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(cols)
    for r in rows:
        d1 = r.get("decomp_c1_compile", {}).get("1e-4", {})
        d2 = r.get("decomp_c2_tf32off", {}).get("1e-4", {})
        b2 = r.get("decomp_c2_tf32off", {}).get("1e-2", {})
        a1 = r.get("decomp_c1_compile", {}).get("1e-2", {})
        w.writerow([r.get("kind"), r.get("level"), r.get("problem_id"), r.get("name"), r.get("op_class"),
                    r.get("s71_strict_count"), r.get("tie_out_ok"),
                    d1.get("verdict_default"), d1.get("fail_default"), d1.get("verdict_anchor"),
                    d1.get("bit_identical_ca"),
                    d2.get("verdict_default"), d2.get("fail_default"), d2.get("ref_induced"),
                    d2.get("cand_induced"), d2.get("ref_share_of_fails"), d2.get("verdict_anchor"),
                    d2.get("fail_anchor"), d2.get("ratio_ref_median"), d2.get("ratio_ref_q90"),
                    d2.get("max_ratio_ref"), a1.get("fail_default"), b2.get("fail_default"),
                    b2.get("ref_induced"), b2.get("cand_induced"), d1.get("total"), r.get("status")])
print("rows", len(rows), "->", DST)
