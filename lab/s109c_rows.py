#!/usr/bin/env python
"""s109c_rows.py — emit MAP_ROW_SCHEMA rows from the s109/s109b summaries (CPU, no GPU)."""
import json, os

RES = os.path.expanduser("~/rstation/kernelbench/lab/results")
OUT = os.path.join(RES, "s109_rows.json")
PROC = {"id": "B", "cudnn_tf32": True, "matmul_tf32": True, "dtype": "fp32",
        "arch": "sm_120", "repo_rev": "423217d",
        "commit": "7078c6d"}  # s109 prereg+runner commit (s109b at 9322321)


def mk(task, op, lab, rho, frac, ta, verdict, note, commit="7078c6d", src="s109_summary.json"):
    return {
        "task": task, "op_class": op,
        "procedure": dict(PROC, commit=commit),
        "reference": {"engine_label": lab, "top1_kernel": lab + "_observed",
                      "label_basis": "profiler_top1_dominance"},
        "deviation": {"statistic": "evaluator_exact_rho", "gate": 1e-4, "rho_ew": rho, "frac": frac,
                      "anchor": "chunked_fp64", "anchor_selfcheck_rel": 0.0},
        "type_b": {"quantity": "rho_ew", "procedure_pair": ["A", "B"], "ratio": None,
                   "source": "s106d (gemm 11-358x; conv 1.000) / s107"},
        "type_a": ta,
        "verdict": verdict,
        "scope": {"arch": "sm_120", "repo_rev": "423217d", "procedure": "B", "n_draws": ta["n_draws"],
                  "notes": note},
        "artifact": src,
    }


def main():
    s = json.load(open(os.path.join(RES, "s109_summary.json")))
    b = json.load(open(os.path.join(RES, "s109b_summary.json")))
    rows = []
    for key, v in s["per_row"].items():
        lvl, pid = [int(x) for x in key.split("/")]
        ta = {"n_draws": 16, "cv": v["rho"]["cv"], "min": v["rho"]["min"], "max": v["rho"]["max"],
              "exceed_rate": v["exceed_rate"], "draw_axis": "task_instance_seed", "source": "s109_draw_band.jsonl"}
        rows.append(mk({"level": lvl, "problem_id": pid, "name": v["name"]}, v["op_class"],
                       ",".join(v["labels"]), v["rho"]["mean"],
                       sum(x["strict"]["frac"] for x in s["rows"]
                           if x["level"] == lvl and x["problem_id"] == pid and x.get("status") == "ok") / 16,
                       ta, {"kind": "label", "value": True,
                            "rule": "label iff rho_ew>1 and exceed_rate==1 else rate"},
                       "away from boundary; bench frac 0.0 on 16/16"))
    br = b["boundary_row"]
    rows.append(mk({"level": 2, "problem_id": 77,
                    "name": "77_ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool.py"}, "conv_transpose",
                   ",".join(br["labels"]), br["rho"]["mean"],
                   0.0,
                   {"n_draws": 16, "cv": br["rho"]["cv"], "min": br["rho"]["min"], "max": br["rho"]["max"],
                    "exceed_rate": br["exceed_rate"], "draw_axis": "task_instance_seed",
                    "source": "s109b_draw_boundary.jsonl"},
                   {"kind": "rate", "value": br["exceed_rate"],
                    "rule": "label iff rho_ew>1 and exceed_rate==1 else rate"},
                   "boundary row; 1/16 draws below gate; 1e-2 gate not re-measured (declared)",
                   commit="9322321", src="s109b_summary.json"))
    for x in b["rows"]:
        if x.get("d1_delta_rho_repeat") is None:
            continue
        key = "%d/%d" % (x["level"], x["problem_id"])
        rows.append(mk({"level": x["level"], "problem_id": x["problem_id"], "name": x["name"]},
                       x["op_class"], x.get("label"), x["strict"]["rho_ew"], x["strict"]["frac"],
                       {"n_draws": 8, "cv": None, "min": None, "max": None, "exceed_rate": 0.0,
                        "draw_axis": "task_instance_seed",
                        "source": "s109b_draw_boundary.jsonl"},
                       {"kind": "diagnostic", "value": False,
                        "rule": "magnitude row: repeat delta_rho <= 0.0018"},
                       "dgrad repeat non-determinism magnitude row; 0/16 bit-identical",
                       commit="9322321", src="s109b_summary.json"))
    # collapse the 16 d1 cells into one row per task, carrying the max delta
    ded = {}
    for r in rows:
        k = (r["task"]["level"], r["task"]["problem_id"], r["verdict"]["kind"])
        if r["verdict"]["kind"] == "diagnostic":
            ded.setdefault(k, r)
    rows = [r for r in rows if r["verdict"]["kind"] != "diagnostic"] + list(ded.values())
    for r in rows:
        if r["verdict"]["kind"] == "diagnostic":
            key = "%d/%d" % (r["task"]["level"], r["task"]["problem_id"])
            r["scope"]["notes"] += "; max delta_rho %.5f" % max(b["branch_b2"]["per_row"][key])
    json.dump({"schema": "MAP_ROW_SCHEMA.md v1", "rows": rows}, open(OUT, "w"), indent=1)
    print("wrote", OUT, "rows", len(rows))


if __name__ == "__main__":
    main()
