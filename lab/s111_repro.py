#!/usr/bin/env python3
"""s111_repro.py — one-command CPU reproduction of the published reference-deviation map numbers.

Declared scope: CPU only, no GPU. It recomputes every headline figure in the s107/s108/s110 map from the
committed artifact rows and compares each to the value published in ATLAS_MAP.md. Exit 0 means every
published number reproduced; any mismatch prints FAIL and exits 1.

Checks (published value in parentheses):
  C1 map population 191 ok rows (s107 85 + s107b 107 - 1 dtype error)            -> 191
  C2 element-wise strict-gate exceedances at 1e-4, procedure B                   -> 83/191
  C3 benchmark-gate (1e-2) exceedances, procedure B                              -> 2/191
  C4 the two shipped-gate tasks are L2/14 and L2/18                              -> names match
  C5 class medians of rho_ew under B: conv 4.5666 / convT 3.3063 / gemm 2.4219    -> within 1e-3 rel
  C6 s110 ensemble strict-gate count                                             -> 82.3125
  C7 s110 Wilson-summed 95% interval                                             -> 79.34-84.58 (+-0.01)
  C8 row-schema validator on the 11 s109/s109b rows                              -> 11/11 pass

Usage:  python lab/s111_repro.py
"""
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
PUBLISHED = {"n_ok": 191, "exceed_strict": 83, "exceed_bench": 2, "ensemble": 82.3125,
             "wilson": (79.339, 84.584), "class_medians": {"conv": 4.5666, "conv_transpose": 3.3063,
                                                           "gemm": 2.4219}}
checks = []


def load_jsonl(name):
    rows = []
    with open(os.path.join(RES, name)) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def check(cid, label, got, want, ok):
    checks.append({"id": cid, "label": label, "got": got, "want": want, "pass": bool(ok)})
    print(f"{cid} {'PASS' if ok else 'FAIL'}  {label}: got={got} want={want}")


def main():
    rows = []
    for name in ("s107_rekey_map.jsonl", "s107b_rekey_rest.jsonl"):
        rows.extend(r for r in load_jsonl(name) if r.get("status") == "ok")
    check("C1", "map rows ok", len(rows), PUBLISHED["n_ok"], len(rows) == PUBLISHED["n_ok"])

    strict = [r for r in rows if "B_strict" in r]
    bench = [r for r in rows if "B_bench" in r]
    n_strict = sum(1 for r in strict if r["B_strict"]["rho_ew"] > 1)
    n_bench = sum(1 for r in bench if r["B_bench"]["rho_ew"] > 1)
    check("C2", "strict-gate exceedances (1e-4, procedure B)", n_strict, PUBLISHED["exceed_strict"],
          n_strict == PUBLISHED["exceed_strict"])
    check("C3", "benchmark-gate exceedances (1e-2, procedure B)", n_bench, PUBLISHED["exceed_bench"],
          n_bench == PUBLISHED["exceed_bench"])

    names = sorted(f"L{r['level']}/{r['problem_id']}" for r in bench if r["B_bench"]["rho_ew"] > 1)
    check("C4", "the two shipped-gate tasks", names, ["L2/14", "L2/18"], names == ["L2/14", "L2/18"])

    # C5 class medians, procedure B, TF32-labelled rows
    med = {}
    for cls in ("conv", "conv_transpose", "gemm"):
        vals = sorted(r["B_strict"]["rho_ew"] for r in strict
                      if r.get("op_class") == cls and r.get("label_B") == "TF32_TENSOROP")
        if vals:
            n = len(vals)
            med[cls] = round(vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2]), 4)
    ok5 = all(abs(med.get(c, 0) - v) / v < 1e-3 for c, v in PUBLISHED["class_medians"].items())
    check("C5", "class medians rho_ew under B", med, PUBLISHED["class_medians"], ok5)

    # C6/C7 s110 ensemble count and Wilson interval from the draw band
    band = [r for r in load_jsonl("s110_band.jsonl") if r.get("status") == "ok"]
    per = {}
    for r in band:
        k = (r["level"], r["problem_id"])
        per.setdefault(k, [0, 0])
        per[k][1] += 1
        if r["strict"]["rho_ew"] > 1:
            per[k][0] += 1
    rates = {k: v[0] / v[1] for k, v in per.items()}
    ensemble = 83 - 14 + sum(rates.values())
    check("C6", "s110 ensemble strict-gate count", round(ensemble, 4), PUBLISHED["ensemble"],
          abs(ensemble - PUBLISHED["ensemble"]) < 1e-6)

    z = 1.959964
    lo = hi = 0.0
    for k, v in per.items():
        p, n = v[0] / v[1], v[1]
        den = 1 + z * z / n
        c = (p + z * z / (2 * n)) / den
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        lo += max(0.0, c - h)
        hi += min(1.0, c + h)
    lo, hi = 83 - 14 + lo, 83 - 14 + hi
    ok7 = abs(lo - PUBLISHED["wilson"][0]) < 0.01 and abs(hi - PUBLISHED["wilson"][1]) < 0.01
    check("C7", "s110 Wilson-summed 95% interval", [round(lo, 3), round(hi, 3)], PUBLISHED["wilson"], ok7)

    # C8 schema validator (delegated to the committed validator)
    proc = subprocess.run([sys.executable, os.path.join(HERE, "s109c_schema_check.py"),
                           os.path.join(RES, "s109_rows.json")],
                          capture_output=True, text=True)
    ok8 = proc.returncode == 0
    check("C8", "row-schema validator exit", proc.returncode, 0, ok8)

    out = {"date": "s111_repro", "n_checks": len(checks),
           "n_pass": sum(1 for c in checks if c["pass"]),
           "verdict": "REPRO_ALL_PUBLISHED" if all(c["pass"] for c in checks) else "REPRO_MISMATCH",
           "checks": checks,
           "inputs": ["results/s107_rekey_map.jsonl", "results/s107b_rekey_rest.jsonl",
                      "results/s110_band.jsonl", "s109c_schema_check.py"]}
    with open(os.path.join(RES, "s111_repro.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n{out['verdict']} {out['n_pass']}/{out['n_checks']} checks pass -> results/s111_repro.json")
    return 0 if out["verdict"] == "REPRO_ALL_PUBLISHED" else 1


if __name__ == "__main__":
    sys.exit(main())
