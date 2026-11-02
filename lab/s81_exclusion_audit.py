"""s81_exclusion_audit.py — what are the 10 rows s74/s75b excluded? Record-schema limit, measurement gap,
or instrument defect? CPU only, committed artifacts, no GPU.

PREREG (frozen at commit BEFORE the first run; no threshold, population or read may be changed after results)

Inputs (committed, unmodified):
  lab/results/s73c_merged_56.jsonl           s73/s73b decomp rows (per-arm, per-tolerance dicts)
  lab/results/s71_full_audit.jsonl           s71 audit rows (input_shapes, output_shape, n_params, anchor)
  lab/results/s75_engine_class_s73pop.jsonl  s75 labels (status/kind)
  lab/results/s75b_class_conditioned.json    s75b A3 unparsed list (the 10 excluded rows, as published)

Q1 CLASSIFICATION (per excluded row; rule coded before the run):
  FALSY_ZERO      ratio_ref_median == 0.0 exactly and n_ratio_samples > 0  => a measured zero excluded by
                  a falsy test (`if not r.get("bias")`), not a missing value.
  NO_RATIO_SAMPLE ratio_ref_median is None or n_ratio_samples == 0        => true measurement gap.
  PARSER_SCOPE    geometry missing under s75b's parser but s71 stores shapes that encode the geometry
                  (3D activation for 1D conv/conv_transpose, or a gemm whose second operand is a parameter
                  rather than an input) => parser scope, not data absence.
  SCHEMA_ABSENT   s71 record missing, or shapes absent / uninterpretable => record-schema limit.
  FROZEN VERDICT: EXCLUSIONS_INSTRUMENT_ARTIFACT iff every one of the 10 is FALSY_ZERO or PARSER_SCOPE;
  SCHEMA_LIMIT_PARTIAL otherwise. (No third branch: the two are disjoint by construction.)

Q2 GEOMETRY REPAIR (measurement from committed data): extend s75b's parser with (a) 3D activation for
  conv/conv_transpose as (N, Cin, L) with spatial = L, (b) gemm with one 2D input and a vector/2D output,
  features logM/logK/logNout/logParams. Recompute n_joined and within-op_class LOO R2 on log10(bias).
  Bar unchanged from s74/s75b: LOO R2 >= 0.50 => geometry-predictable; < 0.50 => not.
  Rows with bias == 0 cannot enter a log model: they are excluded from Q2 and counted, not silently dropped.

Q3 SPREAD SENSITIVITY (descriptive): the published within-class spread uses median ratio_ref with
  zero-median rows dropped. Recompute with max_ratio_ref as the magnitude statistic over ALL exceed rows
  (a different quantity, declared as such). Report both spreads and the zero-median count so the
  "26,083x inside one class" figure carries its exclusion count.

Q4 TAIL FACTOR COVERAGE: count rows dropped from T3 (tail = q99/bias) because bias == 0; report n and the
  q99 values of those rows (so the tail side is not silently lost).

Limits declared: bias is a prefix-sampled median (s73 takes the first RATIO_CAP elements in chunk order),
so a 0.0 median is a statement about the prefix, not the tensor; no raw ratios are stored, so the tensor
median cannot be recomputed without a re-run (named as the residual measurement debt if Q3 shows it matters).
"""
import json
import math
import os

LAB = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(LAB, "results")
OUT = os.path.join(RES, "s81_exclusion_audit.json")


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def ols_fit(X, y):
    n, p = len(X), len(X[0])
    A = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
    bvec = [sum(X[i][a] * y[i] for i in range(n)) for a in range(p)]
    M = [row[:] + [bvec[i]] for i, row in enumerate(A)]
    for c in range(p):
        piv = max(range(c, p), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-12:
            return None
        M[c], M[piv] = M[piv], M[c]
        for r in range(c + 1, p):
            f = M[r][c] / M[c][c]
            for k in range(c, p + 1):
                M[r][k] -= f * M[c][k]
    coef = [0.0] * p
    for c in reversed(range(p)):
        coef[c] = (M[c][p] - sum(M[c][k] * coef[k] for k in range(c + 1, p))) / M[c][c]
    return coef


def loo_r2(feats, y):
    n = len(y)
    if n < 6:
        return None, n
    preds = []
    for i in range(n):
        X = [[1.0] + list(feats[j]) for j in range(n) if j != i]
        yy = [y[j] for j in range(n) if j != i]
        coef = ols_fit(X, yy)
        if coef is None:
            return None, n
        preds.append(sum(c * v for c, v in zip(coef, [1.0] + list(feats[i]))))
    ybar = sum(y) / n
    ss_res = sum((y[i] - preds[i]) ** 2 for i in range(n))
    ss_tot = sum((y[i] - ybar) ** 2 for i in range(n))
    return (1 - ss_res / ss_tot) if ss_tot > 0 else None, n


def log10(x):
    return math.log10(x) if x and x > 0 else None


def parse_geometry_extended(rec71, op_class):
    """s75b parser extended for 1D conv (3D activation) and single-input gemm (output-vector N)."""
    shapes = [s for s in (rec71.get("input_shapes") or []) if isinstance(s, list)]
    out = rec71.get("output_shape") if isinstance(rec71.get("output_shape"), list) else None
    if op_class in ("conv", "conv_transpose"):
        cand = [s for s in shapes if len(s) in (3, 4, 5)]
        if not cand and out and len(out) in (3, 4, 5):
            cand = [out]
        if not cand:
            return None
        act = max(cand, key=lambda s: abs(math.prod(s)))
        if len(act) == 3:
            N, Cin, L = act
            sp = L
        elif len(act) == 4:
            N, Cin, H, W = act
            sp = H * W
        else:
            N, Cin, d, H, W = act
            sp = d * H * W
        feats = {"logN": log10(N), "logCin": log10(Cin), "logSpatial": log10(sp),
                 "logParams": log10(rec71.get("n_params") or 0)}
        if out and len(out) == 3:
            feats["logOutSpatial"] = log10(out[2])
        elif out and len(out) == 4:
            feats["logOutSpatial"] = log10(out[2] * out[3])
        elif out and len(out) == 5:
            feats["logOutSpatial"] = log10(out[2] * out[3] * out[4])
        return feats
    if op_class == "gemm":
        t2 = [s for s in shapes if len(s) == 2]
        if len(t2) >= 2:
            a, b = sorted(t2, key=lambda s: -abs(math.prod(s)))
            return {"logM": log10(a[0]), "logK": log10(a[1]), "logN": log10(b[1]),
                    "logParams": log10(rec71.get("n_params") or 0)}
        if len(t2) == 1:
            a = t2[0]
            nout = out[-1] if out else None
            return {"logM": log10(a[0]), "logK": log10(a[1]), "logN": log10(nout or 0),
                    "logParams": log10(rec71.get("n_params") or 0)}
        return None
    return None


def main():
    rows73 = {(r["level"], r["problem_id"]): r for r in load_jsonl(os.path.join(RES, "s73c_merged_56.jsonl"))}
    rows71 = {(r["level"], r["problem_id"]): r for r in load_jsonl(os.path.join(RES, "s71_full_audit.jsonl"))}
    rows75 = load_jsonl(os.path.join(RES, "s75_engine_class_s73pop.jsonl"))
    s75b = json.load(open(os.path.join(RES, "s75b_class_conditioned.json")))

    exceed = [r for r in rows75 if r.get("status") == "ok" and r.get("kind") == "exceed"]
    excluded = [(u["level"], u["problem_id"], u["reason"]) for u in s75b["A3"]["unparsed"]]

    res = {"prereg": "docstring of this file, frozen pre-run", "n_exceed": len(exceed),
           "n_excluded_published": len(excluded), "Q1_rows": [], "Q1_counts": {}, "Q2": {}, "Q3": {},
           "Q4": {}}

    counts = {}
    for level, pid, reason in excluded:
        r73 = rows73.get((level, pid)) or {}
        d = (r73.get("decomp_c2_tf32off") or {}).get("1e-4", {})
        med = d.get("ratio_ref_median")
        nsmp = d.get("n_ratio_samples")
        r71 = rows71.get((level, pid)) or {}
        geo_ext = parse_geometry_extended(r71, r73.get("op_class") or r71.get("op_class"))
        if med == 0.0 and nsmp:
            cause = "FALSY_ZERO"
        elif med is None or not nsmp:
            cause = "NO_RATIO_SAMPLE"
        elif geo_ext is None:
            cause = "SCHEMA_ABSENT"
        else:
            cause = "PARSER_SCOPE"
        counts[cause] = counts.get(cause, 0) + 1
        res["Q1_rows"].append({
            "level": level, "problem_id": pid, "published_reason": reason, "cause": cause,
            "op_class": r73.get("op_class"), "input_shapes": r71.get("input_shapes"),
            "output_shape": r71.get("output_shape"), "n_params": r71.get("n_params"),
            "ratio_ref_median": med, "n_ratio_samples": nsmp,
            "max_ratio_ref": d.get("max_ratio_ref"), "ref_over_budget": d.get("ref_over_budget"),
            "total": d.get("total"), "s71_anchor": r71.get("anchor"),
        })
    res["Q1_counts"] = counts
    res["Q1_verdict"] = ("EXCLUSIONS_INSTRUMENT_ARTIFACT"
                         if all(c in ("FALSY_ZERO", "PARSER_SCOPE") for c in counts)
                         and not any(k in counts for k in ("NO_RATIO_SAMPLE", "SCHEMA_ABSENT"))
                         else "SCHEMA_LIMIT_PARTIAL")

    # Q2: geometry repair on the published (s75b) parser vs extended parser.
    def build(parser):
        feats, ys, unparsed, zero_bias = {}, {}, [], 0
        for r in exceed:
            key = (r["level"], r["problem_id"])
            r73 = rows73.get(key) or {}
            d = (r73.get("decomp_c2_tf32off") or {}).get("1e-4", {})
            med = d.get("ratio_ref_median")
            if med is None:
                unparsed.append({"level": r["level"], "problem_id": r["problem_id"], "reason": "no ratio"})
                continue
            if med == 0.0:
                zero_bias += 1
                continue
            r71 = rows71.get(key)
            if not r71:
                unparsed.append({"level": r["level"], "problem_id": r["problem_id"], "reason": "no s71"})
                continue
            g = parser(r71, r73.get("op_class") or r71.get("op_class"))
            if g is None:
                unparsed.append({"level": r["level"], "problem_id": r["problem_id"],
                                 "reason": "geometry", "op_class": r73.get("op_class")})
                continue
            op = r73.get("op_class")
            need = (["logN", "logCin", "logSpatial", "logParams"]
                    if op in ("conv", "conv_transpose") else ["logM", "logK", "logN", "logParams"])
            if any(g.get(k) is None for k in need):
                unparsed.append({"level": r["level"], "problem_id": r["problem_id"],
                                 "reason": "feature-none", "op_class": op, "missing": [k for k in need if g.get(k) is None]})
                continue
            feats.setdefault(op, {"keys": need, "X": []})["X"].append([g[k] for k in need])
            ys.setdefault(op, []).append(math.log10(med))
        out = {}
        for op in feats:
            r2, n = loo_r2(feats[op]["X"], ys[op])
            out[op] = {"n": n, "loo_r2": r2, "features": feats[op]["keys"]}
        return {"per_op": out, "n_unparsed": len(unparsed), "unparsed": unparsed, "zero_bias_rows": zero_bias}

    # s75b parser reproduced exactly for the comparison leg
    def parse_s75b(rec71, op_class):
        shapes = [s for s in (rec71.get("input_shapes") or []) if isinstance(s, list)]
        out = rec71.get("output_shape") if isinstance(rec71.get("output_shape"), list) else None
        if op_class in ("conv", "conv_transpose"):
            cand = [s for s in shapes if len(s) in (4, 5)]
            if not cand and out and len(out) in (4, 5):
                cand = [out]
            if not cand:
                return None
            act = max(cand, key=lambda s: abs(math.prod(s)))
            if len(act) == 4:
                N, Cin, H, W = act
                sp = H * W
            else:
                N, Cin, d, H, W = act
                sp = d * H * W
            return {"logN": log10(N), "logCin": log10(Cin), "logSpatial": log10(sp),
                    "logParams": log10(rec71.get("n_params") or 0)}
        if op_class == "gemm":
            t2 = [s for s in shapes if len(s) == 2]
            if len(t2) < 2:
                return None
            a, b = sorted(t2, key=lambda s: -abs(math.prod(s)))
            return {"logM": log10(a[0]), "logK": log10(a[1]), "logN": log10(b[1]),
                    "logParams": log10(rec71.get("n_params") or 0)}
        return None

    res["Q2"]["published_parser"] = build(parse_s75b)
    res["Q2"]["extended_parser"] = build(parse_geometry_extended)
    res["Q2"]["verdict"] = ("GEOMETRY_VERDICT_UNCHANGED"
                            if all((v.get("loo_r2") or -9) < 0.50 for v in res["Q2"]["extended_parser"]["per_op"].values())
                            else "GEOMETRY_PREDICTABLE_ON_EXTENDED_POPULATION")

    # Q3: spread sensitivity — median statistic (published) vs max statistic (declared alternative).
    def spread(vals, include_zero=False):
        v = [x for x in vals if x is not None and (x > 0 or (include_zero and x == 0))]
        if include_zero and any(x == 0 for x in v):
            return None  # max/min undefined once a true zero is in the set
        return (max(v) / min(v)) if len(v) >= 2 else None

    by_class = {}
    for r in exceed:
        d = (rows73.get((r["level"], r["problem_id"])) or {}).get("decomp_c2_tf32off", {}).get("1e-4", {})
        cls = r.get("engine_class_primary") or "UNLABELLED"
        by_class.setdefault(cls, {"med": [], "max": []})
        by_class[cls]["med"].append(d.get("ratio_ref_median"))
        by_class[cls]["max"].append(d.get("max_ratio_ref"))
    res["Q3"]["per_class"] = {c: {"n": len(v["med"]), "n_zero_median": sum(1 for x in v["med"] if x == 0.0),
                                 "spread_median_excl_zero": spread(v["med"]),
                                 "spread_max_all_rows": spread(v["max"])}
                              for c, v in by_class.items()}
    dom = "TF32_TENSOROP"
    res["Q3"]["dominant_class"] = dom
    res["Q3"]["published_spread"] = 26083.09071996373
    res["Q3"]["spread_max_all_rows_dominant"] = res["Q3"]["per_class"].get(dom, {}).get("spread_max_all_rows")

    # Q4: tail-factor coverage.
    tail_rows, dropped = [], []
    for r in exceed:
        d = (rows73.get((r["level"], r["problem_id"])) or {}).get("decomp_c2_tf32off", {}).get("1e-4", {})
        med, q99 = d.get("ratio_ref_median"), d.get("ratio_ref_q99")
        if med and q99:
            tail_rows.append({"level": r["level"], "problem_id": r["problem_id"], "tail": q99 / med})
        else:
            dropped.append({"level": r["level"], "problem_id": r["problem_id"], "bias": med, "q99": q99})
    res["Q4"] = {"n_tail_rows": len(tail_rows), "n_dropped": len(dropped), "dropped": dropped}

    with open(OUT, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps({"Q1_counts": res["Q1_counts"], "Q1_verdict": res["Q1_verdict"],
                      "Q2_pub": res["Q2"]["published_parser"]["per_op"],
                      "Q2_ext": res["Q2"]["extended_parser"]["per_op"],
                      "Q2_unparsed_ext": res["Q2"]["extended_parser"]["n_unparsed"],
                      "Q3": res["Q3"]["per_class"], "Q4": {"n": res["Q4"]["n_tail_rows"],
                                                           "dropped": len(res["Q4"]["dropped"])}},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
