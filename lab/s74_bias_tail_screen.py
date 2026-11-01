#!/usr/bin/env python
"""s74_bias_tail_screen.py — is the reference's within-class deviation a geometry-driven bias or a
rare-element tail? CPU screen on committed artifacts, no GPU, no new measurement.

PREREG (frozen before the run; do not edit after).

Motivation: s73 decomposed candidate failures against the default reference and found 99.72% of one arm's
failing elements reference-induced, with per-task reference budget consumption spanning median 0.330 to
max 23.63. That spread is unexplained. Two mechanisms imply opposite research decisions: a geometry-driven
bias means the reference share is predictable from declared shape coordinates (a derived-coordinate law,
auditable without GPU); a rare-element tail means it stays an enumerated per-shape measurement and only the
budget column transfers.

Population: every row of results/s73_cand_ref_decomp.jsonl with status ok and kind exceed (52 rows), joined
by (level, problem_id) to results/s71_full_audit.jsonl for input_shapes and op_class. Rows whose geometry
cannot be parsed are excluded and named.

Quantities per task, from the recorded decomp_c2_tf32off["1e-4"] block (the arm whose failures are
reference-dominated, so the reference's own error is what is being described):
  bias   = ratio_ref_median      (median d_ref/b_e over elements)
  q90, q99 = recorded ratio percentiles
  tail   = ratio_ref_q99 / ratio_ref_median  (tail-to-bias factor)
  exceed = ref_over_budget / total           (element exceedance fraction)

Geometry parse: for op_class conv / conv_transpose, the 4-D input tensor with the largest element count is
(N, C_in, H, W) and the 4-D tensor with the smallest spatial dims is the weight (C_out, C_in, k, k); features
are log10 of N, C_in, C_out, H, W, k. For gemm, features are log10 of M, N, K from the two input shapes.

Predeclared tests and reads (no post-hoc thresholds):
  T1 bias spread: max/min of per-task bias. >=10 => bias varies strongly across tasks; <=2 => near-constant.
  T2 geometry model: within op_class, OLS of log10(bias) on the geometry features, leave-one-out R^2.
     >=0.50 => bias is geometry-predictable on this population; <0.50 => not.
  T3 tail invariance: coefficient of variation of the tail-to-bias factor across tasks.
     <=0.30 => tail is a fixed multiple of bias (shape-invariant error distribution);
     >=0.50 => tail is decoupled from bias.
  T4 tail-bias coupling: Spearman(q99, bias). >=0.80 => tail scales with bias; <0.50 => independent.
Composite verdict (first matching rule wins):
  BIAS_LAW        T2 >= 0.50 and T3 <= 0.30
  TAIL_DOMINANT   T4 < 0.50 or T3 >= 0.50
  MIXED           otherwise
Kill clause K0: fewer than 30 joined rows => screen inconclusive, no verdict claimed.
Declared limits: one seed per task; no engine-class label for this population (s63 covers a different 56
rows), so op_class is the class proxy; conv and conv_transpose pooled only after a per-class check.
"""
import json, math, os

LAB = os.path.expanduser("~/rstation/kernelbench/lab")
S73 = os.path.join(LAB, "results", "s73_cand_ref_decomp.jsonl")
S71 = os.path.join(LAB, "results", "s71_full_audit.jsonl")
OUT = os.path.join(LAB, "results", "s74_bias_tail_screen.json")


def log10(x):
    return math.log10(x) if x and x > 0 else None


def parse_geometry(rec71, op_class):
    shapes = rec71.get("input_shapes") or []
    t4 = [s for s in shapes if isinstance(s, list) and len(s) == 4]
    if op_class in ("conv", "conv_transpose"):
        if len(t4) < 2:
            return None
        act = max(t4, key=lambda s: s[0] * s[1] * s[2] * s[3])
        w = min(t4, key=lambda s: s[2] * s[3])
        N, Cin, H, W = act[0], act[1], act[2], act[3]
        Cout, _, kh, kw = w[0], w[1], w[2], w[3]
        k = max(kh, kw)
        feats = {"logN": log10(N), "logCin": log10(Cin), "logCout": log10(Cout),
                 "logH": log10(H), "logW": log10(W), "logK": log10(k)}
        return feats
    if op_class == "gemm":
        t2 = [s for s in shapes if isinstance(s, list) and len(s) == 2]
        if len(t2) < 2:
            return None
        a, b = t2[0], t2[1]
        return {"logM": log10(a[0]), "logN": log10(b[1]), "logK": log10(a[1])}
    return None


def ols_loo_r2(X, y):
    n = len(y)
    if n < 5:
        return None
    def fit(idx):
        # normal equations with intercept
        import itertools
        k = len(X[0]) + 1
        A = [[0.0] * k for _ in range(k)]
        b = [0.0] * k
        for i in idx:
            row = [1.0] + X[i]
            for p in range(k):
                b[p] += row[p] * y[i]
                for q in range(k):
                    A[p][q] += row[p] * row[q]
        # gaussian elimination with tiny ridge
        for p in range(k):
            A[p][p] += 1e-9
        for c in range(k):
            piv = max(range(c, k), key=lambda r: abs(A[r][c]))
            A[c], A[piv] = A[piv], A[c]
            b[c], b[piv] = b[piv], b[c]
            if abs(A[c][c]) < 1e-12:
                return None
            for r in range(c + 1, k):
                f = A[r][c] / A[c][c]
                for q in range(c, k):
                    A[r][q] -= f * A[c][q]
                b[r] -= f * b[c]
        coef = [0.0] * k
        for p in range(k - 1, -1, -1):
            coef[p] = (b[p] - sum(A[p][q] * coef[q] for q in range(p + 1, k))) / A[p][p]
        return coef
    preds = []
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        coef = fit(idx)
        if coef is None:
            return None
        preds.append(coef[0] + sum(c * v for c, v in zip(coef[1:], X[i])))
    ybar = sum(y) / n
    ss_res = sum((y[i] - preds[i]) ** 2 for i in range(n))
    ss_tot = sum((y[i] - ybar) ** 2 for i in range(n))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else None


def spearman(a, b):
    n = len(a)
    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        for pos, i in enumerate(order):
            r[i] = pos + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((x - mb) ** 2 for x in rb))
    return num / (da * db) if da and db else None


def main():
    rows73 = [json.loads(l) for l in open(S73)]
    rows71 = {r["problem_id"]: r for r in (json.loads(l) for l in open(S71))}
    pop = [r for r in rows73 if r.get("status") == "ok" and r.get("kind") == "exceed"]
    joined, excluded = [], []
    for r in pop:
        r71 = rows71.get(r["problem_id"])
        d = r.get("decomp_c2_tf32off", {}).get("1e-4", {})
        if r71 is None or not d.get("ratio_ref_median"):
            excluded.append({"problem_id": r["problem_id"], "reason": "no geometry or no ratio"})
            continue
        geo = parse_geometry(r71, r.get("op_class") or r71.get("op_class"))
        if geo is None:
            excluded.append({"problem_id": r["problem_id"], "reason": "geometry unparsed"})
            continue
        if any(v is None for v in geo.values()):
            excluded.append({"problem_id": r["problem_id"], "reason": "nonpositive geometry"})
            continue
        joined.append({"problem_id": r["problem_id"], "name": r["name"],
                       "op_class": r.get("op_class") or r71.get("op_class"),
                       "bias": d["ratio_ref_median"], "q90": d.get("ratio_ref_q90"),
                       "q99": d.get("ratio_ref_q99"), "max": d.get("max_ratio_ref"),
                       "exceed_frac": (d["ref_over_budget"] / d["total"]) if d.get("total") else None,
                       "total": d.get("total"), "geo": geo})
    res = {"prereg": "docstring of this file, frozen pre-run", "population": {
        "s73_exceed_rows": len(pop), "joined": len(joined), "excluded": excluded}}
    if len(joined) < 30:
        res["verdict"] = "INCONCLUSIVE_K0"
        json.dump(res, open(OUT, "w"), indent=1)
        print(json.dumps(res, indent=1)[:2000])
        return

    biases = sorted(r["bias"] for r in joined)
    tails = [r["q99"] / r["bias"] for r in joined if r.get("q99") and r["bias"]]
    t1 = biases[-1] / biases[0] if biases[0] else None
    tb_mean = sum(tails) / len(tails) if tails else None
    tb_sd = math.sqrt(sum((t - tb_mean) ** 2 for t in tails) / len(tails)) if tails else None
    t3 = (tb_sd / tb_mean) if tb_mean else None
    q99s = [r["q99"] for r in joined if r.get("q99")]
    bias_match = [r["bias"] for r in joined if r.get("q99")]
    t4 = spearman(q99s, bias_match) if len(q99s) >= 5 else None

    per_class = {}
    for cls in ("conv", "conv_transpose", "gemm"):
        sub = [r for r in joined if r["op_class"] == cls]
        if len(sub) < 8:
            per_class[cls] = {"n": len(sub), "skipped": True}
            continue
        keys = sorted(sub[0]["geo"].keys())
        X = [[r["geo"][k] for k in keys] for r in sub]
        y = [log10(r["bias"]) for r in sub]
        per_class[cls] = {"n": len(sub), "features": keys, "loo_r2": ols_loo_r2(X, y),
                          "bias_min": min(r["bias"] for r in sub), "bias_max": max(r["bias"] for r in sub),
                          "median_bias": sorted(r["bias"] for r in sub)[len(sub) // 2]}
    r2s = [v["loo_r2"] for v in per_class.values() if not v.get("skipped") and v.get("loo_r2") is not None]
    t2 = max(r2s) if r2s else None

    res.update({"T1_bias_spread_max_over_min": t1, "T1_bias_min": biases[0], "T1_bias_max": biases[-1],
                "T2_best_within_class_loo_r2": t2, "T2_per_class": per_class,
                "T3_tail_factor_mean": tb_mean, "T3_tail_factor_cv": t3,
                "T4_spearman_q99_vs_bias": t4, "n": len(joined)})
    if t2 is not None and t2 >= 0.50 and t3 is not None and t3 <= 0.30:
        verdict = "BIAS_LAW"
    elif (t4 is not None and t4 < 0.50) or (t3 is not None and t3 >= 0.50):
        verdict = "TAIL_DOMINANT"
    else:
        verdict = "MIXED"
    res["verdict"] = verdict
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "T2_per_class"}, indent=1)[:2500])
    print(json.dumps(res["T2_per_class"], indent=1)[:1500])


if __name__ == "__main__":
    main()
