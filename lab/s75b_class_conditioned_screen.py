#!/usr/bin/env python
"""s75b_class_conditioned_screen.py — does engine class explain the reference-deviation spread, and does
the geometry exclusion survive a repaired join? CPU analysis on committed artifacts plus s75's labels.

PREREG (frozen before the run; do not edit after).

Inputs: results/s75_engine_class_s73pop.jsonl (s75's profiler labels), results/s73c_merged_56.jsonl (the
54 exceed + 8 control rows whose bias ratios s74 analysed), results/s71_full_audit.jsonl (geometry,
n_params, A_vs_B_bit_identical).

Quantities per task, identical to s74 so the numbers stay comparable: bias = ratio_ref_median from
decomp_c2_tf32off["1e-4"], q99 = ratio_ref_q99, exceed = ref_over_budget/total.

Tests, predeclared:
  A1 class distribution of the exceed rows under s75's primary and refined labels, by op_class.
  A2 class-conditioned spread: per-class bias min/max/median and within-class max/min ratio. If >=2
     classes carry n>=5 each, a rank-based separation test (Kruskal-Wallis H with a permutation null,
     20000 shuffles, seed 20261101). Report within-class spread whether or not the test is admissible.
  A3 GEOMETRY BACKFILL — the exclusion s74 declared ("geometry unparsed", 10 rows) is retested with a
     parser that reads the shapes s71 actually stores: for conv/conv_transpose the maximum-volume 4-D or
     5-D input tensor, falling back to the recorded output_shape; for gemm two 2-D operands. Features as
     in s74 (log10 N, Cin, spatial, params) plus a descriptive second model using output spatial extent.
     LOO R2 within op_class on the repaired population. Rows still unparsed are named.
  A4 tree-identity descriptors: split the exceed rows by a_vs_b_bit_identical (whether disabling tf32
     changed the reference output, i.e. whether the accumulation tree changed) and compare bias
     distributions (median, max/min, rank test if both groups n>=5). Count distinct top-1 kernel names
     and, where a name repeats, compare within-name vs between-name bias spread.

Predeclared reads (first matching rule wins; no post-hoc thresholds):
  CLASS_NOT_EXPLANATORY  >=80% of exceed rows carry one primary class => class membership cannot
                         partition the spread; magnitude is a within-class property.
  CLASS_PARTIAL          classes split but within-class spread remains >=10x in every class.
  CLASS_EXPLANATORY      between-class separation p<0.05 AND within-class spread <10x in every class.
  GEOMETRY_STILL_DEAD    repaired-population within-class LOO R2 < 0.50 => s74's TAIL_DOMINANT verdict
                         holds on the population the exclusion removed; the exclusion hid no law.
  GEOMETRY_REVIVED       LOO R2 >= 0.50 on the repaired population => the exclusion was masking a law.
  TREE_FACTOR            a_vs_b_bit_identical splits bias with p<0.05 => tree change is a magnitude
                         factor on this population (hypothesis-grade, one seed).
Kill clauses:
  K3 fewer than 45 exceed rows join s71 geometry => geometry verdict reported as partial, no verdict.
  K4 s75 label coverage < 80% => class verdict reported on the labelled subset only.

Declared limits: one seed per task; s75 labels are one profiler pass per task; op_class is not an engine
label and is used only as a stratification; bias is a median over a prefix-sampled element set (s73's
RATIO_CAP), not over the whole tensor.
"""
import json
import math
import os
import random

LAB = os.path.expanduser("~/rstation/kernelbench/lab")
S75 = os.path.join(LAB, "results", "s75_engine_class_s73pop.jsonl")
S73C = os.path.join(LAB, "results", "s73c_merged_56.jsonl")
S71 = os.path.join(LAB, "results", "s71_full_audit.jsonl")
OUT = os.path.join(LAB, "results", "s75b_class_conditioned.json")

SEED = 20261101


def log10(x):
    return math.log10(x) if x and x > 0 else None


def parse_geometry(rec71, op_class):
    """Repaired parser: uses the shapes s71 stores, widest-tensor first, output-shape fallback."""
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
            d = None
        else:
            N, Cin, d, H, W = act
            sp = d * H * W
        feats = {"logN": log10(N), "logCin": log10(Cin), "logSpatial": log10(sp),
                 "logParams": log10(rec71.get("n_params") or 0)}
        if out and len(out) == 4:
            N2, Cout, Ho, Wo = out
            feats["logOutSpatial"] = log10(Ho * Wo)
        elif out and len(out) == 5:
            N2, Cout, Do, Ho, Wo = out
            feats["logOutSpatial"] = log10(Do * Ho * Wo)
        return feats
    if op_class == "gemm":
        t2 = [s for s in shapes if len(s) == 2]
        if len(t2) < 2:
            return None
        a, b = sorted(t2, key=lambda s: -abs(math.prod(s)))
        return {"logM": log10(a[0]), "logK": log10(a[1]), "logN": log10(b[1]),
                "logParams": log10(rec71.get("n_params") or 0)}
    return None


def ols_fit(X, y):
    n, p = len(X), len(X[0])
    # normal equations with intercept column already in X
    A = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
    bvec = [sum(X[i][a] * y[i] for i in range(n)) for a in range(p)]
    # gaussian elimination with partial pivoting
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
        X = [[1.0] + [feats[j][k] for k in range(len(feats[j]))] for j in range(n) if j != i]
        yy = [y[j] for j in range(n) if j != i]
        coef = ols_fit(X, yy)
        if coef is None:
            return None, n
        preds.append(sum(c * v for c, v in zip(coef, [1.0] + feats[i])))
    ybar = sum(y) / n
    ss_res = sum((y[i] - preds[i]) ** 2 for i in range(n))
    ss_tot = sum((y[i] - ybar) ** 2 for i in range(n))
    return (1 - ss_res / ss_tot) if ss_tot > 0 else None, n


def rankdata(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    ranks = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def kw_h(groups):
    vals, labels = [], []
    for gi, g in enumerate(groups):
        vals.extend(g)
        labels.extend([gi] * len(g))
    n = len(vals)
    ranks = rankdata(vals)
    sums = [0.0] * len(groups)
    counts = [0] * len(groups)
    for r, l in zip(ranks, labels):
        sums[l] += r
        counts[l] += 1
    H = 12.0 / (n * (n + 1)) * sum(s * s / c for s, c in zip(sums, counts)) - 3 * (n + 1)
    rng = random.Random(SEED)
    hits = 0
    trials = 20000
    for _ in range(trials):
        rng.shuffle(labels)
        sums = [0.0] * len(groups)
        counts = [0] * len(groups)
        for r, l in zip(ranks, labels):
            sums[l] += r
            counts[l] += 1
        Hp = 12.0 / (n * (n + 1)) * sum(s * s / c for s, c in zip(sums, counts)) - 3 * (n + 1)
        if Hp >= H:
            hits += 1
    return H, (hits + 1) / (trials + 1)


def spread(v):
    v = [x for x in v if x and x > 0]
    return (max(v) / min(v)) if len(v) >= 2 else None


def main():
    rows75 = [json.loads(l) for l in open(S75)]
    rows73 = {(r["level"], r["problem_id"]): r for r in (json.loads(l) for l in open(S73C))}
    rows71 = {(r["level"], r["problem_id"]): r for r in (json.loads(l) for l in open(S71))}

    labelled = [r for r in rows75 if r.get("status") == "ok"]
    exceed = [r for r in labelled if r.get("kind") == "exceed"]
    controls = [r for r in labelled if r.get("kind") == "control"]

    def bias_of(r):
        d = (rows73.get((r["level"], r["problem_id"]), {}).get("decomp_c2_tf32off") or {}).get("1e-4", {})
        return d.get("ratio_ref_median")

    for r in labelled:
        r["bias"] = bias_of(r)
        r["q99"] = ((rows73.get((r["level"], r["problem_id"]), {}).get("decomp_c2_tf32off") or {})
                    .get("1e-4", {}).get("ratio_ref_q99"))

    res = {
        "prereg": "docstring of this file, frozen pre-run",
        "n_labelled": len(labelled), "n_exceed": len(exceed), "n_control": len(controls),
        "label_coverage": len(labelled) / len(rows75) if rows75 else None,
        "A1_classes": {}, "A2": {}, "A3": {}, "A4": {}, "A5": {},
    }
    for field in ("engine_class_primary", "engine_class_refined"):
        counts = {}
        for r in exceed:
            counts[r.get(field) or "UNLABELLED"] = counts.get(r.get(field) or "UNLABELLED", 0) + 1
        res["A1_classes"][field] = counts
        by_op = {}
        for r in exceed:
            k = (r.get("op_class"), r.get(field))
            by_op[k] = by_op.get(k, 0) + 1
        res["A1_classes"][field + "_by_op"] = {"%s|%s" % k: v for k, v in sorted(by_op.items())}

    # A2 class-conditioned spread
    for field in ("engine_class_primary", "engine_class_refined"):
        groups = {}
        for r in exceed:
            if r.get("bias"):
                groups.setdefault(r.get(field) or "UNLABELLED", []).append(r["bias"])
        stats = {}
        for k, v in groups.items():
            stats[k] = {"n": len(v), "median": sorted(v)[len(v) // 2], "min": min(v), "max": max(v),
                        "spread": spread(v)}
        big = [v for v in groups.values() if len(v) >= 5]
        test = None
        if len(big) >= 2:
            H, p = kw_h(big)
            test = {"H": H, "p_perm": p, "k_groups": len(big)}
        res["A2"][field] = {"per_class": stats, "rank_test": test,
                            "pooled_spread": spread([r["bias"] for r in exceed if r.get("bias")])}
        dom = max(groups.items(), key=lambda kv: len(kv[1])) if groups else None
        res["A2"][field]["dominant_class"] = dom[0] if dom else None
        res["A2"][field]["dominant_frac"] = (len(dom[1]) / len(exceed)) if dom else None

    # A3 geometry backfill
    per_class_feats, per_class_y, unparsed = {}, {}, []
    for r in exceed:
        if not r.get("bias"):
            unparsed.append({"level": r["level"], "problem_id": r["problem_id"], "reason": "no bias"})
            continue
        r71 = rows71.get((r["level"], r["problem_id"]), {})
        f = parse_geometry(r71, r.get("op_class"))
        if f is None or any(v is None for v in f.values()):
            unparsed.append({"level": r["level"], "problem_id": r["problem_id"], "reason": "geometry"})
            continue
        per_class_feats.setdefault(r.get("op_class"), []).append(f)
        per_class_y.setdefault(r.get("op_class"), []).append(math.log10(r["bias"]))
    geo = {}
    for cls in per_class_feats:
        keys = sorted(per_class_feats[cls][0].keys())
        base = ["logN", "logCin", "logSpatial", "logParams"]
        base = [k for k in base if k in keys]
        feats = [[per_class_feats[cls][i][k] for k in base] for i in range(len(per_class_feats[cls]))]
        r2, n = loo_r2(feats, per_class_y[cls])
        entry = {"n": n, "features": base, "loo_r2": r2}
        if "logOutSpatial" in keys:
            idx2 = [i for i in range(len(per_class_feats[cls])) if "logOutSpatial" in per_class_feats[cls][i]]
            feats2 = [[per_class_feats[cls][i][k] for k in base + ["logOutSpatial"]] for i in idx2]
            r2b, nb = loo_r2(feats2, [per_class_y[cls][i] for i in idx2])
            entry["loo_r2_with_outspatial"] = r2b
            entry["n_with_outspatial"] = nb
        geo[cls] = entry
    res["A3"] = {"per_class": geo, "unparsed": unparsed,
                 "n_joined": sum(v["n"] for v in geo.values()),
                 "s74_had_joined": 42, "s74_excluded": 10}
    best = max([v.get("loo_r2") for v in geo.values() if v.get("loo_r2") is not None], default=None)
    res["A3"]["best_loo_r2"] = best

    # A4 tree identity
    bit = [r for r in exceed if r.get("bias") and r.get("a_vs_b_bit_identical") is not None]
    g_true = [r["bias"] for r in bit if r["a_vs_b_bit_identical"]]
    g_false = [r["bias"] for r in bit if not r["a_vs_b_bit_identical"]]
    tree = {"n_bit_identical": len(g_true), "n_tree_changed": len(g_false),
            "median_bit_identical": sorted(g_true)[len(g_true) // 2] if g_true else None,
            "median_tree_changed": sorted(g_false)[len(g_false) // 2] if g_false else None,
            "spread_bit_identical": spread(g_true), "spread_tree_changed": spread(g_false)}
    if len(g_true) >= 5 and len(g_false) >= 5:
        H, p = kw_h([g_true, g_false])
        tree["rank_test"] = {"H": H, "p_perm": p}
    names = {}
    for r in exceed:
        names.setdefault(r.get("kernel_top1") or "NONE", []).append(r.get("bias"))
    dup = {k: v for k, v in names.items() if len(v) >= 2}
    tree["distinct_top1"] = len(names)
    tree["repeated_top1_groups"] = len(dup)
    tree["within_name_spreads"] = {k[:60]: spread([x for x in v if x]) for k, v in dup.items()}
    res["A4"] = tree

    # A4b POST-HOC DESCRIPTIVE (declared after the prereg run; no threshold changed): group by the
    # matched engine kernel name rather than the profiler top-1 (which is sometimes a layout-conversion
    # or memcpy kernel on the first call). Reports within-instance spread and the between-group share of
    # log10(bias) variance for groups with n>=3.
    inst = {}
    for r in exceed:
        if r.get("bias"):
            inst.setdefault(r.get("kernel_top1_primary") or "NONE", []).append(r["bias"])
    inst_rows = []
    for k, v in inst.items():
        inst_rows.append({"name": k, "n": len(v), "min": min(v), "max": max(v),
                          "median": sorted(v)[len(v) // 2], "within_spread": spread(v)})
    inst_rows.sort(key=lambda d: -d["n"])
    groups = [v for v in inst.values() if len(v) >= 3]
    between_share = None
    if groups:
        import statistics
        grand = [math.log10(x) for v in groups for x in v]
        gm = statistics.mean(grand)
        ss_tot = sum((x - gm) ** 2 for x in grand)
        ss_within = 0.0
        for v in groups:
            lv = [math.log10(x) for x in v]
            m = statistics.mean(lv)
            ss_within += sum((x - m) ** 2 for x in lv)
        between_share = 1 - ss_within / ss_tot if ss_tot else None
    res["A4b_engine_instance"] = {
        "note": "post-hoc descriptive, declared after the prereg run",
        "distinct_instances": len(inst), "rows": sum(len(v) for v in inst.values()),
        "instances": inst_rows,
        "between_group_log_variance_share_groups_n_ge_3": between_share,
        "n_groups_n_ge_3": len(groups),
        "n_rows_in_groups_n_ge_3": sum(len(v) for v in groups),
        "tight_groups_le_5x": sum(1 for d in inst_rows if d.get("within_spread") and d["within_spread"] <= 5),
        "wide_groups_gt_100x": sum(1 for d in inst_rows if d.get("within_spread") and d["within_spread"] > 100),
    }

    # A5 controls
    ctl = [{"level": r["level"], "problem_id": r["problem_id"], "bias": r.get("bias"),
            "class": r.get("engine_class_primary")} for r in controls]
    res["A5"] = {"controls": ctl, "max_bias": max([c["bias"] for c in ctl if c["bias"]], default=None),
                 "instrument_ok": all((c["bias"] or 0) <= 0.10 for c in ctl)}

    # verdicts
    verdicts = []
    for field in ("engine_class_primary", "engine_class_refined"):
        a2 = res["A2"][field]
        df = a2.get("dominant_frac") or 0
        spreads = [v["spread"] for v in a2["per_class"].values() if v.get("spread")]
        test = a2.get("rank_test")
        if df >= 0.80:
            verdicts.append("CLASS_NOT_EXPLANATORY(%s:%.2f)" % (field, df))
        elif test and test["p_perm"] < 0.05 and spreads and max(spreads) < 10:
            verdicts.append("CLASS_EXPLANATORY(%s)" % field)
        else:
            verdicts.append("CLASS_PARTIAL(%s)" % field)
    if res["A3"]["n_joined"] < 45:
        verdicts.append("GEOMETRY_PARTIAL(n=%d)" % res["A3"]["n_joined"])
    elif best is not None and best >= 0.50:
        verdicts.append("GEOMETRY_REVIVED")
    else:
        verdicts.append("GEOMETRY_STILL_DEAD")
    t = res["A4"]
    if t.get("rank_test") and t["rank_test"]["p_perm"] < 0.05:
        verdicts.append("TREE_FACTOR")
    res["verdicts"] = verdicts

    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
