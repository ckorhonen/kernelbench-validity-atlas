#!/usr/bin/env python
"""s106c_map_scale_check.py — does the map's 80x conv-vs-gemm gap reduce to the scale coordinate u?

Declared post-hoc analysis of committed records (s105c map join carries ref_max_abs, rho_strict, u_strict per task);
no GPU, no new rows. Frozen rules, written before the numbers:
  BRANCH_MAP_SCALE_EXPLAINED if median rho/u for conv|TF32 and gemm|TF32 differ by < 2x AND conv's median u is at
    least 5x below gemm's.
  BRANCH_MAP_CLASS_RESIDUAL if median rho/u differs >= 2x with u medians within 5x.
  BRANCH_MAP_PARTIAL otherwise, with the decomposition reported.
"""
import json, collections, os

SRC = os.path.expanduser("~/rstation/kernelbench/lab/results/s105c_map_join.json")
OUT = os.path.expanduser("~/rstation/kernelbench/lab/results/s106c_map_scale_check.json")


def med(xs):
    xs = sorted(xs)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def spearman(xs, ys):
    def rank(a):
        order = sorted(range(len(a)), key=lambda i: a[i])
        r = [0.0] * len(a)
        for pos, i in enumerate(order):
            r[i] = pos + 1.0
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    return num / (dx * dy) if dx * dy else None


rows = [r for r in json.load(open(SRC))["rows"]
        if r.get("status") == "ok" and r.get("rho_strict") is not None and r.get("u_strict")]
groups = collections.defaultdict(list)
for r in rows:
    groups[(r.get("op_class"), r.get("engine_class"))].append(r)

table = {}
for k, v in groups.items():
    us = [r["u_strict"] for r in v]
    rs = [r["rho_strict"] for r in v]
    ratios = [r["rho_strict"] / r["u_strict"] for r in v if r["u_strict"] > 0]
    table["%s|%s" % k] = dict(n=len(v), med_u=med(us), med_rho=med(rs), med_rho_over_u=med(ratios),
                              max_rho=max(rs), min_u=min(us), max_u=max(us),
                              frac_med=med([r.get("frac_strict") or 0 for r in v]))

conv = table.get("conv|TF32_TENSOROP", {})
gemm = table.get("gemm|TF32_TENSOROP", {})
u_ratio = (gemm.get("med_u") / conv.get("med_u")) if conv.get("med_u") else None
ru_ratio = (conv.get("med_rho_over_u") / gemm.get("med_rho_over_u")) if gemm.get("med_rho_over_u") else None
if ru_ratio is not None and ru_ratio < 2.0 and u_ratio is not None and u_ratio >= 5.0:
    branch = "BRANCH_MAP_SCALE_EXPLAINED"
elif ru_ratio is not None and ru_ratio >= 2.0 and u_ratio is not None and u_ratio < 5.0:
    branch = "BRANCH_MAP_CLASS_RESIDUAL"
else:
    branch = "BRANCH_MAP_PARTIAL"

corr = {}
for k, v in groups.items():
    if len(v) >= 6:
        corr["%s|%s" % k] = dict(n=len(v),
                                 spearman_u_rho=spearman([r["u_strict"] for r in v],
                                                         [r["rho_strict"] for r in v]))

# pooled: does rho track u across the whole population (the map's class gap being a scale artifact predicts yes)
pooled = spearman([r["u_strict"] for r in rows], [r["rho_strict"] for r in rows])
out = dict(n_rows=len(rows), table=table, conv_gemm=dict(u_ratio=u_ratio, rho_over_u_ratio=ru_ratio),
           branch=branch, within_class_corr=corr, pooled_spearman_u_rho=pooled,
           note="u_strict = 2.2e-4*ref_max_abs/1e-4 (s91 coordinate); rho/u is the scale-normalized deviation")
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps({k: out[k] for k in ("branch", "conv_gemm", "pooled_spearman_u_rho", "n_rows")}, indent=1))
for k, v in table.items():
    print("%-28s n=%3d med_u=%10.2f med_rho=%8.3f med_rho/u=%8.4f max_rho=%8.3f" %
          (k, v["n"], v["med_u"] or 0, v["med_rho"] or 0, v["med_rho_over_u"] or 0, v["max_rho"] or 0))
print(json.dumps(corr, indent=1))
