#!/usr/bin/env python
"""s106c_map_scale_check.py — does the map's 80x conv-vs-gemm gap reduce to the scale coordinate u?

Declared post-hoc analysis (no GPU, no new rows): re-expresses s105c's map rows in the s91 coordinate
u = 2.2e-4 * ref_max_abs / atol and asks whether the class gap survives it.

Frozen rules (written before this script produced numbers):
  BRANCH_MAP_SCALE_EXPLAINED if the median rho/u of conv|TF32 and gemm|TF32 differ by < 2x AND the median u of
    conv|TF32 is at least 5x below gemm|TF32's.
  BRANCH_MAP_CLASS_RESIDUAL if median rho/u differs by >= 2x with u medians comparable (ratio within 5x).
  BRANCH_MAP_PARTIAL otherwise (both terms present; report the decomposition).
"""
import json, os, collections

S71 = os.path.expanduser("~/kb-lab/results/s71_full_audit.jsonl")
S105B = os.path.expanduser("~/kb-lab/results/s105b_map_engine_class.jsonl")
S105C = os.path.expanduser("~/kb-lab/results/s105c_map_join.jsonl")
OUT = os.path.expanduser("~/kb-lab/results/s106c_map_scale_check.json")
REF_REL = 2.2e-4


def med(xs):
    xs = sorted(xs)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def load(path):
    return [json.loads(l) for l in open(path)]


s71 = {(r["level"], r["problem_id"]): r for r in load(S71)}
joined = load(S105C)
rows = []
for r in joined:
    key = (r.get("level"), r.get("problem_id"))
    src = s71.get(key)
    if not src:
        continue
    ref = src.get("ref_max_abs")
    av = src.get("A_vs_anchor") or {}
    strict = (av.get("ew_1e-4") or {}).get("rho_global")
    bench = (av.get("ew_1e-2") or {}).get("rho_global")
    if ref is None or strict is None:
        continue
    rows.append(dict(level=key[0], pid=key[1], name=src["name"], op_class=src.get("op_class"),
                     engine_class=r.get("engine_class"), ref_max_abs=ref, rho_strict=strict,
                     rho_bench=bench, u_strict=REF_REL * ref / 1e-4, u_bench=REF_REL * ref / 1e-2))

groups = collections.defaultdict(list)
for r in rows:
    groups[(r["op_class"], r["engine_class"])].append(r)

table = {}
for k, v in groups.items():
    us = [r["u_strict"] for r in v]
    rs = [r["rho_strict"] for r in v]
    ratios = [r["rho_strict"] / r["u_strict"] for r in v if r["u_strict"] > 0]
    table["%s|%s" % k] = dict(n=len(v), med_u=med(us), med_rho=med(rs), med_rho_over_u=med(ratios),
                              max_rho=max(rs), min_u=min(us), max_u=max(us))

conv = table.get("conv|TF32_TENSOROP", {})
gemm = table.get("gemm|TF32_TENSOROP", {})
u_ratio = (gemm.get("med_u") or 0) / (conv.get("med_u") or 1) if conv.get("med_u") else None
ru_ratio = (conv.get("med_rho_over_u") or 0) / (gemm.get("med_rho_over_u") or 1) if gemm.get("med_rho_over_u") else None
if ru_ratio is not None and ru_ratio < 2.0 and u_ratio is not None and u_ratio >= 5.0:
    branch = "BRANCH_MAP_SCALE_EXPLAINED"
elif ru_ratio is not None and ru_ratio >= 2.0 and u_ratio is not None and u_ratio < 5.0:
    branch = "BRANCH_MAP_CLASS_RESIDUAL"
else:
    branch = "BRANCH_MAP_PARTIAL"

# within-class monotonicity: Spearman of u vs rho, per class (rank correlation, no scipy)
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

corr = {}
for k, v in groups.items():
    if len(v) >= 6:
        corr["%s|%s" % k] = dict(n=len(v), spearman_u_rho=spearman([r["u_strict"] for r in v],
                                                                  [r["rho_strict"] for r in v]))

out = dict(rows=len(rows), table=table, conv_gemm=dict(u_ratio=u_ratio, rho_over_u_ratio=ru_ratio),
           branch=branch, within_class_corr=corr,
           note="u = 2.2e-4*ref_max_abs/atol, the s91 coordinate; rho/u is the scale-normalized deviation")
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps({k: out[k] for k in ("branch", "conv_gemm")}, indent=1))
print(json.dumps(table, indent=1))
print(json.dumps(corr, indent=1))
