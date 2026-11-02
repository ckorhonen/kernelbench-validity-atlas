#!/usr/bin/env python
"""s79a_dgrad_instance_trend.py — is the 10,654x within-instance dgrad span monotone in a declared derived coordinate?

PREREG (frozen before the run; do not edit after).

Context: s75b grouped the 49 s73 exceed rows into 16 engine kernel instances. One instance
(sm80_xmma_dgrad_implicit_gemm_indexed_tf32..._nhwc..., s75 kernel_top1/kernels match) has n=12 rows spanning
4.5378e-05 to 0.4835 in s73_bias_1e4 = 10,654x. s74's geometry model failed population-wide (LOO R^2 -0.99 conv /
-0.19 conv_transpose). This script asks a narrower question: inside ONE dispatch name, is the reference budget
ratio monotone in an observable shape coordinate?

Population (rule, frozen): every s75_engine_class_s73pop.jsonl row whose `kernels` list contains the substring
"sm80_xmma_dgrad_implicit_gemm_indexed_tf32", kind == "exceed". Rows with s73_bias_1e4 <= 0 are excluded from
rank tests (log undefined) and counted, not silently dropped.

Declared coordinates (all derived from the recorded shapes/params, no new measurement):
  K      = accumulation depth per output element = n_params / Cout, Cout = output channel count
           (conv_transpose weight is (Cin, Cout, *kernel); each output element sums over Cin*prod(kernel) terms)
  dim    = number of spatial dims of the input (1/2/3)
  log_in_vol, log_out_vol, log_params, log_spatial_extent (product of input spatial dims)

Test: Spearman rank correlation between log10(s73_bias_1e4) and each declared coordinate, with a permutation
p-value (10000 permutations, seed 20261102, one-sided |rho|). n is reported; no correction for multiplicity is
applied because the rule is a disjunction over pre-declared coordinates and every rho is printed.

FROZEN VERDICT RULE (single branch, no post-hoc edits):
  MONOTONE_INSIDE_INSTANCE if any declared coordinate reaches |rho| >= 0.6 with p < 0.05;
  IDIOSYNCRATIC_AT_INSTANCE otherwise.
Both outcomes are reportable; neither is a kill of the reference axis. A monotone result would say the within-
instance span is a depth/extent law and the earlier population-wide failure was a mixture effect; an idiosyncratic
result says the span is per-shape idiosyncratic even inside one dispatch, which is s74's conclusion at finer
resolution.

Kill/limits declared: n=12 (2 rows at bias 0), one dispatch name, one population (s73 exceed rows), no GPU, no
new draw. A monotone finding here is a hypothesis for s79b's fixed-tuple test, not a law.
"""
import json, math, os, random

LAB = os.path.expanduser("~/rstation/kernelbench/lab")
OUT = os.path.join(LAB, "results", "s79a_dgrad_instance_trend.json")
SRC = os.path.join(LAB, "results", "s75_engine_class_s73pop.jsonl")
NEEDLE = "sm80_xmma_dgrad_implicit_gemm_indexed_tf32"


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def perm_p(x, y, obs, iters=10000, seed=20261102):
    rnd = random.Random(seed)
    n = len(y)
    cnt = 0
    for _ in range(iters):
        ys = y[:]
        rnd.shuffle(ys)
        r = spearman(x, ys)
        if r is not None and abs(r) >= abs(obs):
            cnt += 1
    return (cnt + 1) / (iters + 1)


rows = [json.loads(l) for l in open(SRC)]
sel = [r for r in rows if any(NEEDLE in str(k) for k in (r.get("kernels") or []))]
pop = [r for r in sel if r.get("kind") == "exceed"]
print("selected", len(sel), "exceed", len(pop), flush=True)

recs = []
excluded_zero = []
for r in pop:
    bias = r.get("s73_bias_1e4")
    ins = r.get("input_shapes") or []
    outs = r.get("output_shape") or []
    try:
        in_shape = ins[0]
        cin = int(in_shape[1])
        cout = int(outs[1])
        spatial = 1
        for d in in_shape[2:]:
            spatial *= int(d)
        in_vol = 1
        for d in in_shape:
            in_vol *= int(d)
        out_vol = 1
        for d in outs:
            out_vol *= int(d)
        params = int(r.get("n_params"))
        dim = len(in_shape) - 2
        K = params / float(cout)
    except Exception as e:
        print("WARN unparsed", r.get("level"), r.get("problem_id"), e, flush=True)
        continue
    rec = dict(level=r["level"], problem_id=r["problem_id"], name=r["name"], bias=bias,
               K=K, dim=dim, log_in_vol=math.log10(in_vol), log_out_vol=math.log10(out_vol),
               log_params=math.log10(params), log_spatial=math.log10(spatial),
               frac=r.get("s73_ref_over_budget_frac"), kernel_top1=r.get("kernel_top1"))
    if bias is None or bias <= 0:
        excluded_zero.append(rec)
    else:
        recs.append(rec)

print("usable", len(recs), "excluded(bias<=0)", len(excluded_zero), flush=True)
for rec in recs:
    print("  L%d/%d %s bias=%.3e K=%.0f dim=%d log_in=%.2f log_out=%.2f" %
          (rec["level"], rec["problem_id"], rec["name"][:34], rec["bias"], rec["K"], rec["dim"],
           rec["log_in_vol"], rec["log_out_vol"]), flush=True)

y = [math.log10(r["bias"]) for r in recs]
coords = ["K", "dim", "log_in_vol", "log_out_vol", "log_params", "log_spatial"]
results = {}
for c in coords:
    x = [float(r[c]) for r in recs]
    rho = spearman(x, y)
    p = perm_p(x, y, rho) if rho is not None else None
    results[c] = {"rho": rho, "p_perm": p, "n": len(recs)}
    print("%-14s rho=%s p=%s" % (c, None if rho is None else round(rho, 4),
                                 None if p is None else round(p, 5)), flush=True)

hits = [c for c in coords if results[c]["rho"] is not None and abs(results[c]["rho"]) >= 0.6
        and results[c]["p_perm"] is not None and results[c]["p_perm"] < 0.05]
verdict = "MONOTONE_INSIDE_INSTANCE" if hits else "IDIOSYNCRATIC_AT_INSTANCE"

# descriptive: is the span reproducible from a 1-feature fit in log space?
span = max(r["bias"] for r in recs) / min(r["bias"] for r in recs)
best = max(coords, key=lambda c: abs(results[c]["rho"] or 0.0))

out = {
    "prereg": "s79a docstring frozen pre-run",
    "population_rule": "s75 rows whose kernels contain %s, kind=exceed" % NEEDLE,
    "n_selected": len(sel), "n_exceed": len(pop), "n_usable": len(recs),
    "n_excluded_bias_zero": len(excluded_zero),
    "excluded": excluded_zero,
    "rows": recs,
    "within_instance_span": span,
    "tests": results,
    "hit_coordinates": hits,
    "best_coordinate": best,
    "verdict": verdict,
    "limits": ["n=12", "one dispatch name", "s73 exceed population only", "no GPU",
               "log-rank test only; a monotone result is a hypothesis for s79b, not a law"],
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w"), indent=1)
print("VERDICT", verdict, "span", round(span, 1), "best", best, flush=True)
print("wrote", OUT, flush=True)
