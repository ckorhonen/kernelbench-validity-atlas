#!/usr/bin/env python
"""s79c_instance_regroup.py — CORRECTION: instance membership must be dispatch identity (top-1), not kernel containment.

Declared this tick, before running: s79b's engine check found that level1/79, which s79a counted inside the
indexed-dgrad instance because s75's `kernels` list contains that name, actually dispatches
`cudnn::detail::dgrad_engine<float,128,6,8,3,3,5,false>` as its top-1 kernel on GB10 today. So s75b's A4b
grouping (any-kernel containment) mixes tasks that merely touch a kernel with tasks whose dominant dispatch is
that kernel. This script re-derives the grouping on the recorded s75 rows using top-1 identity only and reports:

  (a) how many top-1 instances exist, their n and budget span (s75b reported 16 instances / 49 rows with the
      containment rule and a 10,654x dgrad span);
  (b) the corrected dgrad instance: its n, span, and the s79a rank test re-run on that population with the SAME
      frozen rule (|rho|>=0.6, p<0.05 over the declared coordinates);
  (c) how many rows move out of the containment group when membership is restricted to top-1.

No new measurement, no GPU. Population = the same s75_engine_class_s73pop.jsonl exceed rows. Rows with
s73_bias_1e4 <= 0 are excluded from rank tests and counted.

This is a correction of a labelling rule, not of a measurement: every bias value is s73's recorded number.
"""
import json, math, os, random, collections

LAB = os.path.expanduser("~/rstation/kernelbench/lab")
OUT = os.path.join(LAB, "results", "s79c_instance_regroup.json")
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
    cnt = 0
    for _ in range(iters):
        ys = y[:]
        rnd.shuffle(ys)
        r = spearman(x, ys)
        if r is not None and abs(r) >= abs(obs):
            cnt += 1
    return (cnt + 1) / (iters + 1)


rows = [json.loads(l) for l in open(SRC)]
pop = [r for r in rows if r.get("kind") == "exceed"]
print("exceed rows", len(pop), flush=True)

# (a) top-1 grouping
groups = collections.defaultdict(list)
for r in pop:
    groups[str(r.get("kernel_top1"))].append(r)

group_stats = []
for name, rs in groups.items():
    biases = [r["s73_bias_1e4"] for r in rs if r.get("s73_bias_1e4")]
    pos = [b for b in biases if b > 0]
    span = (max(pos) / min(pos)) if len(pos) >= 2 else None
    group_stats.append({"name": name[:110], "n": len(rs), "span": span,
                        "median": sorted(pos)[len(pos) // 2] if pos else None})
group_stats.sort(key=lambda g: -g["n"])
multi = [g for g in group_stats if g["n"] >= 2]
print("top1 distinct instances", len(group_stats), "multi-row", len(multi), flush=True)
for g in multi[:10]:
    print("  n=%d span=%s %s" % (g["n"], None if g["span"] is None else round(g["span"], 1), g["name"][:80]), flush=True)

# (b) corrected dgrad instance
dg = [r for r in pop if NEEDLE in str(r.get("kernel_top1"))]
contain = [r for r in pop if any(NEEDLE in str(k) for k in (r.get("kernels") or []))]
moved_out = [r for r in contain if r not in dg]
print("containment group", len(contain), "top1 group", len(dg), "moved out", len(moved_out), flush=True)
for r in moved_out:
    print("  moved out L%d/%d top1=%s" % (r["level"], r["problem_id"], str(r.get("kernel_top1"))[:80]), flush=True)

recs = []
for r in dg:
    bias = r.get("s73_bias_1e4")
    ins, outs = r.get("input_shapes") or [], r.get("output_shape") or []
    try:
        in_shape = ins[0]
        cout = int(outs[1])
        K = int(r.get("n_params")) / float(cout)
        spatial = 1
        for d in in_shape[2:]:
            spatial *= int(d)
        in_vol = 1
        for d in in_shape:
            in_vol *= int(d)
        out_vol = 1
        for d in outs:
            out_vol *= int(d)
        dim = len(in_shape) - 2
    except Exception as e:
        print("WARN unparsed", r.get("problem_id"), e, flush=True)
        continue
    recs.append({"level": r["level"], "problem_id": r["problem_id"], "name": r["name"], "bias": bias,
                 "K": K, "dim": dim, "log_in_vol": math.log10(in_vol), "log_out_vol": math.log10(out_vol),
                 "log_params": math.log10(int(r["n_params"])), "log_spatial": math.log10(spatial)})

usable = [r for r in recs if r["bias"] and r["bias"] > 0]
print("top1 dgrad usable", len(usable), flush=True)
tests = {}
verdict = None
if len(usable) >= 4:
    y = [math.log10(r["bias"]) for r in usable]
    for c in ["K", "dim", "log_in_vol", "log_out_vol", "log_params", "log_spatial"]:
        x = [float(r[c]) for r in usable]
        rho = spearman(x, y)
        p = perm_p(x, y, rho) if rho is not None else None
        tests[c] = {"rho": rho, "p_perm": p, "n": len(usable)}
    hits = [c for c in tests if tests[c]["rho"] is not None and abs(tests[c]["rho"]) >= 0.6
            and tests[c]["p_perm"] is not None and tests[c]["p_perm"] < 0.05]
    verdict = "MONOTONE_INSIDE_INSTANCE" if hits else "IDIOSYNCRATIC_AT_INSTANCE"
    span = max(r["bias"] for r in usable) / min(r["bias"] for r in usable)
else:
    span = None
    verdict = "UNDERPOWERED"

out = {
    "correction": "instance membership = top-1 dispatch identity, not kernel-list containment",
    "containment_group_n": len(contain), "top1_group_n": len(dg), "moved_out": [
        {"level": r["level"], "problem_id": r["problem_id"], "kernel_top1": r.get("kernel_top1")}
        for r in moved_out],
    "top1_instances_total": len(group_stats), "top1_multi_row": len(multi),
    "top1_group_stats": group_stats,
    "dgrad_top1_rows": recs, "dgrad_top1_usable": len(usable),
    "dgrad_top1_span": span, "tests": tests, "verdict": verdict,
    "s75b_reported": {"instances": 16, "rows": 49, "dgrad_span": 10654.43425652576,
                      "grouping_rule": "kernel-list containment"},
    "note": "s79a's MONOTONE_INSIDE_INSTANCE verdict was computed on the containment group; this re-run states "
            "whether it survives the corrected membership rule. Both verdicts are recorded, neither overwritten.",
}
json.dump(out, open(OUT, "w"), indent=1)
print("VERDICT", verdict, "span", span, "tests", json.dumps({k: (None if v["rho"] is None else round(v["rho"], 3)) for k, v in tests.items()}), flush=True)
print("wrote", OUT, flush=True)
