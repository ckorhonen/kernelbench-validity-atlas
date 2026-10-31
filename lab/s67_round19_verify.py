#!/usr/bin/env python3
"""s67 — round-19 rejection verification (CPU arithmetic on committed records).

PREREG (frozen before computation, 2026-10-31 ~09:50Z):

Reviewer claim under test (F1): the manuscript defines the delivered arm as the fastest
admitted arm with a baseline-only fallback when the pool is empty, but the shipped
producers implement a fixed rule (H<=14 and C<=64 -> imp_B32x32 in precision order,
else torch).  Claimed consequences: disjoint choice sets on 19 rows (default->tf32-off)
and 49 rows (default->fp64); 27 anchor rows where the rule returns an inadmissible arm
despite a nonempty pool.

V1 policy recomputation.  Pool = the 12 recorded arms (torch, torch_cmp, 5 imp pairs).
Stated policy S: delivered = argmin t over {k : pass_k}, ties recorded as a choice SET;
if no arm passes, delivered = torch.  Variant S2 excludes torch itself from selection
(fallback only).  For each reference realization (def/ieee/64) and each row, record:
  D1 arm-identity change (delivered arm differs between realizations),
  D2 selected-arm validity change (delivered arm admitted vs not),
  D3 nonempty-pool status change (>=1 admitted arm available),
  D4 admissible-set change (full pool membership),
plus disjoint choice-set pairs (S(def) vs S(ieee), S(def) vs S(64)), and rows where the
RECORDED rule arm is inadmissible while the pool is nonempty.
Predicted by reviewer: disjoint 19 (ieee) / 49 (64); inadmissible-rule 27 (anchor).

V2 attribution counts.  Per-arm verdict flips def->ieee and def->64 from the recorded
pass maps; compare with the manuscript's attribution of 52/49/48 flips to the
engine-class swap (reviewer says those are anchor-swap counts) and 145 for tf32-off.
Predicted: anchor torch 52, torch_cmp 49, each ieee tile 48; ieee 35/35/15 = 145.

V3 input-draw population.  The two held-out artifacts contain 168 reduction and 60
convolution arm comparisons (24x7 and 10x6); all 26 flips are reductions; pooled
fraction 26/228 = 11.40%, reduction-only 26/168 = 15.48%.
Predicted: 168/60/26-in-reduction/11.40/15.48.

V4 crossing neighborhood.  a10_crossing_neighborhood.json at K=3608/3632 reports pass
fractions 4/6 and 2/6 with zero monotone seeds.
Predicted: exactly those numbers.

No thresholds: every check is a reproduction of a stated number.
"""
import json, os, itertools

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
def load(p):
    with open(os.path.join(RES, p)) as f:
        return json.load(f)
def rows(p):
    with open(os.path.join(RES, p)) as f:
        return [json.loads(l) for l in f if l.strip()]

out = {"artifact": "s67_round19_verify", "date": "2026-10-31T09:50:00Z",
       "prereg": "docstring frozen before computation", "checks": {}}

# ---------- V1 ----------
gs = rows("gate_stability.jsonl")
times = json.load(open(os.path.join(HERE, "r1_times.json")))
def policy(passmap, tmap, arms, exclude_torch):
    pool = [a for a in arms if (a != "torch") or not exclude_torch]
    admitted = sorted([a for a in pool if passmap.get(a) and a in tmap])
    if not admitted:
        return {"arm": "torch", "choices": ["torch"], "admitted": [], "valid": True}
    best = min(tmap[a] for a in admitted)
    tie = 1e-12
    choices = sorted([a for a in admitted if abs(tmap[a] - best) <= tie * max(1.0, best)])
    return {"arm": choices[0], "choices": choices, "admitted": admitted,
            "valid": bool(passmap.get(choices[0]))}

res_v1 = {"rows": len(gs), "variants": {}}
for excl in (False, True):
    key = "S2_exclude_torch_from_selection" if excl else "S1_full_pool"
    stats = {}
    for pair in (("def", "ieee"), ("def", "64")):
        d1 = d2 = d3 = d4 = disj = 0
        rule_inadm = 0
        rule_differs = 0
        detail_inadm = []
        for r in gs:
            tag = "%s|%s|%s|%s" % (r["tag"], r["H"], r["N"], r["C"])
            tmap = times.get(tag)
            if tmap is None:
                continue
            arms = r["arms"]
            A = policy(r["pass_" + pair[0]], tmap, arms, excl)
            B = policy(r["pass_" + pair[1]], tmap, arms, excl)
            if A["arm"] != B["arm"]:
                d1 += 1
            if A["valid"] != B["valid"]:
                d2 += 1
            if bool(A["admitted"]) != bool(B["admitted"]):
                d3 += 1
            if A["admitted"] != B["admitted"]:
                d4 += 1
            if not (set(A["choices"]) & set(B["choices"])):
                disj += 1
            rec = r["gate_" + pair[1]]["arm"]
            if rec not in B["admitted"] and B["admitted"]:
                rule_inadm += 1
                detail_inadm.append({"tag": tag, "rule_arm": rec, "n_admitted": len(B["admitted"])})
            if rec != B["arm"]:
                rule_differs += 1
        stats["def->%s" % pair[1]] = {"D1_arm_identity": d1, "D2_selected_validity": d2,
                                      "D3_nonempty_pool": d3, "D4_admissible_set": d4,
                                      "disjoint_choice_sets": disj,
                                      "recorded_rule_inadmissible_with_nonempty_pool": rule_inadm,
                                      "recorded_rule_differs_from_stated_policy": rule_differs,
                                      "inadmissible_detail": detail_inadm}
    res_v1["variants"][key] = stats
out["checks"]["V1_policy"] = res_v1

# ---------- V2 ----------
per_arm = {}
for pair in (("def", "ieee"), ("def", "64")):
    flips = {}
    total = 0
    for r in gs:
        for a in r["arms"]:
            if bool(r["pass_" + pair[0]].get(a)) != bool(r["pass_" + pair[1]].get(a)):
                flips[a] = flips.get(a, 0) + 1
                total += 1
    per_arm["def->%s" % pair[1]] = {"total": total, "per_arm": flips}
out["checks"]["V2_attribution"] = per_arm

# ---------- V3 ----------
rec = load("s56_heldout_contract_recorded.json")
fre = load("s56_heldout_contract.json")
def armcount(d):
    red = sum(len(r["arms"]) for r in d["rows"] if r["family"] == "reduction")
    conv = sum(len(r["arms"]) for r in d["rows"] if r["family"] == "conv")
    return {"reduction_arm_comparisons": red, "conv_arm_comparisons": conv,
            "reduction_rows": sum(1 for r in d["rows"] if r["family"] == "reduction"),
            "conv_rows": sum(1 for r in d["rows"] if r["family"] == "conv")}
v3 = {"recorded": armcount(rec), "fresh": armcount(fre)}
s59 = load("s59_f1_input_draw.json")
v3["s59_population"] = s59["population"]
v3["s59_flips"] = s59["pass_label_flips"]
n_checks = s59["population"]["arm_checks_compared"]
red_checks = v3["recorded"]["reduction_arm_comparisons"]
conv_checks = v3["recorded"]["conv_arm_comparisons"]
v3["recomputed"] = {
    "pooled_frac_26_of_228": 26 / 228,
    "reduction_only_frac_26_of_%d" % red_checks: 26 / red_checks,
    "reduction_plus_conv_checks": red_checks + conv_checks,
    "all_flips_in_reduction": s59["pass_label_flips"]["by_family"].get("conv", 0) == 0,
}
out["checks"]["V3_input_draw"] = v3

# ---------- V4 ----------
a10 = load("a10_crossing_neighborhood.json")
byk = a10.get("by_K", {})
v4 = {"by_K_keys": sorted(byk.keys()) if isinstance(byk, dict) else None}
if isinstance(byk, dict):
    v4["by_K"] = {k: byk[k] for k in ("3608", "3632") if k in byk}
v4["monotonicity"] = a10.get("monotonicity")
v4["verdict"] = a10.get("verdict")
out["checks"]["V4_crossing"] = v4

with open(os.path.join(RES, "s67_round19_verify.json"), "w") as f:
    json.dump(out, f, indent=1, default=str)
print(json.dumps(out["checks"]["V1_policy"]["variants"]["S1_full_pool"], indent=1)[:2600])
print("V2:", json.dumps(per_arm)[:900])
print("V3:", json.dumps(v3["recomputed"]), json.dumps(v3["recorded"]))
print("V4:", json.dumps(v4)[:800])
