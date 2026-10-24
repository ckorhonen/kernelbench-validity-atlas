"""s48 prose-audit: recompute every number destined for paper sections 3.3/3.4/3.5
directly from committed artifacts, BEFORE writing the prose (P1 provenance).

Conventions frozen here, pre-run:
  delivered_gate(row) = the arm the published gate would deliver under the CONTRACT-LETTER
  reading: hand iff H<=14 and C<=64 AND that hand passes the 1e-4 contract against the
  row's own recorded reference (pass_1e4); otherwise torch. This is the same convention as
  r1_conv_gated_summary 'by_tag' g (gate regret) and r45 rider A1.
  'gate == torch' means the delivered arm is torch, i.e. delivery equals the baseline.
  d_i = (torch_cmp - delivered)/torch_cmp*100, per-row from r42_analysis (already computed).
No new measurements; CPU only; reads lab/results/*.

Outputs: lab/results/s48_prose_audit.json
"""
import json, os, collections, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


out = {"date": "2026-10-23T18:xxZ", "prereg": "conventions in module docstring"}

# ---- r1 conv-gated per-row: gate delivery under contract-letter reading
r1 = load_jsonl(os.path.join(RES, "r1_conv_gated.jsonl"))
gate_torch = collections.Counter()
gate_hand = collections.Counter()
hand_pass = collections.Counter()
for r in r1:
    tag = r["tag"]
    v = r["variants"]
    chosen = None
    if r["H"] <= 14 and r["C"] <= 64:
        for prec in ("ieee", "tf32"):
            for k in v:
                if k.startswith("imp_B32x32[" + prec + "]"):
                    if v[k].get("pass_1e4"):
                        chosen = k
                        break
            if chosen:
                break
    if chosen:
        gate_hand[tag] += 1
    else:
        gate_torch[tag] += 1
    # per-arm hand pass counts under the same reading
    for k, a in v.items():
        if k.startswith("imp_") and a.get("pass_1e4"):
            hand_pass[(tag, k.split("[")[1].rstrip("]"))] += 1

n = len(r1)
out["r1_gate_delivery"] = {
    "n_rows": n,
    "gate_delivers_torch": sum(gate_torch.values()),
    "gate_delivers_torch_by_tag": dict(gate_torch),
    "gate_delivers_hand": sum(gate_hand.values()),
    "gate_delivers_hand_by_tag": dict(gate_hand),
    "hand_pass_counts": {f"{t}/{p}": c for (t, p), c in sorted(hand_pass.items())},
}

# ---- gate regret / d_i from r42_analysis
r42 = json.load(open(os.path.join(RES, "r42_analysis.json")))
rows42 = r42["per_row"]
adm_rows_pass = [r for r in rows42 if r.get("n_pass", 0) > 0]
adm_rows_pass32 = [r for r in rows42 if r.get("n_pass32", 0) > 0]
out["r42"] = {
    "n_rows": len(rows42),
    "rows_with_passing_hand_contract": len(adm_rows_pass),
    "rows_with_passing_hand_scenario_pass32": len(adm_rows_pass32),
    "gate_regret_max_all_rows": max(r["gate_regret_pct"] for r in rows42),
    "gate_regret_max_on_passing_rows": max((r["gate_regret_pct"] for r in adm_rows_pass), default=None),
    "gate_delivered_hand_rows": sum(1 for r in rows42 if r["gate"] != "torch"),
    "gate_delivered_hand_by_tag": dict(collections.Counter(r["tag"] for r in rows42 if r["gate"] != "torch")),
    "oracle_is_torch_by_tag": dict(collections.Counter(r["tag"] for r in rows42 if r["oracle"] == "torch")),
    "d_i_pct_mean_by_tag": {t: v["d_i_pct_mean"] for t, v in r42["by_tag"].items()},
    "d_i_t_by_tag": {t: v["d_i"]["t"] for t, v in r42["by_tag"].items()},
    "d_i_sd_by_tag": {t: v["d_i"]["sd"] for t, v in r42["by_tag"].items()},
    "gate_beats_torch_by_tag": {t: v["gate_beats_torch"] for t, v in r42["by_tag"].items()},
    "gate_beats_cmp_by_tag": {t: v["gate_beats_cmp"] for t, v in r42["by_tag"].items()},
}

# ---- torch_cmp failing rows inside the paired d_i denominator (F2)
f24 = json.load(open(os.path.join(RES, "f24_verify.json")))
fail_rows = [tuple(x) for x in f24["f2"]["torch_cmp_failing"]["rows"]]
in_union = sum(1 for r in rows42 if (r["tag"], r["N"], r["H"], r["C"]) in set(fail_rows))
out["f24"] = {
    "torch_cmp_failing_n": len(fail_rows),
    "in_r42_paired_denominator": in_union,
    "f2_corrected": f24["f2"]["hand_pass_default"]["corrected"],
    "f2_ieee_vs_fp32ref_corrected": f24["f2"]["ieee_pass_fp32ref"]["corrected"],
    "f2_asrun_hand": f24["f2"]["hand_pass_default"]["r1_asrun"],
    "f2_asrun_ieee": f24["f2"]["ieee_pass_fp32ref"]["r1_asrun"],
    "f3_torch_fails_fp64": f24["f3"]["torch_fails_fp64_anchor"],
    "f3_bootstrap_median": f24["f3"]["bootstrap_torch_flip_median"],
    "f3_anchor_median_claim": 0.9286,
}

# ---- CF1 / CF2 / CF3 from cf43
cf = json.load(open(os.path.join(RES, "cf43.json")))
out["cf43"] = {
    "cf1_conv_rows": cf["cf1_chain_order"]["m3_conv_rows"],
    "cf1_conv_order_dependent": cf["cf1_chain_order"]["order_dependent"],
    "cf1_rider_rows": cf["cf1_chain_order"]["rider_rows"],
    "cf1_rider_cells": cf["cf1_chain_order"]["rider_cells"],
    "cf1_rider_order_dependent": cf["cf1_chain_order"]["rider_order_dependent"],
    "cf2_profile_counts": cf["cf2_anchor_profile"]["profile_counts"],
    "cf2_iee64_exact_agreement": cf["cf2_anchor_profile"]["iee64_exact_agreement"],
    "cf2_arm_level_patterns": cf["cf2_anchor_profile"]["arm_level_patterns"],
    "cf3_rows_matched": cf["cf3_cross_family"]["rows_matched"],
    "cf3_arms_compared": cf["cf3_cross_family"]["arms_compared"],
    "cf3_pass_flips": cf["cf3_cross_family"]["pass_flips"],
    "cf3_flip_frac": cf["cf3_cross_family"]["flip_frac"],
    "conv_contrast": cf["cf3_cross_family"]["conv_contrast"],
}

# ---- r1 summary (day1 as-run) for the as-run vs corrected contrast
r1s = json.load(open(os.path.join(RES, "r1_conv_gated_summary.json")))
out["r1_summary"] = {
    "checks": r1s["checks"],
    "passes": r1s["passes"],
    "ieee_pass": r1s["imp_by_precision"]["[ieee"]["n_pass"],
    "ieee_n": r1s["imp_by_precision"]["[ieee"]["n"],
    "tf32_pass": r1s["imp_by_precision"]["[tf32"]["n_pass"],
    "tf32_n": r1s["imp_by_precision"]["[tf32"]["n"],
}

# ---- r45 rider: A1/A2 delivery numbers cited in 3.4/4
r45 = json.load(open(os.path.join(RES, "r45_rider_adm.json")))
out["r45"] = {
    "torch_pass_own_check": r45["torch_pass_own_check"],
    "a1_delivered_change_def64_published": r45["a1_delivered_change_def64_published"],
    "per_ref": {k: {kk: vv for kk, vv in v.items() if kk in
                    ("n", "oracle_undef", "no_delivery", "arm_disagree_a2_vs_a1",
                     "paired_delta_pp_mean", "paired_n", "a2_change_both_arms", "a2_change_to_none")}
                for k, v in r45["per_ref"].items()},
}

# ---- m3_r42_analysis: family-level oracle population for 3.3 (conv all-torch)
m3 = json.load(open(os.path.join(RES, "m3_r42_analysis.json")))
out["m3_r42"] = {
    "reduction_rows_with_passing_hand": m3["verdicts"]["reduction"]["rows_with_passing_hand"],
    "reduction_n": m3["verdicts"]["reduction"]["n"],
    "conv_rows_with_passing_hand": m3["verdicts"]["conv"]["rows_with_passing_hand"],
    "conv_n": m3["verdicts"]["conv"]["n"],
    "conv_torch_only_rows": m3["verdicts"]["conv"]["torch_only_rows"],
    "reduction_torch_only_rows": m3["verdicts"]["reduction"]["torch_only_rows"],
}

path = os.path.join(RES, "s48_prose_audit.json")
json.dump(out, open(path, "w"), indent=1)
print(json.dumps(out, indent=1))
