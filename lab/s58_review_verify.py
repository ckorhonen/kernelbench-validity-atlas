#!/usr/bin/env python3
"""s58: verify review findings on submission 5WxkQjcUg_JD3wtv from committed records.
Implements lab/s58_review_verify_prereg.md verbatim. CPU only."""
import json, math, os, hashlib, random, statistics as st
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

def _kb_res_dir(here):
    for cand in (os.path.join(here, "results"), os.path.join(os.path.dirname(here), "results")):
        if os.path.isdir(cand):
            return cand
    return os.path.join(here, "results")


def _kb_open(path):
    if os.path.exists(path):
        return open(path)
    if os.path.exists(path + ".gz"):
        import gzip as _gz
        return _gz.open(path + ".gz", "rt")
    raise FileNotFoundError(path)


RES = _kb_res_dir(HERE)
OUT = os.path.join(RES, "s58_review_verify.json")
PREREG = os.path.join(HERE, "s58_review_verify_prereg.md")
PREREG_SHA = hashlib.sha256(open(PREREG, "rb").read()).hexdigest()[:16]

def fam(a):
    return "torch" if a == "torch" else ("compile" if a == "torch_cmp" else "hand")

def auc(scores, labels):
    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    rank_sum = 0.0; i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            if pairs[k][1] == 1:
                rank_sum += avg_rank
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

# ---- population
rows = [json.loads(l) for l in _kb_open(os.path.join(RES, "r1_conv_gated.jsonl"))]
recs = []
for r in rows:
    key = (r["tag"], r["C"], r["H"], r["W"], r["N"], r["K"])
    for a, v in r["variants"].items():
        if v.get("pass_1e4") is None:
            continue
        recs.append(dict(key=key, arm=a, fam=fam(a), prec=v.get("precision"),
                         y=1 if v["pass_1e4"] else 0,
                         feats=[math.log2(r["C"]), math.log2(r["K"]), math.log2(r["H"] * r["W"]), float(r["N"]),
                                math.log2(r["bytes"]),
                                1.0 if fam(a) == "compile" else 0.0,
                                1.0 if fam(a) == "hand" else 0.0,
                                1.0 if v.get("precision") == "tf32" else 0.0]))
n = len(recs); pos = sum(x["y"] for x in recs)
keys = sorted({x["key"] for x in recs})

out = {"date": "2026-10-29", "prereg_sha": PREREG_SHA, "source": "r1_conv_gated.jsonl",
       "population": {"n_armchecks": n, "positives": pos, "shapes": len(keys)}}

# V1 arm-class rule
tp = tn = fp = fn = 0
for x in recs:
    pred = 1 if x["fam"] in ("torch", "compile") else 0
    if pred == 1 and x["y"] == 1: tp += 1
    elif pred == 0 and x["y"] == 0: tn += 1
    elif pred == 1: fp += 1
    else: fn += 1
acc = (tp + tn) / n
out["V1_armclass_baseline"] = {"rule": "pass iff fam in {torch,compile}; fail iff hand",
                               "accuracy": round(acc, 6), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                               "pooled_majority": round(max(pos, n - pos) / n, 6),
                               "gain_over_pooled_majority_pp": round(100 * (acc - max(pos, n - pos) / n), 4)}

# V2 declared-features-only LPM, leave-one-shape-out
X = np.array([x["feats"] for x in recs]); y = np.array([x["y"] for x in recs], float)
pred = np.zeros(n)
for k in keys:
    tr = np.array([i for i, x in enumerate(recs) if x["key"] != k])
    te = np.array([i for i, x in enumerate(recs) if x["key"] == k])
    mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1.0
    Ztr = (X[tr] - mu) / sd; Zte = (X[te] - mu) / sd
    A = np.hstack([np.ones((len(tr), 1)), Ztr])
    beta = np.linalg.lstsq(A, y[tr], rcond=None)[0]
    pred[te] = np.hstack([np.ones((len(te), 1)), Zte]) @ beta
auc_v2 = auc(list(pred), list(y)); acc_v2 = float(((pred > 0.5) == y).mean())
out["V2_declared_only"] = {"features": "log2C,log2K,log2H*W,N,log2bytes,compile,hand,tf32 (no refdev, no interaction)",
                           "auc": round(auc_v2, 6), "accuracy": round(acc_v2, 6),
                           "bar_auc_ge_0.90": bool(auc_v2 >= 0.90),
                           "r57_reported_auc": 0.986, "r57_reported_accuracy": 0.964}

# V3 hand-ieee stratum, shape-clustered bootstrap
hi = [i for i, x in enumerate(recs) if x["fam"] == "hand" and x["prec"] == "ieee"]
hi_pos_keys = sorted({recs[i]["key"] for i in hi if recs[i]["y"] == 1})
auc_hi = auc([pred[i] for i in hi], [recs[i]["y"] for i in hi])
by_key = {}
for i in hi:
    by_key.setdefault(recs[i]["key"], []).append(i)
rng = random.Random(58); boot = []
klist = sorted(by_key)
for _ in range(1000):
    samp = []
    for _k in klist:
        samp.extend(by_key[rng.choice(klist)])
    a = auc([pred[i] for i in samp], [recs[i]["y"] for i in samp])
    if a is not None:
        boot.append(a)
boot.sort()
admits = sum(1 for i in hi if pred[i] > 0.5 and recs[i]["y"] == 1)
out["V3_hand_ieee"] = {"n": len(hi), "positives": sum(recs[i]["y"] for i in hi),
                       "shapes_with_positives": len(hi_pos_keys),
                       "auc": round(auc_hi, 6) if auc_hi else None,
                       "cluster_boot_ci95": [round(boot[int(.025 * len(boot))], 4), round(boot[int(.975 * len(boot)) - 1], 4)],
                       "positives_admitted_at_0.5": admits,
                       "bar_ci_lower_ge_0.70": bool(boot[int(.025 * len(boot))] >= 0.70)}

# V4 fallback accounting. PRIMARY source = the recorded run (s56_heldout_contract_recorded.json, the artifact
# the reviewer read); SECONDARY = the fresh regeneration (s56_heldout_contract.json, 2026-10-29T06:11Z). The
# reviewer's 3.49614% is a function of the recorded times, so reproducing it requires the recorded file; quoting
# only the fresh run would silently replace the population under test.
def compute_v4(rows):
    regA = []; regB = []; detail = []; regfam = []
    for r in rows:
        arms = r["arms"]; gate = r["gate_choice"]; ga = arms.get(gate, {})
        adm = [k for k, v in arms.items() if v.get("pass_full") and v.get("t_ms") is not None]
        if not adm:
            continue
        best = min(adm, key=lambda k: arms[k]["t_ms"]); tb = arms[best]["t_ms"]
        if ga.get("pass_full") and ga.get("t_ms") is not None:
            td = ga["t_ms"]
        else:
            cand = [k for k in adm if k == "torch"]
            if not cand:
                continue
            td = arms["torch"]["t_ms"]
        regA.append((td - tb) / tb); regB.append(0.0)
        regfam.append((r["family"], (td - tb) / tb))
        if not ga.get("pass_full"):
            detail.append({"family": r["family"], "shape": {k: r.get(k) for k in ("K", "M", "H", "C") if r.get(k) is not None},
                           "gate": gate, "policy_A_regret": round((td - tb) / tb, 6)})
    meanA = st.mean(regA); sdA = st.stdev(regA) if len(regA) > 1 else 0.0
    seA = sdA / len(regA) ** .5

    def _split(fam):
        xs = [x for f, x in regfam if f == fam]
        if not xs:
            return {"n": 0, "policy_a_mean_pp": None, "sd_pp": None, "se_pp": None}
        m = st.mean(xs); sd = st.stdev(xs) if len(xs) > 1 else 0.0
        return {"n": len(xs), "policy_a_mean_pp": round(100 * m, 5), "sd_pp": round(100 * sd, 4),
                "se_pp": round(100 * sd / len(xs) ** .5, 5)}

    return {"n": len(regA), "policy_A_mean_pp": round(100 * meanA, 5),
            "policy_A_sd_pp": round(100 * sdA, 4), "policy_A_se_pp": round(100 * seA, 5),
            "policy_B_mean_pp": 0.0,
            "reviewer_claim_mean_pp": 3.49614, "reviewer_claim_se_pp": 1.37116,
            "delta_pp": round(100 * meanA - 3.49614, 5),
            "bar_within_0.05pp": abs(100 * meanA - 3.49614) <= 0.05,
            "failing_gate_rows": detail,
            "manuscript_claim": 3.5304,
            "posthoc_family_split": {"label": "POST-HOC decomposition of a declared bar that failed (pooled 34 rows); the manuscript's 3.5304% was a reduction-only figure",
                                     "reduction": _split("reduction"), "conv": _split("conv")}}


rec_rows = json.load(_kb_open(os.path.join(RES, "s56_heldout_contract_recorded.json")))["rows"]
out["V4_fallback_accounting"] = compute_v4(rec_rows)
out["V4_fallback_accounting"]["source"] = "s56_heldout_contract_recorded.json (the run the reviewer read)"

fresh_rows = json.load(_kb_open(os.path.join(RES, "s56_heldout_contract.json")))["rows"]
v4f = compute_v4(fresh_rows)
v4f["source"] = "s56_heldout_contract.json regenerated 2026-10-29T06:11Z (fresh timings; run-to-run variance)"
try:
    s56a = json.load(_kb_open(os.path.join(RES, "s56_analysis.json")))["summary"]
    v4f["producer_summary"] = {"reduction_policy_a_mean_pp": round(100 * s56a["reduction"]["regret_policy_a_fallback"]["mean"], 5),
                               "reduction_policy_a_se_pp": round(100 * s56a["reduction"]["regret_policy_a_fallback"]["se"], 5),
                               "conv_policy_a_mean_pp": round(100 * s56a["conv"]["regret_policy_a_fallback"]["mean"], 5)}
except Exception as e:
    v4f["producer_summary"] = {"error": str(e)}
out["V4_fresh_remeasurement"] = v4f

# V5 cf3 denominator
cf = json.load(open(os.path.join(RES, "cf43.json")))
cf3 = cf["cf3_cross_family"]
flips = cf3["pass_flips"]; arms_compared = cf3["arms_compared"]; rider = cf["cf1_chain_order"]["rider_cells"]
red = arms_compared - rider
out["V5_cf3_denominator"] = {"flips": flips, "recorded_arms_compared": arms_compared, "cf1_rider_cells": rider,
                             "corrected_denominator": red, "corrected_frac": round(flips / red, 4),
                             "published_frac": cf3["flip_frac"],
                             "mechanism": "tot not reset before cf3; 528 = 168 rider cells + 360 reduction comparisons; tot_arm unused"}

# V4 addendum: POST-HOC family split of a declared bar that failed. Kept in the script rather than
# hand-added to the artifact (round-13 lesson: a hand-added section is erased by the next regeneration).
def _split(fam):
    xs = [x for f, x in regfam if f == fam]
    if not xs:
        return {"n": 0, "policy_a_mean_pp": None, "sd_pp": None, "se_pp": None}
    m = st.mean(xs); sd = st.stdev(xs) if len(xs) > 1 else 0.0
    return {"n": len(xs), "policy_a_mean_pp": round(100 * m, 5), "sd_pp": round(100 * sd, 4),
            "se_pp": round(100 * sd / len(xs) ** .5, 5)}


try:
    s56a = json.load(_kb_open(os.path.join(RES, "s56_analysis.json")))["summary"]
    rec = {"reduction_all_rows_pp": round(100 * s56a["reduction"]["regret_all_rows"]["mean"], 5),
           "reduction_se_pp": round(100 * s56a["reduction"]["regret_all_rows"]["se"], 5),
           "conv_all_rows_pp": round(100 * s56a["conv"]["regret_all_rows"]["mean"], 5)}
except Exception as e:  # the artifact is optional; the split does not depend on it
    rec = {"error": str(e)}

out["V4_addendum_posthoc_family_split"] = {
    "label": "POST-HOC decomposition of a declared bar that failed (pooled 34 rows); reported because the manuscript's 3.5304% was a reduction-only figure",
    "reduction": _split("reduction"), "conv": _split("conv"),
    "recorded_s56_analysis": rec,
    "reading": "the reduction-only split is the reviewer's population; the recorded all-rows conv mean is negative because that field charged a failing arm's time (fixed in the producer, regenerated 2026-10-29)"}

json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps(out, indent=1))
