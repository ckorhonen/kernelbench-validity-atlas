"""S51 prose audit — regenerate every number cited in §4.5, §5, §6, §7, §8 from committed artifacts.

Conventions frozen in this docstring BEFORE the run (P3/P1): each check names its artifact and its field;
a check that cannot find its source is FAIL, not skipped. Exit nonzero on any FAIL.

Checks (fixed list, no post-hoc additions):
 C01 a8_compress_v2 row count == 48 and control count == 4
 C02 a8_compress_v2 runtime facts: cudnn.allow_tf32 True, matmul.allow_tf32 False, precision highest
 C03 a8b corrected class counts: TF32 27, GEMM_FP32 18, W 3
 C04 a8b M3 spearman 0.9862 and M2 0.5735 and M4 0.8327
 C05 a8b M3 bulk_acc 1.0, M4 bulk_acc 0.7917
 C06 a8b within-class spans W .824, TF32 .924, GEMM_FP32 3.042
 C07 a8b class LOO accuracy 0.6042, majority 0.5625
 C08 a8 dispatch boundary: C64 k3s1 H7/H13 = W, H21 = TF32; C256 k3s1 H7 = W, H13/H21 = TF32; C16 k3s1 all TF32
 C09 a8 rho jumps: C64 H13->H21 ratio >= 20x; C256 H7->H13 ratio >= 30x
 C10 a8 controls: im2col 0.7548, cudnn-off 0.4002, tf32off k3s1 12.8068 (W), tf32off k3s2 0.7512
 C11 fp32-SIMT monotone depth ladder: rho(keff16)=0.0051..0.0078, rho(keff8192)>=1.05
 C12 f24_verify f1 corrected crossovers: b1 n6 med195.05, b2 n7 med225.39, b4 n15 med141.38, b8 n21 med232.33
 C13 f24_verify f2: 15/560, 80/280, adm corrected 14/2/0, torch_cmp_failing 5
 C14 f24_verify f3: torch fails fp64 52/56, gate delivers failing def 52 / ieee 41, obsolete shapes 4
 C15 r45_rider_adm: def->64 = 21 arm->arm + 31 ->none + 4 unchanged + 0 none->arm; no admissible 24/56 ieee, 31/56 64
 C16 s6 audit: gate worse1pct 3/24 reduction, 0/10 conv; regret 0.0451/0.0201 vs 1.5277/2.1469
 C17 s6 audit: V2 certificate-health false, S5_DECISION SCOPE_CUT_S5
 C18 cf43: cf1 order_dependent 0/36 and 0/168; cf2 mixed 48 nested 4 flat 4, iee64 agreement .3929, pattern 001=165
 C19 cf43: cf3 169/528 = 0.3201
 C20 cushion: train AUC .7429, c* .4578, rider AUC .1211, rider below c n=10 flip 0
 C21 b5_stage1: pooled share_T2 0.1494
 C22 direction audits: 0/350 corpus, 0/816 held-out label flips
 C23 s50 prose audit: every check passes (>=18), 0 unsupported
 C24 provenance: cited 25 found 25 missing 0
 C25 pop_confirm: P90_fresh 15.3584 <= 16.1783, cells 420
 C26 e2_v3_err: 1260 measurements, med 2-5e-4 range, max 4.2e-2
 C27 b2 corpus: 180 rows, 53 fail / 122 pass
 C28 a7b: Q1 ratios .0016/.0022/.0007 (99.9% cut)
 C29 r1 conv: gate torch 52/56
 C30 paper sections present: s2,s3,s3b,s4,s5,s6,s7,s8 and each cites at least one results/ artifact
 C32 s94 population: 5/5 rows ok, tids T1-T5, M=N in {1024,512}
 C33 s94 share: pooled 300 fail / 168 ref / 132 cand = 0.560, per-tuple .600/.562015503875969/.5405405405405406, spread .05946
 C34 s94 depth: rho_ref .5798813114391407/.5585034175067228/1.2346484213295181/2.449441385699855, T5 1.6524742737102525,
     slope .7380331925187715, resid .321516695619999, crossing bracket 2048<->4096
 C35 s94 controls: C3 bit-identical 5/5, ratio_cand_over_ref 1.0094/1.1081/.9562/.9858/1.1487
 C36 s95 population: 33/33 rows ok, instrument repeat bit-identical (rho, rho_cand, frac)
 C37 s95 floor branch: dip in 1/8 draws, means .5042261951375578/.6263878155027611, FLOOR_NOISE, K2 fired
     (CV .12702/.14305), law claim not permitted
 C38 s95 slope/share: slope mean .7713786472806671 sd .08076628792149573, share mean .5024039188035422 sd .13447622897245026
     min .25 max .75 n 16, SHARE_UNSTABLE, frac_over_ref at 1e-2 exactly 0.0 on all rows
 C39 s96 population: 33 rows, 32 main ok, instrument repeat bit-identical on rho and frac_over, kills K1 ok / K2 no / K3 no
 C40 s96 amplitude: rho_ref min 1.0455225225704408 max 1.4736522134189658 cv .09041688922578099 spread 1.4094887308558006
 C41 s96 covariates + verdict: sigma_max(a) cv .002022155853666543, kappa cv .005147098612922159, spearman arms
     -.19208211143695014/-.12573313782991202/-.008064516129032258/-.21664222873900293/-.005498533724340176/.050219941348973604,
     verdict AMPLITUDE_LOTTERY, tail clause TAIL_INDETERMINATE, engine clause ENGINE_STABLE with 1 distinct kernel set
 C42 s96 tail ladder cv: p99 .007459945025920637, p99.9 .010655803116466073, p99.99 .02017463592490212,
     p99.999 .03154449842052123
 C43 s96b: 32/32 ok, identity residual <= 1e-6, e at ratio-argmax 1.075415514217859e-4-1.7380212578776977e-4,
     spearman rho-vs-e .6612903225806451, rho-vs-depth -.020894428152492667
 C48 s108 population: 2/2 rows ok, K1 bit-identical both, K2 anchor rel 0.0
 C49 s108 ownership: R_B frac 1e-2 .01953125 / .0009765625, R_A frac 1e-2 0.0 / 0.0, verdicts PROCEDURE_CARRIED
 C50 s108 shipped: compile vs ref rho 1e-2 .0275 / .0004 with allclose True on both, ref-vs-anchor rho 16.4779 / 1.2232
 C51 s108 mechanism: L2/14 spearman_x_rho -.6102859385420594, frac_pred .0185546875 vs meas .01953125; L2/18
     verdict MECH_NOT_CANCELLATION, frac_pred 0.0 vs meas .0009765625
"""
import gzip, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))

def _res_dir(here):
    """Resolve the results directory: lab/results in the repo, ../results when the script is run from
    the supplement package's kernels/ directory (packaged auditors must stay runnable, #52)."""
    for cand in (os.path.join(here, "results"), os.path.join(os.path.dirname(here), "results")):
        if os.path.isdir(cand):
            return cand
    return os.path.join(here, "results")


RES = _res_dir(HERE)
BASE = os.path.dirname(HERE)
PAPER = os.path.join(BASE, "paper") if os.path.isdir(os.path.join(BASE, "paper")) else BASE

checks = []
def ck(cid, ok, detail):
    checks.append({"id": cid, "ok": bool(ok), "detail": detail})

def load(name):
    for cand in (os.path.join(RES, name), os.path.join(RES, name + ".gz")):
        if os.path.exists(cand):
            if cand.endswith(".gz"):
                with gzip.open(cand, "rt") as f:
                    return json.load(f)
            return json.load(open(cand))
    return None


def _load_any(name):
    """Load a plain or gzipped artifact; raises when neither exists (row-streams may ship gzipped)."""
    for cand in (os.path.join(RES, name), os.path.join(RES, name + ".gz")):
        if os.path.exists(cand):
            if cand.endswith(".gz"):
                with gzip.open(cand, "rt") as f:
                    return json.load(f)
            with open(cand) as f:
                return json.load(f)
    raise FileNotFoundError(name)

# C01/C02
a8 = load("a8_compress_v2.json")
ck("C01", a8 is not None and len(a8["rows"]) == 48 and len(a8["controls"]) == 4,
   f"rows={len(a8['rows']) if a8 else None} controls={len(a8['controls']) if a8 else None}")
rf = a8["header"]["runtime_facts"] if a8 else {}
ck("C02", rf.get("cudnn_allow_tf32") is True and rf.get("matmul_allow_tf32") is False
   and rf.get("float32_precision") == "highest",
   f"{rf.get('cudnn_allow_tf32')}/{rf.get('matmul_allow_tf32')}/{rf.get('float32_precision')}")

a8b = load("a8b_analysis.json")
spans = a8b["within_class_spans_corrected"] if a8b else {}
models = a8b["models_corrected"] if a8b else {}
ck("C03", all(k in spans for k in ("W", "TF32", "GEMM_FP32")) and spans.get("TF32", {}).get("n") == 27
   and spans.get("GEMM_FP32", {}).get("n") == 18 and spans.get("W", {}).get("n") == 3,
   {k: spans.get(k, {}).get("n") for k in ("W", "TF32", "GEMM_FP32")})
m3 = models.get("class_derived", {}).get("spearman_logrho")
m2 = models.get("class_median", {}).get("spearman_logrho")
m4 = models.get("derived_only", {}).get("spearman_logrho")
ck("C04", abs(m3 - 0.9862) < 0.001 and abs(m2 - 0.5735) < 0.001 and abs(m4 - 0.8327) < 0.001,
   f"m3={m3} m2={m2} m4={m4}")
ck("C05", models.get("class_derived", {}).get("bulk_acc") == 1.0
   and abs(models.get("derived_only", {}).get("bulk_acc") - 0.7917) < 0.001,
   f"{models.get('class_derived', {}).get('bulk_acc')}/{models.get('derived_only', {}).get('bulk_acc')}")
ck("C06", abs(spans.get("W", {}).get("span_decades", 0) - 0.824) < 0.002
   and abs(spans.get("TF32", {}).get("span_decades", 0) - 0.924) < 0.002
   and abs(spans.get("GEMM_FP32", {}).get("span_decades", 0) - 3.042) < 0.002,
   {k: spans.get(k, {}).get("span_decades") for k in ("W", "TF32", "GEMM_FP32")})
cp = a8b.get("class_predictability", {}) if a8b else {}
ck("C07", abs(cp.get("loo_accuracy", 0) - 0.6042) < 0.001 and abs(cp.get("majority_baseline", 0) - 0.5625) < 0.001,
   f"{cp.get('loo_accuracy')}/{cp.get('majority_baseline')}")

# C08 dispatch boundary from raw rows
rows = a8["rows"] if a8 else []
def cls_of(row):
    s = " ".join(row.get("kernels") or []).lower()
    if "winograd" in s: return "W"
    if "tensorop_s1688" in s or "tf32" in s: return "TF32"
    if "simt_sgemm" in s or "implicit_convolve_sgemm" in s or "gemm" in s: return "GEMM_FP32"
    return "OTHER"
conv = {(r["C"], r["H"], r["k"], r["s"], r["d"]): cls_of(r) for r in rows if r["family"] == "conv2d"}
expect = {(64,7,3,1,1):"W",(64,13,3,1,1):"W",(64,21,3,1,1):"TF32",
          (256,7,3,1,1):"W",(256,13,3,1,1):"TF32",(256,21,3,1,1):"TF32",
          (16,7,3,1,1):"TF32",(16,13,3,1,1):"TF32",(16,21,3,1,1):"TF32"}
got = {str(k): conv.get(k) for k in expect}
ck("C08", got == {str(k): v for k, v in expect.items()}, {"expect": {str(k): v for k, v in expect.items()}, "got": got})

rho = {(r["C"], r["H"], r["k"], r["s"], r["d"]): r["rho_max"] for r in rows if r["family"] == "conv2d"}
j1 = rho[(64,21,3,1,1)] / rho[(64,13,3,1,1)]
j2 = rho[(256,13,3,1,1)] / rho[(256,7,3,1,1)]
ck("C09", j1 >= 20 and j2 >= 30, f"C64 H13->H21 {j1:.2f}x, C256 H7->H13 {j2:.2f}x")

ctr = {c["row"]: c for c in (a8["controls"] if a8 else [])}
def ctl(pred):
    for k, v in ctr.items():
        if pred(k):
            return v
    return None
im2 = ctl(lambda k: k.startswith("im2col"))
off = ctl(lambda k: "cudnnFalse" in k)
t1 = ctl(lambda k: "tf32False" in k and "_s1_" in k)
t2 = ctl(lambda k: "tf32False" in k and "_s2_" in k)
ck("C10", im2 and abs(im2["rho_max"] - 0.7548) < 0.001 and off and abs(off["rho_max"] - 0.4002) < 0.001
   and t1 and abs(t1["rho_max"] - 12.8068) < 0.01 and t2 and abs(t2["rho_max"] - 0.7512) < 0.001,
   {"im2col": im2 and im2["rho_max"], "cudnn_off": off and off["rho_max"],
    "tf32off_s1": t1 and t1["rho_max"], "tf32off_s2": t2 and t2["rho_max"]})

fp32 = [r for r in rows if cls_of(r) == "GEMM_FP32" and r["family"] == "conv2d"]
low = [r["rho_max"] for r in fp32 if r["k_eff"] == 16]
hi = [r["rho_max"] for r in rows if cls_of(r) == "GEMM_FP32" and r["k_eff"] == 8192]
ck("C11", low and min(low) > 0.004 and max(low) < 0.009 and hi and min(hi) >= 1.0,
   {"keff16": (min(low) if low else None, max(low) if low else None), "keff8192": hi})

f24 = load("f24_verify.json")
f1 = f24["f1"]["reduction"]["corrected_t_lt_gate"] if f24 else {}
ck("C12", f1 and f1["b1"]["n_finite"] == 6 and abs(f1["b1"]["median"] - 195.0484) < 0.01
   and f1["b2"]["n_finite"] == 7 and abs(f1["b2"]["median"] - 225.3871) < 0.01
   and f1["b4"]["n_finite"] == 15 and abs(f1["b4"]["median"] - 141.3787) < 0.01
   and f1["b8"]["n_finite"] == 21 and abs(f1["b8"]["median"] - 232.3333) < 0.01,
   {k: (f1.get(k, {}).get("n_finite"), f1.get(k, {}).get("median")) for k in ("b1","b2","b4","b8")})

f2 = f24["f2"] if f24 else {}
ck("C13", f2.get("hand_pass_default", {}).get("corrected") == 15
   and f2.get("ieee_pass_fp32ref", {}).get("corrected") == 80
   and f2.get("adm_ieee_fp32ref_rows", {}).get("corrected") == {"m3m2": 14, "w4": 2, "a2": 0}
   and f2.get("torch_cmp_failing", {}).get("n") == 5,
   {"hand": f2.get("hand_pass_default"), "ieee": f2.get("ieee_pass_fp32ref"),
    "adm": f2.get("adm_ieee_fp32ref_rows", {}).get("corrected"), "cmp": f2.get("torch_cmp_failing", {}).get("n")})

f3 = f24["f3"] if f24 else {}
ck("C14", f3.get("torch_fails_fp64_anchor") == 52
   and f3.get("gate_delivers_arm_failing_fp64") == {"def": 52, "ieee": 41}
   and f3.get("n_obsolete_shapes") == 4,
   {"torch_fail": f3.get("torch_fails_fp64_anchor"), "gate": f3.get("gate_delivers_arm_failing_fp64"),
    "obs": f3.get("n_obsolete_shapes")})

s6 = load("s6_dispatcher_audit.json")
red = s6["families"]["reduction"] if s6 else {}
cv = s6["families"]["conv"] if s6 else {}
ck("C16", red.get("worse1pct_count") == 3.0 and cv.get("worse1pct_count") == 0.0
   and abs(red.get("gate_mean_regret_total", 0) - 0.04511657284904949) < 1e-9
   and abs(cv.get("gate_mean_regret_total", 0) - 0.02006293937967579) < 1e-9
   and abs(red.get("best_b_mean_regret_total", 0) - 1.5277373759961599) < 1e-9
   and abs(cv.get("best_b_mean_regret_total", 0) - 2.146890348215717) < 1e-9,
   {"red_worse": red.get("worse1pct_count"), "conv_worse": cv.get("worse1pct_count"),
    "red_gate": red.get("gate_mean_regret_total"), "conv_gate": cv.get("gate_mean_regret_total")})
ck("C17", s6.get("V2_certificates_clean") is False and s6.get("S5_DECISION") == "SCOPE_CUT_S5",
   {"V2": s6.get("V2_certificates_clean"), "S5": s6.get("S5_DECISION")})

cf = load("cf43.json")
ck("C18", cf["cf1_chain_order"]["order_dependent"] == 0 and cf["cf1_chain_order"]["rider_order_dependent"] == 0
   and cf["cf2_anchor_profile"]["profile_counts"] == {"mixed": 48, "flat": 4, "nested": 4}
   and abs(cf["cf2_anchor_profile"]["iee64_exact_agreement"] - 0.3929) < 0.0005
   and cf["cf2_anchor_profile"]["arm_level_patterns"]["001"] == 165,
   {"cf1": cf["cf1_chain_order"]["order_dependent"], "profiles": cf["cf2_anchor_profile"]["profile_counts"],
    "001": cf["cf2_anchor_profile"]["arm_level_patterns"]["001"]})
ck("C19", cf["cf3_cross_family"]["pass_flips"] == 169 and cf["cf3_cross_family"]["arms_compared"] == 528
   and abs(cf["cf3_cross_family"]["flip_frac"] - 0.3201) < 0.0005,
   {"flips": cf["cf3_cross_family"]["pass_flips"], "arms": cf["cf3_cross_family"]["arms_compared"]})

cu = load("cushion_screen.json")
ck("C20", abs(cu["train"]["auc_x1"] - 0.7429) < 0.0005 and abs(cu["train"]["c_star"] - 0.4578) < 0.0005
   and abs(cu["rider"]["auc"] - 0.1211) < 0.0005 and cu["rider_below_c"]["n"] == 10
   and cu["rider_below_c"]["flip_frac"] == 0.0,
   {"train_auc": cu["train"]["auc_x1"], "c": cu["train"]["c_star"], "rider": cu["rider"]["auc"]})

b5 = _load_any("b5_stage1.json")
ck("C21", abs(b5.get("primary_share_T2_pooled", -1) - 0.1494) < 0.001, b5.get("primary_share_T2_pooled"))

b2d = _load_any("b2_direction_audit.json")
b4d = _load_any("b4_direction_audit.json")
b2_flips = b2d.get("partA", {}).get("n_label_flips")
b2_n = b2d.get("partA", {}).get("n_summaries")
b4_flips = b4d.get("n_label_flips")
b4_n = b4d.get("n_heldout")
ck("C22", b2_flips == 0 and b2_n == 350 and b4_flips == 0 and b4_n == 816,
   {"b2": (b2_flips, b2_n), "b4": (b4_flips, b4_n),
    "b4_auc": b4d.get("legacy", {}).get("pooled_auc"), "b4_acc": b4d.get("legacy", {}).get("accepted_n")})

s50 = load("s50_prose_audit.json")
s50_ok = sum(1 for c in s50.get("checks", []) if c.get("ok"))
# C23 counts s50's own checks; after DE1paW the s50 set grew (F1 recompute + F1b provenance of the
# stored-field bug), so the bar is "s50 reports zero failures", not a frozen 18.
ck("C23", s50_ok == len(s50.get("checks", [])) and s50_ok >= 18,
   {"pass": s50_ok, "n": len(s50.get("checks", []))})

prov = load("s48_provenance.json")
ck("C24", prov.get("n_cited") == prov.get("n_found") and prov.get("missing") == []
   and prov.get("n_cited") >= 25,
   {"cited": prov.get("n_cited"), "found": prov.get("n_found"), "missing": prov.get("missing")})

pop = load("pop_confirm.json").get("summary", {})
ck("C25", abs(pop.get("P90_fresh_pct", 0) - 15.3584) < 0.01 and abs(pop.get("T_pct", 0) - 16.1783) < 0.01
   and pop.get("P90_fresh_pct", 99) <= pop.get("T_pct", 0) and pop.get("n_eval") == 420
   and pop.get("verdict") == "PASS",
   {"p90": pop.get("P90_fresh_pct"), "T": pop.get("T_pct"), "cells": pop.get("n_eval")})

e2 = load("e2_v3_err.json")
e2_rows = e2.get("rows", [])
e2_n = sum(len(v.get("max_abs_err", [])) for r in e2_rows for v in r.get("variants", {}).values())
ck("C26", e2_n == 1260, e2_n)

b2c = os.path.join(RES, "b2_failure_corpus.jsonl")
if os.path.exists(b2c):
    n_rows = sum(1 for _ in open(b2c))
elif os.path.exists(b2c + ".gz"):
    import gzip as _gz
    n_rows = sum(1 for _ in _gz.open(b2c + ".gz", "rt"))
else:
    n_rows = None
ck("C27", n_rows == 180, n_rows)

a7b = load("a7b_precision_policy.json")
q1 = a7b.get("Q1", {})
ratios = q1.get("ratios", {})
ck("C28", q1.get("pass") is True and abs(ratios.get("s2", 1) - 0.0015664) < 1e-5
   and abs(ratios.get("d2", 1) - 0.0021595) < 1e-5 and abs(ratios.get("k1", 1) - 0.0006696) < 1e-5,
   {"pass": q1.get("pass"), "ratios": ratios})

r45 = load("r45_rider_adm.json")
ck("C15", r45["per_ref"]["def"]["a2_change_both_arms"] == 21
   and r45["per_ref"]["def"]["a2_change_to_none"] == 31
   and r45["per_ref"]["def"]["a2_change_from_none"] == 0
   and r45["per_ref"]["def"]["a2_delivered_change_def64"] == 52
   and r45["per_ref"]["ieee"]["no_delivery"] == 24 and r45["per_ref"]["64"]["no_delivery"] == 31
   and r45["torch_pass_own_check"] == {"def": 56, "ieee": 21, "64": 4},
   {"def": {k: r45["per_ref"]["def"][k] for k in ("a2_change_both_arms", "a2_change_to_none",
                                                  "a2_change_from_none", "a2_delivered_change_def64")},
    "ieee_nodel": r45["per_ref"]["ieee"]["no_delivery"], "64_nodel": r45["per_ref"]["64"]["no_delivery"],
    "torch": r45["torch_pass_own_check"]})

r1s = load("r1_conv_gated_summary.json")
by_tag = r1s.get("by_tag", {})
ck("C29", r1s.get("n_rows") == 56 and sum(t.get("gate_beats_torch", 0) for t in by_tag.values()) >= 2,
   {"n_rows": r1s.get("n_rows"), "by_tag": {k: v.get("gate_beats_torch") for k, v in by_tag.items()}})

e2b = load("e2b_floor.json")
an = e2b.get("analysis", {})
ck("C31", abs(an.get("two_launch_us", 0) - 7.904) < 0.01
   and abs(e2b["meta"]["launch_floor_one_ms"]["med_ms"] * 1000 - 5.856) < 0.01
   and abs(an.get("spearman_r_vs_Mstar", [None])[0] + 0.6429) < 0.001
   and an.get("falsifier_verdict") == "NON-MONOTONE",
   {"two_us": an.get("two_launch_us"), "one_us": e2b["meta"]["launch_floor_one_ms"]["med_ms"] * 1000,
    "spearman": an.get("spearman_r_vs_Mstar"), "verdict": an.get("falsifier_verdict")})

want = ["s2_reference_axis.md", "s3_atlas.md", "s3b_conv_anchor.md", "s4_certificates.md",
        "s5_tolerance.md", "s6_selector.md", "s7_threats.md", "s8_reproduction.md"]
present = {}
for f in want:
    p = os.path.join(PAPER, "sections", f)
    txt = open(p).read() if os.path.exists(p) else ""
    present[f] = bool(txt) and ("results/" in txt)
ck("C30", all(present.values()), present)

# --- §4.5e / §4.5f extension (declared extension of the fixed list; adds s94 and s95 to the audited set) ---
s94rows = []
try:
    s94rows = [json.loads(l) for l in open(os.path.join(RES, "s94_nonconv_decomp.jsonl"))]
except Exception:
    pass
s94 = load("s94_analysis.json")
ck("C32", len(s94rows) == 5 and all(r.get("status") == "ok" for r in s94rows)
   and [r["tid"] for r in s94rows] == ["T1", "T2", "T3", "T4", "T5"]
   and all(r["M"] == r["N"] and r["M"] in (1024, 512) for r in s94rows),
   {"n": len(s94rows), "tids": [r.get("tid") for r in s94rows], "status": sorted({r.get("status") for r in s94rows})})

pool = (s94 or {}).get("pooled", {})
stab = (s94 or {}).get("share_stability", {})
sh = {r["tid"]: r.get("share_ref_induced") for r in ((s94 or {}).get("table") or [])}
ck("C33", pool.get("n_fail") == 300 and pool.get("n_ref_induced") == 168 and pool.get("n_cand_induced") == 132
   and abs(pool.get("share_ref_induced", 0) - 0.56) < 1e-9
   and abs(sh.get("T3", 0) - 0.6) < 1e-9 and abs(sh.get("T4", 0) - 0.562015503875969) < 1e-9
   and abs(sh.get("T5", 0) - 0.5405405405405406) < 1e-9
   and abs(stab.get("spread", 0) - 0.05945945945945941) < 1e-9,
   {"pooled": pool, "per_tuple": sh, "spread": stab.get("spread")})

df = (s94 or {}).get("depth_fit", {})
rho = {r["tid"]: r.get("rho_ref_1e4") for r in ((s94 or {}).get("table") or [])}
ck("C34", abs(rho.get("T1", 0) - 0.5798813114391407) < 1e-9 and abs(rho.get("T2", 0) - 0.5585034175067228) < 1e-9
   and abs(rho.get("T3", 0) - 1.2346484213295181) < 1e-9 and abs(rho.get("T4", 0) - 2.449441385699855) < 1e-9
   and abs(rho.get("T5", 0) - 1.6524742737102525) < 1e-9
   and abs(df.get("slope", 0) - 0.7380331925187715) < 1e-9
   and abs(df.get("max_log_resid", 0) - 0.321516695619999) < 1e-9
   and df.get("crossing_bracket", {}).get("below") == [1024, 2048]
   and df.get("crossing_bracket", {}).get("above") == [4096, 8192],
   {"rho": rho, "slope": df.get("slope"), "resid": df.get("max_log_resid")})

rat = {r["tid"]: r.get("ratio_cand_over_ref") for r in ((s94 or {}).get("table") or [])}
ck("C35", all(r.get("C3_compile_bit_identical_to_A") is True for r in s94rows)
   and all(abs(rat.get(t, 0) - v) < 1e-9 for t, v in
           [("T1", 1.009449598816413), ("T2", 1.1080781174169432), ("T3", 0.9561703800873281),
            ("T4", 0.9858347803504247), ("T5", 1.148717664180466)]),
   {"c3_bit_identical": sorted({r.get("C3_compile_bit_identical_to_A") for r in s94rows}), "ratios": rat})

s95rows = []
try:
    s95rows = [json.loads(l) for l in open(os.path.join(RES, "s95_draw_repeat.jsonl"))]
except Exception:
    pass
s95 = load("s95_analysis.json") or {}
ck("C36", len(s95rows) == 33 and all(r.get("status") == "ok" for r in s95rows)
   and s95.get("instrument_pass") is True
   and all(s95.get("instrument", {}).values()),
   {"n": len(s95rows), "status": sorted({r.get("status") for r in s95rows}), "instrument": s95.get("instrument")})

dip = s95.get("dip", {})
pk = s95.get("per_k", {})
ck("C37", dip.get("n_draws_with_dip") == 1 and dip.get("n_draws") == 8 and dip.get("branch") == "FLOOR_NOISE"
   and abs(dip.get("mean_1024", 0) - 0.5042261951375578) < 1e-9
   and abs(dip.get("mean_2048", 0) - 0.6263878155027611) < 1e-9
   and s95.get("k2_fired") is True and s95.get("law_claim_permitted") is False
   and abs(pk.get("1024", {}).get("cv", 0) - 0.12702075847137118) < 1e-9
   and abs(pk.get("8192", {}).get("cv", 0) - 0.14305243267233525) < 1e-9,
   {"dip": dip, "cv_1024": pk.get("1024", {}).get("cv"), "cv_8192": pk.get("8192", {}).get("cv"),
    "k2": s95.get("k2_fired")})

sl = s95.get("slope_descriptive", {})
shs = s95.get("share", {})
ck("C38", abs(sl.get("mean", 0) - 0.7713786472806671) < 1e-9 and abs(sl.get("sd", 0) - 0.08076628792149573) < 1e-9
   and abs(shs.get("mean", 0) - 0.5024039188035422) < 1e-9 and abs(shs.get("sd", 0) - 0.13447622897245026) < 1e-9
   and shs.get("min") == 0.25 and shs.get("max") == 0.75 and shs.get("n") == 16
   and s95.get("share_branch") == "SHARE_UNSTABLE"
   and all(r.get("frac_over_ref_ship") == 0.0 for r in s95rows),
   {"slope": (sl.get("mean"), sl.get("sd")), "share": shs, "branch": s95.get("share_branch")})

s96rows = []
try:
    s96rows = [json.loads(l) for l in open(os.path.join(RES, "s96_amplitude_mechanism.jsonl"))]
except Exception:
    pass
s96 = load("s96_analysis.json") or {}
s96b = load("s96b_analysis.json") or {}
ck("C39", len(s96rows) == 33 and len([r for r in s96rows[:-1] if r.get("status") == "ok"]) == 32
   and s96rows[-1].get("rho_ref") == s96rows[0].get("rho_ref")
   and s96rows[-1].get("frac_over_ref_strict") == s96rows[0].get("frac_over_ref_strict")
   and s96.get("kills") == {"K1_instrument_ok": True, "K2_vacuous_fired": False, "K3_degenerate_fired": False},
   {"n": len(s96rows), "kills": s96.get("kills")})

rr = s96.get("rho_ref", {})
ck("C40", abs(rr.get("min", 0) - 1.0455225225704408) < 1e-9 and abs(rr.get("max", 0) - 1.4736522134189658) < 1e-9
   and abs(rr.get("cv", 0) - 0.09041688922578099) < 1e-9
   and abs(rr.get("spread_ratio", 0) - 1.4094887308558006) < 1e-9,
   {"rho": rr})

cv = s96.get("covariates", {})
sp = {k: v.get("spearman_vs_rho") for k, v in cv.items()}
ck("C41", abs(cv.get("sigma_max_a", {}).get("cv", 0) - 0.002022155853666543) < 1e-9
   and abs(cv.get("kappa_proxy", {}).get("cv", 0) - 0.005147098612922159) < 1e-9
   and abs(sp.get("sigma_max_a", 0) - -0.19208211143695014) < 1e-9
   and abs(sp.get("kappa_proxy", 0) - -0.12573313782991202) < 1e-9
   and abs(sp.get("median_abs_x", 0) - -0.008064516129032258) < 1e-9
   and abs(sp.get("max_abs_x", 0) - -0.21664222873900293) < 1e-9
   and abs(sp.get("max_abs_ref_err", 0) - -0.005498533724340176) < 1e-9
   and abs(sp.get("extremum_rel_to_median", 0) - 0.050219941348973604) < 1e-9
   and s96.get("verdict") == "AMPLITUDE_LOTTERY" and s96.get("tail_clause") == "TAIL_INDETERMINATE"
   and s96.get("engine_clause") == "ENGINE_STABLE" and s96.get("engine", {}).get("distinct_sets") == 1,
   {"spearman": sp, "verdict": s96.get("verdict"), "tail": s96.get("tail_clause"), "engine": s96.get("engine_clause")})

lad = s96.get("tail_ladder", {})
ck("C42", abs(lad.get("p99", {}).get("cv", 0) - 0.007459945025920637) < 1e-9
   and abs(lad.get("p99.9", {}).get("cv", 0) - 0.010655803116466073) < 1e-9
   and abs(lad.get("p99.99", {}).get("cv", 0) - 0.02017463592490212) < 1e-9
   and abs(lad.get("p99.999", {}).get("cv", 0) - 0.03154449842052123) < 1e-9,
   {"ladder": {k: v.get("cv") for k, v in lad.items()}})

ck("C43", s96b.get("n_ok") == 32 and s96b.get("resid_max", 1) < 1e-6
   and abs(s96b.get("e_at_ratio_argmax", {}).get("min", 0) - 1.075415514217859e-4) < 1e-12
   and abs(s96b.get("e_at_ratio_argmax", {}).get("max", 0) - 1.7380212578776977e-4) < 1e-12
   and abs(s96b.get("spearman", {}).get("rho_vs_e_at_ratio_argmax", 0) - 0.6612903225806451) < 1e-9
   and abs(s96b.get("spearman", {}).get("rho_vs_depth_at_ratio_argmax", 0) - -0.020894428152492667) < 1e-9,
   {"s96b": s96b.get("spearman"), "resid": s96b.get("resid_max"), "e": s96b.get("e_at_ratio_argmax")})

try:
    s99rows = [json.loads(l) for l in open(os.path.join(RES, "s99_conditioning.jsonl"))]
except Exception:
    s99rows = []
s99 = load("s99_analysis.json") or {}
s99sum = load("s99_conditioning_summary.json") or {}
s99inst = load("s99_instrument.json") or {}
ck("C44", len(s99rows) == 25 and sorted({r.get("level_kappa") for r in s99rows}) == [1.0, 10.0, 100.0, 1000.0, 10000.0]
   and s99inst.get("k1_bit_identical") is True
   and s99sum.get("k1_instrument_bit_identical") is True and s99sum.get("k2_engine_identical_across_levels") is True
   and s99sum.get("k3_construction_ok") is True and s99sum.get("k3_construction_worst_rel", 1) < 1e-3
   and s99.get("kills") == {"K1": True, "K2": True, "K3": True},
   {"n": len(s99rows), "kills": s99.get("kills"), "k3": s99sum.get("k3_construction_worst_rel")})

d99 = s99.get("descriptive", {})
means = [d99.get(str(l), {}).get("rho_mean") for l in (1.0, 10.0, 100.0, 1000.0, 10000.0)]
ck("C45", all(m is not None for m in means)
   and abs(means[0] - 1.341282429171134) < 1e-9 and abs(means[1] - 1.3011537045723727) < 1e-9
   and abs(means[2] - 1.356215893077395) < 1e-9 and abs(means[3] - 1.167914424888174) < 1e-9
   and abs(means[4] - 1.2084304498531317) < 1e-9
   and abs(max(means) / min(means) - 1.1613) < 5e-4
   and s99.get("verdict") == "P2_RETIRE_H1",
   {"level_means": means, "verdict": s99.get("verdict")})

ck("C46", abs(s99.get("primary", {}).get("spearman_kappa_rho", 0) - -0.35) < 1e-9
   and abs(s99.get("primary", {}).get("span", 0) - 1.9229777788469817) < 1e-9
   and abs(d99.get("within_level_cv_median", 0) - 0.09121691255868244) < 1e-9
   and abs(d99.get("within_level_cv_max", 0) - 0.2271600803478701) < 1e-9
   and abs(d99.get("rho_min", 0) - 1.0198520021760438) < 1e-9
   and abs(d99.get("rho_max", 0) - 1.9611527378971358) < 1e-9
   and abs(d99.get("spearman_median_absX_rho", 0) - 0.37615384615384617) < 1e-9
   and abs(d99.get("spearman_depth_rho", 0) - -0.10538461538461538) < 1e-9
   and abs(d99.get("spearman_sigma_max_b_rho", 0) - -0.3192307692307692) < 1e-9,
   {"primary": s99.get("primary"), "cv": d99.get("within_level_cv_median")})

ck("C47", all(v.get("frac_ship_mean") == 0.0 for v in d99.values() if isinstance(v, dict) and "frac_ship_mean" in v)
   and all(len(v.get("engine") or []) == 1 for v in d99.values() if isinstance(v, dict) and "engine" in v)
   and len({json.dumps(v.get("engine")) for v in d99.values() if isinstance(v, dict) and "engine" in v}) == 1
   and all(1e-6 <= v.get("frac_strict_mean", 0) <= 1e-5
           for v in d99.values() if isinstance(v, dict) and "frac_strict_mean" in v),
   {"engines": {k: v.get("engine") for k, v in d99.items() if isinstance(v, dict) and "engine" in v}})

try:
    s108rows = [json.loads(l) for l in open(os.path.join(RES, "s108_gate_decomp.jsonl"))]
except Exception:
    s108rows = []
s108 = load("s108_summary.json") or {}
ck("C48", len(s108rows) == 2 and all(r.get("status") == "ok" for r in s108rows)
   and all(r.get("k1_repeat_bit_identical") is True for r in s108rows)
   and all(r.get("k2_anchor_rel") == 0.0 for r in s108rows),
   {"n": len(s108rows), "k1": [r.get("k1_repeat_bit_identical") for r in s108rows]})

r14 = s108.get("rules", {}).get("14_Gemm_Divide_Sum_Scaling.py", {})
r18 = s108.get("rules", {}).get("18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp.py", {})
ck("C49", r14.get("ownership") == "BRANCH_PROCEDURE_CARRIED" and r18.get("ownership") == "BRANCH_PROCEDURE_CARRIED"
   and abs(r14.get("frac_B", 0) - 0.01953125) < 1e-12 and abs(r18.get("frac_B", 0) - 0.0009765625) < 1e-12
   and r14.get("frac_A") == 0.0 and r18.get("frac_A") == 0.0,
   {"14": [r14.get("ownership"), r14.get("frac_B"), r14.get("frac_A")],
    "18": [r18.get("ownership"), r18.get("frac_B"), r18.get("frac_A")]})

c14 = [r for r in s108rows if r.get("name", "").startswith("14_")]
c18 = [r for r in s108rows if r.get("name", "").startswith("18_")]
cb14 = (c14[0].get("C_B") or {}) if c14 else {}
cb18 = (c18[0].get("C_B") or {}) if c18 else {}
ck("C50", cb14.get("shipped_allclose") is True and cb18.get("shipped_allclose") is True
   and abs(cb14.get("vs_ref", {}).get("0.01", {}).get("rho_ew", 0) - 0.0275) < 5e-4
   and abs(cb18.get("vs_ref", {}).get("0.01", {}).get("rho_ew", 0) - 0.0004) < 5e-5
   and abs(r14.get("rho_ew_B", 0) - 16.477939423243555) < 1e-9
   and abs(r18.get("rho_ew_B", 0) - 1.2231567471632419) < 1e-9,
   {"vs_ref": [cb14.get("vs_ref", {}).get("0.01"), cb18.get("vs_ref", {}).get("0.01")],
    "rho_B": [r14.get("rho_ew_B"), r18.get("rho_ew_B")]})

ck("C51", abs(r14.get("spearman_x_rho", 0) - -0.6102859385420594) < 1e-9
   and abs(r14.get("frac_pred", 0) - 0.0185546875) < 1e-12
   and abs(r14.get("frac_meas", 0) - 0.01953125) < 1e-12
   and r14.get("mechanism") == "MECH_CANCELLATION"
   and r18.get("mechanism") == "MECH_NOT_CANCELLATION"
   and r18.get("frac_pred") == 0.0 and abs(r18.get("frac_meas", 0) - 0.0009765625) < 1e-12,
   {"14": [r14.get("spearman_x_rho"), r14.get("frac_pred"), r14.get("frac_meas"), r14.get("mechanism")],
    "18": [r18.get("frac_pred"), r18.get("frac_meas"), r18.get("mechanism")]})

n_pass = sum(1 for c in checks if c["ok"])
out = {"prereg": "docstring of lab/s51_prose_audit.py, frozen before run",
       "date": "2026-10-24", "n_checks": len(checks), "n_pass": n_pass,
       "n_fail": len(checks) - n_pass, "checks": checks}
with open(os.path.join(RES, "s51_prose_audit.json"), "w") as f:
    json.dump(out, f, indent=1)
for c in checks:
    if not c["ok"]:
        print("FAIL", c["id"], c["detail"])
print(f"PASS {n_pass}/{len(checks)}")
sys.exit(0 if n_pass == len(checks) else 1)
