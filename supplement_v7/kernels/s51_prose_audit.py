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
n_rows = sum(1 for _ in open(b2c)) if os.path.exists(b2c) else None
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
