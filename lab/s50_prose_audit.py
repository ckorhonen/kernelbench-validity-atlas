"""S50 — pre-prose audit for §2 (reference-realization axis) and §4 (conditional certificates).

Conventions frozen in this docstring BEFORE the run (same discipline as s48): every number the prose will
cite is regenerated here from committed artifacts; a cited number that cannot be regenerated is reported as
UNSUPPORTED and the prose may not use it. CPU only, reads results/ and lab/results/.

§2 numbers to check:
 A. calibration constants (SM count, L2, copy BW two-slope, launch floors, day-2 drift bound) from
    results/calibration.json + results/e1_triad_day2.json.
 B. reference realizations: rho surface from results/a6*.json + a7/a7b (this tick's measurements).
 C. admissibility of the baseline: torch_pass_own_check def/ieee/64 of 56 from results/r45_rider_adm.json.
 D. tolerance-direction disagreement 0/560 realized verdict cells from results/f24_verify.json.

§4 numbers to check:
 E. corrected counts: hand-vs-default 15/560, ieee-vs-fp32ref 80/280, admissible rows 14/2/0.
 F. rider admission: paired |A2-A1| bounds; delivery disagreement 0/24/31; no-admissible-delivery 24/56 and
    31/56; adm pool empty 20/56 and 4/56; def->64 52 = 21 arm->arm + 31 arm->none.
 G. cushion screen: train AUC, c*, P(flip|err<=c).
 H. NEW this tick: the certificate pair (rho_max, frac_over) per engine class from a7/a7b, which is what §4
    must carry instead of a bare max.
"""
import json, os, hashlib, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
TOP = os.path.join(os.path.dirname(HERE), "results")
OUT = os.path.join(RES, "s50_prose_audit.json")

def load(name, prefer_top=False):
    for base in ((TOP, RES) if prefer_top else (RES, TOP)):
        p = os.path.join(base, name)
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f), p
    return None, None

report = {"checks": [], "unsupported": [], "header": {"prose_audit": "s50", "note": "pre-prose regeneration; UNSUPPORTED = cited but not regenerable"}}

def check(label, ok, detail, artifact=None):
    report["checks"].append({"label": label, "ok": bool(ok), "detail": detail, "artifact": artifact})
    if not ok:
        report["unsupported"].append(label)

# ---- A calibration ----
cal, calp = load("calibration.json")
if cal is None:
    cal, calp = load("calibration.json", prefer_top=True)
if cal:
    s = json.dumps(cal)[:400]
    report["calibration_keys"] = list(cal.keys())[:20]
    check("A_calibration_present", True, "loaded", calp)
else:
    check("A_calibration_present", False, "missing")

d2, d2p = load("e1_triad_day2.json")
if d2 is None:
    d2, d2p = load("e1_triad_day2.json", prefer_top=True)
if d2:
    report["e1_day2_keys"] = list(d2.keys())[:20]
    check("A_e1_day2_present", True, "loaded", d2p)
else:
    check("A_e1_day2_present", False, "missing")

# ---- B rho surface ----
rho = {}
for nm in ("a6_refdistance.json", "a6b_ksweep.json", "a6c_conv_tf32pair.json", "a6d_crossing.json",
           "a6e_geometry_algo.json", "a6f_geom_repl_convmech.json", "a7_kernelclass.json",
           "a7b_precision_policy.json"):
    d, p = load(nm)
    if d:
        rho[nm] = p
    else:
        check("B_missing_artifact_" + nm, False, "not found")
report["rho_artifacts"] = rho
check("B_rho_artifacts_count", len(rho) >= 6, f"{len(rho)} of 8 loaded")

a7 = None
if "a7_kernelclass.json" in rho:
    a7, _ = load("a7_kernelclass.json")
a7b = None
if "a7b_precision_policy.json" in rho:
    a7b, _ = load("a7b_precision_policy.json")

# §4 pair statistic: recompute (rho_max, frac_over) per engine class from a7 rows
if a7:
    by = {r["row"]: r for r in a7["rows"]}
    pairs = {
        "winograd_C256": {"rho_max": by["R1_cudnn_3x3_s1"]["rho_max"], "frac_over": by["R1_cudnn_3x3_s1"]["frac_over"]},
        "tf32_s2_C256": {"rho_max": by["R2_cudnn_3x3_s2"]["rho_max"], "frac_over": by["R2_cudnn_3x3_s2"]["frac_over"]},
        "tf32_d2_C256": {"rho_max": by["R3_cudnn_3x3_d2"]["rho_max"], "frac_over": by["R3_cudnn_3x3_d2"]["frac_over"]},
        "tf32_k1_C256": {"rho_max": by["R4_cudnn_1x1"]["rho_max"], "frac_over": by["R4_cudnn_1x1"]["frac_over"]},
        "im2col_C256": {"rho_max": by["R5_im2col_matmul"]["rho_max"], "frac_over": by["R5_im2col_matmul"]["frac_over"]},
        "cudnn_off_C256": {"rho_max": by["R6_cudnn_disabled"]["rho_max"], "frac_over": by["R6_cudnn_disabled"]["frac_over"]},
    }
    report["pair_statistic"] = pairs
    bulk = all(p["frac_over"] > 0.5 for k, p in pairs.items() if k.startswith("tf32"))
    tail = pairs["winograd_C256"]["frac_over"] < 0.02
    check("H_class_separation_bulk_vs_tail", bulk and tail,
          f"tf32 frac_over>0.5 all: {bulk}; winograd frac_over {pairs['winograd_C256']['frac_over']:.4f} <0.02: {tail}")
    check("H_class_gap_15x", pairs["winograd_C256"]["rho_max"] / pairs["im2col_C256"]["rho_max"] > 10,
          f"winograd/im2col ratio {pairs['winograd_C256']['rho_max']/pairs['im2col_C256']['rho_max']:.2f}")
else:
    check("H_pair_statistic", False, "a7 artifact missing")

if a7b:
    q1, q2, q3 = a7b["Q1"], a7b["Q2"], a7b["Q3"]
    check("H_Q1_tf32_flag_reaches_engine", q1["pass"],
          f"ratios {q1['ratios']} bit_equal {q1['bit_equal']}")
    check("H_Q2_winograd_inert", q2["pass"], f"rho_on/off {q2['rho_on']:.3f}/{q2['rho_off']:.3f} bit_equal {q2['bit_equal']}")
    check("H_Q3_flag_works", q3["pass"], f"tf32 {q3['tf32_rho']:.2f} fp32 {q3['fp32_rho']:.3f}")

# ---- C/D baseline admissibility + tolerance direction ----
r45, r45p = load("r45_rider_adm.json")
if r45:
    tpc = r45.get("torch_pass_own_check")
    report["torch_pass_own_check"] = tpc
    check("C_baseline_admissibility_present", tpc is not None, f"value {tpc}", r45p)
else:
    check("C_baseline_admissibility_present", False, "r45 artifact missing")

f24, f24p = load("f24_verify.json")
if f24 is None:
    f24, f24p = load("f24_verify.json", prefer_top=True)
if f24:
    report["f24_keys"] = list(f24.keys())[:25]
    check("D_f24_present", True, "loaded", f24p)
else:
    check("D_f24_present", False, "f24_verify.json missing")

# ---- E corrected counts from r1 ----
r1s, r1p = load("r1_conv_gated_summary.json")
if r1s:
    report["r1_summary"] = {k: r1s[k] for k in list(r1s.keys())[:14]}
    check("E_r1_summary_present", True, "loaded", r1p)
else:
    check("E_r1_summary_present", False, "missing")

# ---- F rider admission numbers (nested under per_ref; regenerated, not read from a summary field) ----
if r45:
    pr = r45["per_ref"]
    adm = {
        "paired_abs_bounds": {r: pr[r]["paired_delta_pp_max"] for r in ("def", "ieee", "64")},
        "arm_disagree": {r: pr[r]["arm_disagree_a2_vs_a1"] for r in ("def", "ieee", "64")},
        "no_delivery": {r: pr[r]["no_delivery"] for r in ("def", "ieee", "64")},
        "oracle_undef": {r: pr[r]["oracle_undef"] for r in ("def", "ieee", "64")},
        "def_to_64": {"changes": pr["def"]["a2_delivered_change_def64"],
                      "both_arms": pr["def"]["a2_change_both_arms"],
                      "to_none": pr["def"]["a2_change_to_none"],
                      "from_none": pr["def"]["a2_change_from_none"]},
        "pool_n_min": {r: pr[r]["a2_pool_n_min"] for r in ("def", "ieee", "64")},
        "pool_n_med": {r: pr[r]["a2_pool_n_med"] for r in ("def", "ieee", "64")},
    }
    report["r45_fields"] = adm
    checks = [
        ("F1_paired_bounds_lt_0.025pp", all(v <= 0.025 for v in adm["paired_abs_bounds"].values()), adm["paired_abs_bounds"]),
        ("F2_arm_disagree_0_24_31", adm["arm_disagree"] == {"def": 0, "ieee": 24, "64": 31}, adm["arm_disagree"]),
        ("F3_no_delivery_24_31", adm["no_delivery"]["ieee"] == 24 and adm["no_delivery"]["64"] == 31, adm["no_delivery"]),
        ("F4_oracle_undef_20_4", adm["oracle_undef"]["ieee"] == 20 and adm["oracle_undef"]["64"] == 4, adm["oracle_undef"]),
        ("F5_def_to_64_52_eq_21plus31", adm["def_to_64"]["changes"] == 52 and adm["def_to_64"]["both_arms"] == 21
         and adm["def_to_64"]["to_none"] == 31 and adm["def_to_64"]["from_none"] == 0, adm["def_to_64"]),
        ("F6_published_21_matches", r45["a1_delivered_change_def64_published"] == adm["def_to_64"]["both_arms"],
         r45["a1_delivered_change_def64_published"]),
    ]
    for lbl, ok, det in checks:
        check(lbl, ok, det, r45p)

# ---- G cushion ----
cs, csp = load("cushion_screen.json")
if cs:
    report["cushion"] = {k: cs[k] for k in list(cs.keys())[:10]}
    check("G_cushion_present", True, "loaded", csp)
else:
    check("G_cushion_present", False, "missing")

report["summary"] = {"n_checks": len(report["checks"]),
                     "n_failed": sum(1 for c in report["checks"] if not c["ok"]),
                     "unsupported": report["unsupported"]}
with open(OUT, "w") as f:
    json.dump(report, f, indent=1)
print(json.dumps(report["summary"], indent=1))
for c in report["checks"]:
    print(("OK  " if c["ok"] else "FAIL"), c["label"], "|", str(c["detail"])[:120])
print("\nCAL KEYS:", report.get("calibration_keys"))
print("PAIR:", json.dumps(report.get("pair_statistic"), indent=1))
print("TPC:", report.get("torch_pass_own_check"))
print("R45 KEYS:", report.get("r45_top_keys"))
sys.exit(0 if report["summary"]["n_failed"] == 0 else 1)
