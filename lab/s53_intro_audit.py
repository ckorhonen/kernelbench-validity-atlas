"""S53 intro audit — regenerate every number cited in §1 from committed artifacts.

Conventions frozen in this docstring BEFORE the run (P3/P1): each check names its artifact and its field;
a check that cannot find its source is FAIL, not skipped. Exit nonzero on any FAIL. Fixed check list, no
post-hoc additions after the run.

This audit covers §1 only. §2-§8 numbers are audited by s50 (18/18), s51 (31/31) and s52 (14/14); §1 is
written last, against already-audited values, so this script re-derives the numbers §1 quotes in its own
right and asserts the section-level facts it depends on. Field paths were resolved against the committed
artifacts before the check list was frozen (the first draft read absent top-level keys; recorded because
this is the third time a field-path guess cost a run).

Checks:
 B01 calibration: sm_count 48, L2 25165824, launch single-shot 0.007456 ms, back-to-back 0.002940 ms
 B02 calibration: clock_lock.ok False, message names GPU 0F:01:00.0
 B03 a7 rows: 3x3 s1 rho 10.195 frac 0.0052; 3x3 s2 rho 251.692 frac 0.7817
 B04 a6d crossing: lo_row K 3608 rho<1, hi_row K 3632 rho>1 (measured bisection, not interpolated)
 B05 a8b: M3 0.9862 / M4 0.8327 / bulk 1.0 / 0.7917 / class LOO 0.6042 vs majority 0.5625
 B06 a8b dispatch_table: C64 H21/H13 >= 20x and C256 H13/H7 >= 30x (the 20-38x one-step jump)
 B07 r42_analysis by_tag d_i_pct_mean 11.8719/12.0834/10.2964 with t 2.4356/1.6439/1.6025
 B08 f24_verify f2: hand_pass_default corrected 15 (as-run 20), ieee_pass_fp32ref corrected 80 (as-run 95)
 B09 cf43: cf1 0 order-dependent of 36 conv rows and 0 of 168 rider cells; cf2 pattern 001 == 165 of 672
 B10 s48_prose_audit r1_gate_delivery: 56 rows, gate_delivers_torch 52
 B11 a9_analysis: 5 chains pass R2+R3, c6 measured/R3 in [2.2,2.5]
 B12 e1_triad.json vs e1_triad_day2.json: saturated GB/s (n>=2^27) delta <= 0.13% per rule
 B13 three archived full-stage reviews and the 2CmvAb triage naming t_b < t_g and v* = P/(t_g - t_b)
 B14 every results/ path cited in §1 exists under lab/results (no phantom citations)
 B15 §1 present and all of §2-§8 present (paper complete)
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
PAPER = os.path.join(HERE, "..", "paper")
ROOT = os.path.join(HERE, "..")

checks = []


def ck(cid, ok, detail):
    checks.append({"id": cid, "ok": bool(ok), "detail": detail})


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    return json.load(open(p))


cal = load("calibration.json")
lo = cal["launch_overhead"] if cal else {}
ck("B01", cal is not None and cal.get("sm_count") == 48 and cal.get("L2_cache_size") == 25165824
   and abs(lo.get("single_shot_med_ms", 0) - 0.007456) < 2e-5
   and abs(lo.get("back_to_back_ms", 0) - 0.002940) < 2e-5,
   f"sm={cal.get('sm_count') if cal else None} L2={cal.get('L2_cache_size') if cal else None} "
   f"launch={lo.get('single_shot_med_ms')}/{lo.get('back_to_back_ms')}")

cl = cal["clock_lock"] if cal else {}
ck("B02", cl.get("ok") is False and "0F:01:00.0" in str(cl.get("msg", "")),
   f"ok={cl.get('ok')} msg={str(cl.get('msg'))[:60]}")

a7 = load("a7_kernelclass.json")
r7 = {r["row"]: r for r in (a7["rows"] if a7 else [])}
ck("B03", abs(r7.get("R1_cudnn_3x3_s1", {}).get("rho_max", 0) - 10.195) < 1e-3
   and abs(r7.get("R1_cudnn_3x3_s1", {}).get("frac_over", 0) - 0.0052) < 1e-4
   and abs(r7.get("R2_cudnn_3x3_s2", {}).get("rho_max", 0) - 251.692) < 1e-2
   and abs(r7.get("R2_cudnn_3x3_s2", {}).get("frac_over", 0) - 0.7817) < 1e-4,
   f"W={r7.get('R1_cudnn_3x3_s1', {}).get('rho_max')}/{r7.get('R1_cudnn_3x3_s1', {}).get('frac_over')} "
   f"TF32={r7.get('R2_cudnn_3x3_s2', {}).get('rho_max')}/{r7.get('R2_cudnn_3x3_s2', {}).get('frac_over')}")

cr = load("a6d_crossing.json")
lo_row = cr.get("lo_row", {}) if cr else {}
hi_row = cr.get("hi_row", {}) if cr else {}
ck("B04", lo_row.get("K") == 3608 and hi_row.get("K") == 3632
   and lo_row.get("rho", 9) < 1.0 < hi_row.get("rho", 0),
   f"lo K={lo_row.get('K')} rho={lo_row.get('rho')} hi K={hi_row.get('K')} rho={hi_row.get('rho')}")

a8b = load("a8b_analysis.json")
m = a8b["models_corrected"] if a8b else {}
cp = a8b["class_predictability"] if a8b else {}
ck("B05", abs(m.get("class_derived", {}).get("spearman_logrho", 0) - 0.9862) < 1e-3
   and abs(m.get("derived_only", {}).get("spearman_logrho", 0) - 0.8327) < 1e-3
   and m.get("class_derived", {}).get("bulk_acc") == 1.0
   and abs(m.get("derived_only", {}).get("bulk_acc", 0) - 0.7917) < 1e-3
   and abs(cp.get("loo_accuracy", 0) - 0.6042) < 1e-3 and abs(cp.get("majority_baseline", 0) - 0.5625) < 1e-4,
   f"M3={m.get('class_derived', {}).get('spearman_logrho')} M4={m.get('derived_only', {}).get('spearman_logrho')} "
   f"bulk={m.get('class_derived', {}).get('bulk_acc')}/{m.get('derived_only', {}).get('bulk_acc')} "
   f"cls={cp.get('loo_accuracy')} maj={cp.get('majority_baseline')}")

disp = a8b.get("dispatch_table", {}) if a8b else {}
def rho_at(C, H):
    for k, v in disp.get("k3_s1_d1", {}).items():
        if k == f"C{C}_H{H}":
            return v[0]["rho"]
    return None
r64 = (rho_at(64, 21) / rho_at(64, 13)) if rho_at(64, 13) else None
r256 = (rho_at(256, 13) / rho_at(256, 7)) if rho_at(256, 7) else None
ck("B06", bool(r64) and bool(r256) and r64 >= 20 and r256 >= 30,
   f"ratio_C64={r64:.2f} ratio_C256={r256:.2f}" if (r64 and r256) else "missing")

r42 = load("r42_analysis.json")
bt = r42.get("by_tag", {}) if r42 else {}
dvals = [round(bt.get(t, {}).get("d_i_pct_mean", 0), 4) for t in ("m3m2", "w4", "a2")]
tvals = [round(bt.get(t, {}).get("d_i", {}).get("t", 0), 4) for t in ("m3m2", "w4", "a2")]
ck("B07", dvals == [11.8719, 12.0834, 10.2964] and tvals == [2.4356, 1.6439, 1.6025],
   f"d_i={dvals} t={tvals}")

f24 = load("f24_verify.json")
f2 = f24.get("f2", {}) if f24 else {}
ck("B08", f2.get("hand_pass_default", {}).get("corrected") == 15
   and f2.get("hand_pass_default", {}).get("r1_asrun") == 20
   and f2.get("ieee_pass_fp32ref", {}).get("corrected") == 80
   and f2.get("ieee_pass_fp32ref", {}).get("r1_asrun") == 95,
   f"default={f2.get('hand_pass_default')} ieee={f2.get('ieee_pass_fp32ref')}")

cf = load("cf43.json")
cf1 = cf.get("cf1_chain_order", {}) if cf else {}
cf2 = cf.get("cf2_anchor_profile", {}) if cf else {}
pat = cf2.get("arm_level_patterns", {})
ck("B09", cf1.get("order_dependent") == 0 and cf1.get("m3_conv_rows") == 36
   and cf1.get("rider_order_dependent") == 0 and cf1.get("rider_cells") == 168
   and pat.get("001") == 165 and sum(pat.values()) == 672,
   f"cf1={cf1.get('order_dependent')}/{cf1.get('m3_conv_rows')} "
   f"{cf1.get('rider_order_dependent')}/{cf1.get('rider_cells')} "
   f"cf2_001={pat.get('001')}/{sum(pat.values())}")

s48 = load("s48_prose_audit.json")
gd = s48.get("r1_gate_delivery", {}) if s48 else {}
ck("B10", gd.get("n_rows") == 56 and gd.get("gate_delivers_torch") == 52,
   f"n_rows={gd.get('n_rows')} torch={gd.get('gate_delivers_torch')} hand={gd.get('gate_delivers_hand')}")

ana = load("a9_analysis.json")
arows = {x["chain"]: x for x in (ana["rows"] if ana else [])}
within = sum(1 for x in arows.values() if x["R2"]["verdict"] == "PASS" and x["R3"]["verdict"] == "PASS")
c6 = arows.get("c6_mm_deep_then_mm", {})
ck("B11", within == 5 and 2.2 <= c6.get("measured_over_R3", 0) <= 2.5,
   f"within={within} c6_over_R3={c6.get('measured_over_R3')}")

d1 = load("e1_triad.json")
d2 = load("e1_triad_day2.json")


def sat(rows):
    m = {}
    for r in rows:
        if r["n"] >= 2 ** 27:
            m.setdefault(r["rule"], []).append(r["GBps"])
    return {k: sum(v) / len(v) for k, v in m.items()}


sa = sat(d1["rows"]) if d1 else {}
sb = sat(d2["rows"]) if d2 else {}
deltas = {k: abs(sb[k] - sa[k]) / sa[k] * 100 for k in sa if k in sb}
ck("B12", deltas and max(deltas.values()) <= 0.13,
   {k: round(v, 4) for k, v in deltas.items()})

rejs = [f for f in os.listdir(ROOT) if f.startswith("rejection_review_")]
tri = open(os.path.join(ROOT, "rejection_triage_2CmvAb.md")).read()
ck("B13", len(rejs) >= 3 and "t_b < t_g" in tri and "v* = P/(t_g - t_b)" in tri,
   f"reviews={len(rejs)} filter_named={'t_b < t_g' in tri}")

s1 = open(os.path.join(PAPER, "sections", "s1_introduction.md")).read()
cited = set(re.findall(r"results/([A-Za-z0-9_\.\-]+\.jsonl?)", s1))
missing = sorted(c for c in cited if not os.path.exists(os.path.join(RES, c)))
ck("B14", len(cited) > 0 and not missing, f"cited={len(cited)} missing={missing}")

want = ["s1_introduction.md", "s2_reference_axis.md", "s3_atlas.md", "s3b_conv_anchor.md",
        "s4_certificates.md", "s5_tolerance.md", "s6_selector.md", "s7_threats.md", "s8_reproduction.md"]
absent = [w for w in want if not os.path.exists(os.path.join(PAPER, "sections", w))]
ck("B15", not absent, f"present={len(want)-len(absent)}/{len(want)} missing={absent}")

npass = sum(1 for c in checks if c["ok"])
out = {"date": "2026-10-24", "conventions": "docstring", "n_checks": len(checks), "n_pass": npass,
       "all_pass": npass == len(checks), "checks": checks}
with open(os.path.join(RES, "s53_intro_audit.json"), "w") as f:
    json.dump(out, f, indent=1)
for c in checks:
    print(("PASS" if c["ok"] else "FAIL"), c["id"], c["detail"])
print(f"{npass}/{len(checks)} pass")
sys.exit(0 if npass == len(checks) else 1)
