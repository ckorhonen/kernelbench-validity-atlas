"""S55 review repair — recompute every number the DE1paW review says is wrong.

Conventions frozen HERE, before execution (analysis only; no threshold, bar or
outcome rule is changed, and no GPU measurement is made — records only):

R-F2  Regret-under-admission. The published field `paired_delta_pp_*` in
      lab/results/r45_rider_adm.json stores FRACTIONAL differences under a name
      that says percentage points, and its `_max` is a SIGNED max. Both are
      recomputed here as pp with max(|.|), over rows where BOTH regrets are
      defined, and separately over rows where A2 delivery becomes undefined.
      Verdict rule frozen: the published claim "|A2-A1| <= 0.025 pp" stands iff
      max|delta| <= 0.025 pp on every reference; otherwise it is withdrawn and
      replaced by the measured per-reference triple (mean, max|.|, n_defined).

R-F3  Composition. Ratios rho_end/max(rho_A,rho_B) and rho_end/(rho_A+rho_B)
      computed from lab/results/a9_compose.json verbatim; R1/R2/R3 pass counts
      re-read from the frozen a9_analysis.json (not recomputed — the frozen rule
      is the authority). The claim "bounded within 1.0-1.5x on five chains" is
      tested against BOTH readings: within-1.5x count, and within-factor-2
      count. Verdict: a prose claim naming 1.5 is supported only if >=5 chains
      lie in [1.0, 1.5].

R-F4  Compression magnitude. Median and max MULTIPLICATIVE prediction error per
      model from lab/results/a8_analysis.json preds (10**|log10 pred/meas|).
      Rank correlation is not magnitude calibration; both reported.

R-F5  Delivery-vs-compile denominator. d_i recomputed per row from the union set
      (r1_conv_gated.jsonl minus the 4 grid-garbled rows, plus
      r42_gridfix.jsonl) under BOTH denominators: oracle time (as shipped) and
      compile time (as §3.3 states). sd reported in pp under each.

R-F4b Crossing scope. Re-read a6d_crossing.json geometry_at_crossing to test
      whether "matmul inadmissible at K>~3.6k" holds beyond M=N=1024.

Output: lab/results/s55_review_repair.json. Hard-fails if a cited artifact is
absent.
"""
import json, math, os, statistics, time

HERE = os.path.dirname(os.path.abspath(__file__))

def _kb_res_dir(here):
    """Resolve the artifact directory: lab/results in the repo, ../results when run from the
    supplement package's kernels/ directory. Frozen packaging convention (#55)."""
    for cand in (os.path.join(here, "results"), os.path.join(os.path.dirname(here), "results")):
        if os.path.isdir(cand):
            return cand
    return os.path.join(here, "results")


def _kb_open(path):
    """Open a plain or gzipped artifact; row-streams may ship gzipped in the package."""
    if os.path.exists(path):
        return open(path)
    if os.path.exists(path + ".gz"):
        import gzip as _gz
        return _gz.open(path + ".gz", "rt")
    raise FileNotFoundError(path)
RES = _kb_res_dir(HERE)
OUT = os.path.join(RES, "s55_review_repair.json")
DATE = time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())
FAILS = []


def load(name, sub=""):
    p = os.path.join(RES, sub, name) if sub else os.path.join(RES, name)
    if not os.path.exists(p) and not os.path.exists(p + ".gz"):
        FAILS.append(p)
        return None
    if p.endswith(".json"):
        return json.load(_kb_open(p))
    return [json.loads(l) for l in _kb_open(p) if l.strip()]


def stats(xs):
    n = len(xs)
    if n == 0:
        return {"n": 0}
    m = sum(xs) / n
    sd = statistics.stdev(xs) if n > 1 else 0.0
    return {"n": n, "mean": m, "sd": sd, "max_abs": max(abs(x) for x in xs),
            "min": min(xs), "max": max(xs),
            "t": (m / (sd / math.sqrt(n))) if n > 1 and sd > 0 else None}


out = {"header": {"date": DATE, "conventions": "frozen in module docstring before run",
                  "purpose": "recompute DE1paW review findings F2/F3/F4/F4b/F5 from records only",
                  "no_gpu": True}}

# Pre-repair r45_rider_adm.json values (commit 751a096, before the F2 fix). Declared constants: the fix
# rewrote the artifact, so the defect has to be pinned here or the check would certify its own repair.
PRE_REPAIR_R45 = {
    "def": {"paired_delta_pp_mean": 0.0021, "paired_delta_pp_max": 0.0244, "paired_n": 56},
    "ieee": {"paired_delta_pp_mean": -0.0066, "paired_delta_pp_max": 0.0244, "paired_n": 32},
    "64": {"paired_delta_pp_mean": -0.0443, "paired_delta_pp_max": 0.0, "paired_n": 25},
}

# ---------------- R-F2 ----------------
r45 = load("r45_rider_adm.json")
f2 = {}
for ref in ("def", "ieee", "64"):
    rows = [r for r in r45["per_row"] if r["ref"] == ref]
    both = [(r["a2_reg_pct"] - r["a1_reg_pct"]) for r in rows
            if r.get("a1_reg_pct") is not None and r.get("a2_reg_pct") is not None
            and not r.get("a2_oracle_undef")]
    undef = [r for r in rows if r.get("a2_oracle_undef") or r.get("a2_no_delivery")]
    stored = r45["per_ref"][ref]
    f2[ref] = {
        "recomputed_pp": stats(both),
        "n_rows_total": len(rows),
        "n_a2_undefined_or_no_delivery": len(undef),
        "stored_fields": {"paired_delta_pp_mean": stored["paired_delta_pp_mean"],
                          "paired_delta_pp_max": stored["paired_delta_pp_max"],
                          "paired_n": stored["paired_n"]},
        "stored_over_recomputed_mean_ratio": (stored["paired_delta_pp_mean"] / stats(both)["mean"])
        if stats(both)["mean"] else None,
        # the pre-repair stored values, declared constants rather than re-read, because the fix to
        # r45_rider_adm.py rewrote the artifact this check was auditing. Without these the F2 evidence
        # would erase itself on the first rerun (ratio 0.0100 -> 1.0) and the finding would be uncheckable.
        "pre_repair_stored": PRE_REPAIR_R45[ref],
        "pre_repair_over_recomputed_mean_ratio": (PRE_REPAIR_R45[ref]["paired_delta_pp_mean"]
                                                 / stats(both)["mean"]) if stats(both)["mean"] else None,
    }
claim_holds = all(f2[r]["recomputed_pp"]["max_abs"] <= 0.025 for r in f2 if f2[r]["recomputed_pp"]["n"])
out["F2_regret_admission"] = {
    "published_claim": "paired |A2-A1| <= 0.025 pp on the defined overlap (s3b L68, s4 L36-39)",
    "claim_supported": claim_holds,
    "per_ref": f2,
    "diagnosis": "stored paired_delta_pp_* hold fractional differences (no x100) and _max is a "
                 "signed max, so a large negative change reads as 0.0 at the fp64 rung",
}

# ---------------- R-F3 ----------------
a9 = load("a9_compose.json")
a9a = load("a9_analysis.json")
rows3 = []
for r in a9["rows"]:
    rA, rB, rE = r["rho_A"], r["rho_B_alone"], r["rho_end"]
    rows3.append({"chain": r["chain"],
                  "ratio_worst": rE / max(rA, rB),
                  "ratio_additive": rE / (rA + rB),
                  "frac_A": r["frac_A"], "frac_B_alone": r["frac_B_alone"],
                  "frac_end": r["frac_end"],
                  "both_stages_admissible": (rA < 1 and rB < 1),
                  "end_inadmissible": rE > 1})
w15 = sum(1 for x in rows3 if 1.0 <= x["ratio_worst"] <= 1.5)
w20 = sum(1 for x in rows3 if x["ratio_worst"] <= 2.0)
out["F3_composition"] = {
    "chains": rows3,
    "n_within_1.0_1.5x": w15,
    "n_within_factor2": w20,
    "published_claim_1p5": "bounded by the worst stage within 1.0-1.5x on 5/6 chains (s1 L85, s5 L64,105)",
    "claim_1p5_supported": w15 >= 5,
    "claim_factor2_supported": w20 >= 5,
    "frozen_rule_counts_from_a9_analysis": {k: {"n_pass": a9a["rule_summary"][k]["n_pass"],
                                                "n_fail": a9a["rule_summary"][k]["n_fail"],
                                                "pass_chains": a9a["rule_summary"][k]["pass_chains"]}
                                            for k in ("R1", "R2", "R3")},
    "published_claim_R1": "the union rule fails 4/6 (s5 L59)",
    "R1_actual": "passes 4/6, fails 2 (c5, c6) -- the prose inverted its own artifact",
    "c5_control": next(x for x in rows3 if x["chain"].startswith("c5")),
    "c5_note": "two individually admissible stages (rho .069, .656) compose to rho_end 1.294 > 1 "
               "=> composition can fail even when both stages pass; frac stays 3.05e-5 (tail only)",
}

# ---------------- R-F4 ----------------
a8 = load("a8_analysis.json")
f4 = {}
for model, preds in a8["preds"].items():
    mult = [10 ** abs(p["pred_logrho"] - p["meas_logrho"]) for p in preds]
    f4[model] = {"n": len(mult), "median_mult_err": statistics.median(mult),
                 "max_mult_err": max(mult),
                 "spearman": a8["models"][model]["spearman_logrho"],
                 "bulk_acc": a8["models"][model]["bulk_acc"]}
# The corrected engine classification lives in a8b_analysis.json. The headline Spearman 0.986 belongs to
# THAT model, so its magnitude errors must come from the same artifact; reading only a8_analysis.json pairs
# a rank statistic from one model with residuals from another (DE1paW round-5 finding 4).
f4b = {}
a8b = load("a8b_analysis.json")
for model, preds in a8b["preds_corrected"].items():
    mult = [10 ** abs(p["pred_logrho"] - p["meas_logrho"]) for p in preds
            if p.get("pred_logrho") is not None and p.get("meas_logrho") is not None]
    if not mult:
        continue
    f4b[model] = {"n": len(mult), "median_mult_err": statistics.median(mult),
                  "max_mult_err": max(mult),
                  "spearman": a8b["models_corrected"][model]["spearman_logrho"],
                  "bulk_acc": a8b["models_corrected"][model]["bulk_acc"]}
out["F4_compression_magnitude"] = {
    "per_model": f4,
    "per_model_corrected_class": f4b,
    "model_match_note": "the .986 Spearman belongs to the corrected-class model; its magnitude errors are "
                        "1.251x median / 2.409x worst (a8b), NOT the 2.09x/440x older-model figures.",
    "published_claim": "rho_max compresses out-of-fold (.833 geometry-only / .986 with class)",
    "reading": "rank compression holds; magnitude calibration does NOT -- geometry-only median "
               "multiplicative error ~6.8x, max ~157x. Both must be reported together.",
    "bulk_clause": "bulk_acc tests only frac_over > 0.5, not the full admission rule",
}

# ---------------- R-F4b ----------------
a6d = load("a6d_crossing.json")
geo = a6d.get("geometry_at_crossing") or []
out["F4b_crossing_scope"] = {
    "geometry_at_crossing": geo,
    "rho_by_M": {str(g.get("M")): g.get("rho") for g in geo},
    "published_claim": "matmul fp32 cannot pass its own contract at K >~ 3.6k (s9 L38 area)",
    "reading": "crossing measured at M=N=1024 only; rho at K=3632 is .648 at M=256 and 2.777 at "
               "M=4096, so the threshold is geometry-dependent and must be stated per (M,N)",
}

# ---------------- R-F5 ----------------
r1 = load("r1_conv_gated.jsonl")
gf = load("r42_gridfix.jsonl")
WRONG = {("w4", 2, 3, 5), ("w4", 2, 3, 12), ("a2", 2, 4, 9), ("a2", 2, 4, 20)}
union = [r for r in r1 if (r["tag"], r["N"], r["H"], r["C"]) not in WRONG] + gf


def gate_ms(r):
    if r["H"] <= 14 and r["C"] <= 64:
        for prec in ("ieee", "tf32"):
            v = r["variants"].get("imp_B32x32[%s]" % prec)
            if v and v.get("pass_1e4"):
                return v["med_ms"]
    return r["variants"]["torch"]["med_ms"]


by_tag = {}
for r in union:
    gt = gate_ms(r)
    cmp_t = r["variants"]["torch_cmp"]["med_ms"]
    torch_t = r["variants"]["torch"]["med_ms"]
    passing = [v["med_ms"] for k, v in r["variants"].items() if k.startswith("imp_") and v.get("pass_1e4")]
    ora = min([torch_t] + passing)
    by_tag.setdefault(r["tag"], []).append({"oracle": (cmp_t - gt) / ora, "compile": (cmp_t - gt) / cmp_t,
                                            "cmp_failed": not r["variants"]["torch_cmp"].get("pass_1e4", False)})
f5 = {}
for tag, v in by_tag.items():
    f5[tag] = {"oracle_denom": stats([x["oracle"] for x in v]),
               "compile_denom": stats([x["compile"] for x in v]),
               "n_compile_failed_rows": sum(1 for x in v if x["cmp_failed"])}
out["F5_denominator"] = {
    "per_tag": f5,
    "published_claim": "d_i defined as percent of compile time; sd 0.29/0.23/0.20",
    "reading": "r42_analysis divides by ORACLE time, not compile. Under the stated compile "
               "denominator the means are smaller and sd in pp is ~10-30, not 0.29-0.20. The "
               "0.29/0.23/0.20 figures are oracle-denominator sd in FRACTION units.",
    "n_union_rows": len(union),
}

out["hard_fail_absent_artifacts"] = FAILS
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "F3_composition"}, indent=1)[:6000])
print("\n=== F3 chains ===")
for x in rows3:
    print("%-26s ratio_worst %.3f additive %.3f frac_end %.2e both_adm %s end_inadm %s" % (
        x["chain"], x["ratio_worst"], x["ratio_additive"], x["frac_end"],
        x["both_stages_admissible"], x["end_inadmissible"]))
print("\nABSENT:", FAILS)
