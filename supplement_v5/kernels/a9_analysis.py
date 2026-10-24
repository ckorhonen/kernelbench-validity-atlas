"""A9 analysis — frozen rules from lab/a9_compose_prereg.md @ 53ca881 applied as written.

Conventions frozen in THIS docstring before execution (analysis conventions only; no rule or bar is
changed post-hoc):
- R1 predicts frac_end; R2 and R3 predict rho_end. Comparing a frac prediction to a rho measurement
  would be a units error, so each rule is scored only in its own currency.
- Ratio conventions: log10(pred/meas). If meas == 0 and pred == 0 the chain counts as PASS (both say
  "nothing exceeds budget"); if meas == 0 and pred > 0 it counts as FAIL.
- ADEQUATE = passes on >= 5 of 6 chains AND on >= 1 of the two clean-A controls (c4, c5), per prereg.
- Discrimination note reported separately from the verdict: the frozen bar is factor 2, so a rule set
  where every rule passes has not separated the hypotheses. That is reported as a test-design
  limitation, not as a rule change.
"""
import json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "results", "a9_compose.json")
OUT = os.path.join(HERE, "results", "a9_analysis.json")
CONTROLS = {"c4_cleanA_GEMM_then_TF32", "c5_cleanA_mm_then_mm"}
BAR = 0.301  # factor 2


def score(pred, meas):
    if meas == 0.0:
        return ("PASS", 0.0) if pred == 0.0 else ("FAIL", float("inf"))
    if pred <= 0.0:
        return "FAIL", float("-inf")
    lr = math.log10(pred / meas)
    return ("PASS" if abs(lr) <= BAR else "FAIL"), lr


def main():
    d = json.load(open(SRC))
    rows = d["rows"]
    out_rows = []
    for r in rows:
        rho_a, frac_a = r["rho_A"], r["frac_A"]
        rho_b, frac_b = r["rho_B_alone"], r["frac_B_alone"]
        G = r["G1"]
        r1_pred = 1.0 - (1.0 - frac_a) * (1.0 - frac_b)
        r2_pred = max(rho_a, rho_b)
        r3_pred = rho_b + G * rho_a
        r1_v, r1_lr = score(r1_pred, r["frac_end"])
        r2_v, r2_lr = score(r2_pred, r["rho_end"])
        r3_v, r3_lr = score(r3_pred, r["rho_end"])
        out_rows.append({
            "chain": r["chain"], "is_control": r["chain"] in CONTROLS,
            "rho_A": rho_a, "frac_A": frac_a, "rho_B_alone": rho_b, "frac_B_alone": frac_b,
            "G": G, "G_nonlinear": r["G_nonlinear"],
            "rho_end_meas": r["rho_end"], "frac_end_meas": r["frac_end"],
            "R1": {"pred_frac": r1_pred, "verdict": r1_v, "log10_ratio": r1_lr},
            "R2": {"pred_rho": r2_pred, "verdict": r2_v, "log10_ratio": r2_lr},
            "R3": {"pred_rho": r3_pred, "verdict": r3_v, "log10_ratio": r3_lr},
            "measured_over_R3": (r["rho_end"] / r3_pred) if r3_pred > 0 else None,
            "measured_over_R2": (r["rho_end"] / r2_pred) if r2_pred > 0 else None,
            "amplification_term": G * rho_a,
            "amplification_share_of_B": (G * rho_a / rho_b) if rho_b > 0 else None,
            "frac_end_minus_fracB": r["frac_end"] - frac_b,
            "frac_end_over_fracB": (r["frac_end"] / frac_b) if frac_b > 0 else None,
        })

    summary = {}
    for rule in ("R1", "R2", "R3"):
        passes = [x["chain"] for x in out_rows if x[rule]["verdict"] == "PASS"]
        fails = [x["chain"] for x in out_rows if x[rule]["verdict"] == "FAIL"]
        ctrl_pass = any(x["is_control"] and x[rule]["verdict"] == "PASS" for x in out_rows)
        adequate = len(passes) >= 5 and ctrl_pass
        summary[rule] = {"n_pass": len(passes), "n_fail": len(fails), "pass_chains": passes,
                         "fail_chains": fails, "control_pass": ctrl_pass, "adequate": adequate}

    any_adequate = any(summary[r]["adequate"] for r in summary)
    verdict = "COMPOSITION_PREDICTIVE" if any_adequate else "COMPOSITION_NOT_PREDICTIVE"
    # Discrimination: how many rules cleared the bar?
    n_adequate = sum(1 for r in summary if summary[r]["adequate"])
    discriminating = n_adequate < 3

    out = {"header": {"prereg": d["header"]["prereg"], "src": "results/a9_compose.json",
                      "bar_log10": BAR, "conventions": "see docstring"},
           "rows": out_rows, "rule_summary": summary,
           "verdict": verdict, "n_rules_adequate": n_adequate,
           "rules_discriminating": discriminating,
           "declared_limitation": (
               "Two of the three frozen rules clear the factor-2 bar on 5/6 chains, so the test does not "
               "separate no-amplification (R2) from additive amplification (R3) on rho: their predictions "
               "differ by only the G*rho_A term, which is small on 5 of 6 chains (max share of B 0.53). "
               "Independence (R1) IS separated — it fails 4/6. The single chain failing R2/R3 is c6 "
               "(matmul M1024 K8192 -> matmul), where the measured end-to-end rho is 2.34x the R3 "
               "prediction and 2.47x the R2 prediction. Reported as a test-design limitation plus one "
               "declared failing chain, not as a rule change.")}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)

    for x in out_rows:
        print(f"{x['chain']:28s} ctrl={x['is_control']!s:5s} "
              f"R1 {x['R1']['verdict']:4s}({x['R1']['log10_ratio']:+.3f}) "
              f"R2 {x['R2']['verdict']:4s}({x['R2']['log10_ratio']:+.3f}) "
              f"R3 {x['R3']['verdict']:4s}({x['R3']['log10_ratio']:+.3f}) "
              f"meas/R3={x['measured_over_R3']:.2f} dfrac={x['frac_end_minus_fracB']:+.4f}")
    print("verdict:", verdict, "| rules adequate:", n_adequate, "| discriminating:", discriminating)
    for r in summary:
        print(" ", r, summary[r]["n_pass"], "/6 adequate=", summary[r]["adequate"],
              "fails:", summary[r]["fail_chains"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
