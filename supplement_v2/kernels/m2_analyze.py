"""M2 analysis: pre-registered A1-A4 on day-2 conv data (m2_conv_day2.json).

Acceptance criteria frozen in m2_conv_day2.py docstring BEFORE the GPU run; this
file only implements them. No thresholds refit here beyond A2's declared LOO-H
(H,C) refit. P5: report either way.

  A1 day-2 stability of the day-1-frozen rule (H<=14 & C<=64) OR (H<=7):
     per-point hand/torch label agreement >= 33/36 (labels from the same noise
     rule as day-1: relative advantage > noise, noise = mean IQR/2/torch, floor
     1%) AND pooled regret of the frozen rule on day-2 <= 2x 0.0093... day-1
     pooled was 0.0953 -> bar <= 0.1906. Hand arm for the rule: per-point best
     imp_* (day-1 convention: rule selects FAMILY not variant; oracle-hand).
  A2 day-2 (H,C) LOO-H refit: pooled regret of per-fold-refit 2-feature (H,C)
     rule, reported next to day-1 0.0953 / day-1 E-deployed 0.1729.
  A3 candidate (report-only): imp_B32x32 vs torch_cmp median rel advantage at
     H<=14 & C<=64, day 2; day-1 band was 25-48%.
  A4 mechanism: monotone T-response med(T4)<med(T2)<med(T1) on B32x32 at BOTH
     H=28 and H=20 => supported; else recorded not-supported.

Run: python3 m2_analyze.py -> results/m2_analysis.json + stdout verdict.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SRC = os.path.join(RES, "m2_conv_day2.json")
D1 = os.path.join(RES, "w3_conv_gemm.json")
OUT = os.path.join(RES, "m2_analysis.json")


def noise_label(t_torch, t_hand, iqr_t, iqr_h):
    noise = max(((iqr_t[1] - iqr_t[0]) + (iqr_h[1] - iqr_h[0])) / 2 / t_torch, 0.01)
    adv = (t_torch - t_hand) / t_torch
    return "hand" if adv > noise else ("torch" if adv < -noise else "tie")


def load(day=2):
    src = SRC if day == 2 else D1
    rows = json.load(open(src))["rows"]
    if day == 1:  # day-1 file: imp_* only, keys imp_B*
        pass
    out = []
    for r in rows:
        v = r["variants"]
        hands = {k: x for k, x in v.items() if k.startswith("imp_") and "med_ms" in x}
        t_bh, best = min((x["med_ms"], k) for k, x in hands.items())
        out.append(dict(N=r["N"], H=r["H"], C=r["C"], M=r["M"],
                        t_torch=v["torch"]["med_ms"], t_hand_best=t_bh, best_name=best,
                        iqr_t=v["torch"]["iqr_ms"], iqr_h=hands[best]["iqr_ms"],
                        variants=v))
    return out


def frozen_rule(p):
    return (p["H"] <= 14 and p["C"] <= 64) or p["H"] <= 7


def pooled_regret(pts, rule, hand_of):
    regs, worse = [], 0
    for p in pts:
        if rule(p):
            t = hand_of(p)
        else:
            t = p["t_torch"]
        regs.append((t - p["t_oracle"]) / p["t_oracle"])
        worse += t > p["t_torch"] * 1.01
    regs.sort()
    return dict(mean_regret=sum(regs) / len(regs), med_regret=regs[len(regs) // 2],
                n_worse=worse, n=len(regs))


def main():
    d2 = load(2)
    for p in d2:
        p["t_oracle"] = min(p["t_torch"], p["t_hand_best"])
    # A1
    agree = 0; labels = []
    for p in d2:
        lab = noise_label(p["t_torch"], p["t_hand_best"], p["iqr_t"], p["iqr_h"])
        pred = "hand" if frozen_rule(p) else "torch"
        ok = (lab == "hand") == (pred == "hand") or lab == "tie"
        # declared: tie counts as agreement with either (noise rule, day-1 convention)
        agree += ok
        labels.append(lab)
    imp_best = lambda p: p["t_hand_best"]
    a1_regret = pooled_regret(d2, frozen_rule, imp_best)
    a1 = dict(agreement=agree, bar_agreement=33, n=36,
              pooled_regret=a1_regret, bar_regret_le=2 * 0.0953,
              pass_=(agree >= 33 and a1_regret["mean_regret"] <= 2 * 0.0953))
    # A2: LOO-H refit of 2-feature (H,C) rule, grids from day-1 declared extension
    hs = sorted({p["H"] for p in d2})
    pairs = []
    Hs = [4, 7, 10, 14, 20, 28]; Cs = [16, 64, 256]
    for Htest in hs:
        tr = [p for p in d2 if p["H"] != Htest]
        best = None
        for h0 in Hs:
            for c0 in Cs + [512]:
                reg = 0.0; nhand = 0
                for p in tr:
                    rule = (p["H"] <= h0 and p["C"] <= c0) or (h0 <= 7 and p["H"] <= h0)
                    if rule:
                        t = p["t_hand_best"]; nhand += 1
                    else:
                        t = p["t_torch"]
                    reg += (t - p["t_oracle"]) / p["t_oracle"]
                key = (round(reg / len(tr), 9), -nhand)  # tie -> larger hand region? day-1: smaller; keep smaller
                key = (key[0], nhand)
                if best is None or key < best[0]:
                    best = (key, (h0, c0))
        (kh, (h0, c0)) = best
        te = [p for p in d2 if p["H"] == Htest]
        rule = lambda p, h0=h0, c0=c0: (p["H"] <= h0 and p["C"] <= c0) or (h0 <= 7 and p["H"] <= h0)
        pairs += [(p, rule(p)) for p in te]
    regs = []
    for p, r in pairs:
        t = p["t_hand_best"] if r else p["t_torch"]
        regs.append((t - p["t_oracle"]) / p["t_oracle"])
    regs.sort()
    a2 = dict(pooled_regret=sum(regs) / len(regs), med_regret=regs[len(regs) // 2],
              day1_ref=0.0953, day1_Edeployed_ref=0.1729)
    # A3
    adv = []
    for p in d2:
        if p["H"] <= 14 and p["C"] <= 64:
            tc = p["variants"].get("torch_cmp", {})
            if "med_ms" in tc and "med_ms" in p["variants"]["imp_B32x32"]:
                adv.append((tc["med_ms"] - p["variants"]["imp_B32x32"]["med_ms"]) / tc["med_ms"])
    adv.sort()
    a3 = dict(n=len(adv), median_rel_adv_vs_inductor=adv[len(adv) // 2] if adv else None,
              day1_band="25-48%", per_point=[round(a, 4) for a in adv])
    # A4
    a4 = {}
    for Hc in (20, 28):
        for base in ("B32x32", "B64x32"):
            ms = {}
            for T, nm in ((1, "imp_%s" % base), (2, "impw_%sT2" % base), (4, "impw_%sT4" % base)):
                ts = [p["variants"][nm]["med_ms"] for p in d2 if p["H"] == Hc]
                ms[T] = sorted(ts)[len(ts) // 2]
            mono = ms[4] < ms[2] < ms[1]
            a4["H%d_%s" % (Hc, base)] = dict(med={k: round(v, 5) for k, v in ms.items()}, monotone=mono)
    a4_supported = a4["H20_B32x32"]["monotone"] and a4["H28_B32x32"]["monotone"]
    res = dict(A1=a1, A2=a2, A3=a3, A4=dict(detail=a4, supported=a4_supported))
    json.dump(res, open(OUT, "w"), indent=1)
    print("A1 stability: agree %d/36, pooled regret %.4f (bar %.4f) -> %s" % (
        a1["agreement"], a1["pooled_regret"]["mean_regret"], 2 * 0.0953,
        "PASS" if a1["pass_"] else "FAIL (downgrade per wrapup rule)"))
    print("A1 worse-than-torch pts:", a1["pooled_regret"]["n_worse"])
    print("A2 refit pooled %.4f (day1 0.0953, E-deployed 0.1729) med %.4f" % (
        a2["pooled_regret"], a2["med_regret"]))
    print("A3 candidate vs inductor H<=14&C<=64: n=%d median %+.1f%%" % (
        a3["n"], 100 * (a3["median_rel_adv_vs_inductor"] or 0)))
    print("A4 T-response supported:", a4_supported)
    for k, v in a4.items():
        print("  ", k, v["med"], "mono" if v["monotone"] else "non-mono")
    print("A1 label disagreements (day2):")
    for p, lab in zip(d2, labels):
        pred = "hand" if frozen_rule(p) else "torch"
        if not ((lab == "hand") == (pred == "hand") or lab == "tie"):
            print("   N%d H%d C%d: label %s vs rule %s (torch %.4f best %s %.4f)" % (
                p["N"], p["H"], p["C"], lab, pred, p["t_torch"], p["best_name"], p["t_hand_best"]))


if __name__ == "__main__":
    main()
