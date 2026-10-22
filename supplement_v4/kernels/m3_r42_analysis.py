"""Analysis for the contract-enforced m3 rerun (m3_r42.jsonl) -- rejection O4hnCx R2'.

Mirrors the m3_summary verdict fields (same definitions, same decision rule as the
m3 prereg: gate WINS a family iff mean regret_total(gate) <= min_b mean
regret_total(autotune) - pooled SE; parity b* = smallest b with mean
regret_total(autotune) <= gate + pooled SE), computed on the CORRECTED basis:

  oracle   = min med12 over {torch} U passing hand arms   (r42_analysis convention)
  arms     selected only over admissible (passing) candidates, validate-then-time
  gate     published structural rule, delivered arm = first passing precision else torch

Also reported: admissibility population per family (rows with >=1 passing hand arm,
mean passing arms/row, torch-only rows), arm choice counts, and breakeven visits
v* = probes_b / (t_b - t_gate) on rows with t_b > t_gate, median over
finite-crossover rows -- the published convention.

Output: results/m3_r42_analysis.json (+ stdout summary).
"""
import json, os, statistics, time

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "results", "m3_r42.jsonl")
OUT = os.path.join(HERE, "results", "m3_r42_analysis.json")
BUDGETS = [1, 2, 4, 8]


def se(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5 / n ** 0.5


def arm_total(r, key):
    a = r["arms"][key]
    return a.get("regret_total", a["regret"])


def analyze(rows):
    n = len(rows)
    adm_rows = 0
    hand_pass_total = 0
    torch_only = 0
    choices = {}
    breakeven = {b: [] for b in BUDGETS}
    for r in rows:
        adm = r["adm"]
        npass = sum(1 for k, v in adm.items() if k != "torch" and v["passed"])
        hand_pass_total += npass
        adm_rows += 1 if npass else 0
        torch_only += 1 if npass == 0 else 0
        for k, a in r["arms"].items():
            ch = str(a["choice"]).split("|")[0]
            choices[k + ":" + ch] = choices.get(k + ":" + ch, 0) + 1
        t_g = r["arms"]["gate"]["t"]
        for b in BUDGETS:
            a = r["arms"]["b%d" % b]
            if a["t"] > t_g and a["probes"] > 0:
                breakeven[b].append(a["probes"] / (a["t"] - t_g))
    diffs = {}
    for b in [0] + BUDGETS:
        key = "b%d" % b
        d = [arm_total(r, key) - arm_total(r, "gate") for r in rows]
        diffs[key] = dict(mean=sum(d) / n, se=se(d), n=n)
    gate_mean = sum(arm_total(r, "gate") for r in rows) / n
    arm_means = {("b%d" % b) if b else "b0": sum(arm_total(r, "b%d" % b if b else "b0") for r in rows) / n for b in [0] + BUDGETS}
    min_b = min(BUDGETS, key=lambda b: arm_means["b%d" % b])
    min_mean = arm_means["b%d" % min_b]
    pooled = se([arm_total(r, "b%d" % min_b) - arm_total(r, "gate") for r in rows])
    gate_wins = bool(gate_mean <= min_mean - pooled)
    parity = None
    for b in BUDGETS:
        d = [arm_total(r, "b%d" % b) - arm_total(r, "gate") for r in rows]
        if arm_means["b%d" % b] <= gate_mean + se(d):
            parity = b
            break
    be = {}
    for b in BUDGETS:
        xs = breakeven[b]
        be["b%d" % b] = dict(median=(statistics.median(xs) if xs else None), n_finite=len(xs), n=n)
    return dict(n=n, rows_with_passing_hand=adm_rows, torch_only_rows=torch_only,
                mean_passing_hand_arms=hand_pass_total / n,
                gate_regret_total_mean=gate_mean, arm_regret_total_mean=arm_means,
                gate_vs_arm_tot_diff=diffs, min_b=min_b, min_b_mean=min_mean,
                pooled_se=pooled, gate_wins_prereg=gate_wins, parity_b=parity,
                breakeven_visits=be, choice_counts=choices)


def main():
    rows = [json.loads(l) for l in open(IN) if l.strip()]
    assert len(rows) == 96, len(rows)
    by = {"reduction": [r for r in rows if r["family"] == "reduction"],
          "conv": [r for r in rows if r["family"] == "conv"]}
    assert len(by["reduction"]) == 60 and len(by["conv"]) == 36
    out = {"date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "basis": "corrected oracle min(passing hand, torch); validate-then-time arms; m3 grids/seed/budgets/decision rule unchanged from m3_autotune_curve.py",
           "verdicts": {k: analyze(rs) for k, rs in by.items()}}
    json.dump(out, open(OUT, "w"), indent=1)
    for k, v in out["verdicts"].items():
        print("==", k)
        print(json.dumps({kk: vv for kk, vv in v.items() if kk != "choice_counts"}, indent=1)[:1600])
        print("choices:", json.dumps(v["choice_counts"]))


if __name__ == "__main__":
    main()
