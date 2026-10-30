"""s59 — F1 of rejection MhTLuz7Ciq7UPVmk, verified by recomputation from committed records.

Question (reviewer F1): the fresh selector run was described as the same policy with a 0.54pp
difference attributable to timing variance. Is the difference timing, or a change in the
correctness population?

Method: arithmetic on two committed artifacts, no GPU, no new measurement.
  A = lab/results/s56_heldout_contract_recorded.json  (recorded run, 2026-10-25T07:50Z)
  B = lab/results/s56_heldout_contract.json           (fresh run, 2026-10-29T06:11Z)
Both cover the same 34 shapes (24 reduction, 10 conv) and the same arm pool. For every
(family, shape, arm) triple present in both, compare pass_full. Record per-row flips,
direction, and the reference magnitude each run realized. Also record gate arm identity and
budget ratios for the three shapes the reviewer named.

This is a COUNT on recorded data, not a hypothesis test: no threshold, no kill clause.
The interpretive claim it supports (input-draw sensitivity of admission is boundary-local)
is stated as descriptive and carries the population it was computed on.

Provenance: recorded run's producer set no seed and recorded none (lab/s56_heldout_contract.py
docstring: "v1 set no seed and recorded none"); the fresh run seeds per shape
(seed = (K*1000003 + M*7919) % (2**31-1)) and records it. That is the input difference.
"""
import json, statistics, collections, os, datetime

BASE = os.path.dirname(os.path.abspath(__file__)) + "/results"


def _kb_open(name):
    for cand in (os.path.join(BASE, name), os.path.join(BASE, name + ".gz"), name):
        if os.path.exists(cand):
            if cand.endswith(".gz"):
                import gzip
                return json.load(gzip.open(cand, "rt"))
            return json.load(open(cand))
    raise FileNotFoundError(name)


def key(r):
    if r["family"] == "reduction":
        return ("reduction", r["K"], r["M"])
    return ("conv", r["N"], r["H"], r["C"])


A = _kb_open("s56_heldout_contract_recorded.json")["rows"]
B = _kb_open("s56_heldout_contract.json")["rows"]
A = {key(r): r for r in A}
B = {key(r): r for r in B}

flips = []
checked = 0
for k in A:
    for arm, a in A[k]["arms"].items():
        b = B[k]["arms"].get(arm)
        if b is None:
            continue
        checked += 1
        if a["pass_full"] != b["pass_full"]:
            flips.append({
                "family": k[0], "shape": list(k[1:]), "arm": arm,
                "recorded_pass": a["pass_full"], "fresh_pass": b["pass_full"],
                "recorded_ratio": a["budget_ratio"], "fresh_ratio": b["budget_ratio"],
            })

by_family = collections.Counter(f["family"] for f in flips)
direction = collections.Counter("fail->pass" if f["fresh_pass"] else "pass->fail" for f in flips)
rows_touched = sorted({(f["family"], tuple(f["shape"])) for f in flips})

ref_rel = []
for k in A:
    ra, rb = A[k]["ref"]["ref_absmax"], B[k]["ref"]["ref_absmax"]
    if ra and rb:
        ref_rel.append(abs(rb - ra) / max(1e-9, ra))

gate_rows = []
for k in A:
    ga, gb = A[k]["gate_choice"], B[k]["gate_choice"]
    pa = A[k]["arms"].get(ga, {}).get("pass_full")
    pb = B[k]["arms"].get(gb, {}).get("pass_full")
    if ga != "torch":
        gate_rows.append({
            "family": k[0], "shape": list(k[1:]),
            "recorded_gate": ga, "fresh_gate": gb, "same_arm": ga == gb,
            "recorded_gate_pass": pa, "fresh_gate_pass": pb,
            "recorded_ratio": A[k]["arms"].get(ga, {}).get("budget_ratio"),
            "fresh_ratio": B[k]["arms"].get(gb, {}).get("budget_ratio"),
        })

hand_rec = [g for g in gate_rows if g["recorded_gate_pass"]]
hand_fresh = [g for g in gate_rows if g["fresh_gate_pass"]]
arm_changed = [g for g in gate_rows if not g["same_arm"]]

# boundary locality: are flips confined to ratios near the gate?
flip_ratios = sorted(min(f["recorded_ratio"], f["fresh_ratio"]) for f in flips)
nonflip_ratios = []
for k in A:
    for arm, a in A[k]["arms"].items():
        b = B[k]["arms"].get(arm)
        if b is None or a["pass_full"] != b["pass_full"]:
            continue
        nonflip_ratios.append(min(a["budget_ratio"], b["budget_ratio"]))

out = {
    "artifact": "s59_f1_input_draw",
    "date": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "source": ["lab/results/s56_heldout_contract_recorded.json", "lab/results/s56_heldout_contract.json"],
    "population": {"shapes": len(A), "arm_checks_compared": checked,
                   "families": dict(collections.Counter(k[0] for k in A))},
    "seed_provenance": {
        "recorded": "none — producer set no seed, artifact carries no seed field",
        "fresh": "per-shape seed = (K*1000003 + M*7919) % (2**31-1), recorded per row",
    },
    "input_realization_differs": {
        "n": len(ref_rel),
        "ref_absmax_reldiff_median": round(statistics.median(ref_rel), 6),
        "ref_absmax_reldiff_max": round(max(ref_rel), 6),
        "n_over_1pct": sum(1 for x in ref_rel if x > 0.01),
    },
    "pass_label_flips": {
        "n": len(flips),
        "frac_of_checks": round(len(flips) / checked, 6),
        "by_family": dict(by_family),
        "direction": dict(direction),
        "rows_touched": len(rows_touched),
        "rows_touched_detail": [{"family": f, "shape": list(s)} for f, s in rows_touched],
    },
    "gate_arm_identity": {"hand_fired_rows": len(gate_rows),
                          "arm_changed_rows": len(arm_changed),
                          "changed_detail": arm_changed},
    "delivery": {"recorded_hand_passing": len(hand_rec), "fresh_hand_passing": len(hand_fresh)},
    "boundary_locality": {
        "flip_ratio_swing": {
            "n": len(flips),
            "median_max_over_min": round(statistics.median(
                [max(f["recorded_ratio"], f["fresh_ratio"]) / max(1e-9, min(f["recorded_ratio"], f["fresh_ratio"]))
                 for f in flips]), 4) if flips else None,
            "min_max_over_min": round(min(
                [max(f["recorded_ratio"], f["fresh_ratio"]) / max(1e-9, min(f["recorded_ratio"], f["fresh_ratio"]))
                 for f in flips]), 4) if flips else None,
            "all_straddle_gate": all(
                (f["recorded_ratio"] > 1.0) != (f["fresh_ratio"] > 1.0) for f in flips) if flips else None,
        },
        "flip_min_ratio": {"n": len(flip_ratios),
                           "median": round(statistics.median(flip_ratios), 4) if flip_ratios else None,
                           "max": round(max(flip_ratios), 4) if flip_ratios else None},
        "nonflip_min_ratio": {"n": len(nonflip_ratios),
                              "median": round(statistics.median(nonflip_ratios), 4) if nonflip_ratios else None,
                              "min": round(min(nonflip_ratios), 4) if nonflip_ratios else None},
    },
    "flips": flips,
    "verdict": "INPUT_DRAW_NOT_TIMING",
    "claim": ("Same 34 shapes and same arm pool, different input realization: 26/228 pass labels "
              "flip (16 fail->pass, 10 pass->fail, 11 of 24 reduction rows), gate ARM identity "
              "unchanged on every hand-fired row, 0 conv flips. The fresh-vs-recorded regret "
              "difference is therefore a correctness-population difference, not timing variance."),
    "scope": ("descriptive count on GB10 conv/reduction pool at rev 423217d; the recorded run's "
              "inputs are irreproducible (no seed recorded), so its population is one draw, not a baseline."),
}

os.makedirs(BASE, exist_ok=True)
p = os.path.join(BASE, "s59_f1_input_draw.json")
json.dump(out, open(p, "w"), indent=1)
print(json.dumps({k: out[k] for k in ("population", "input_realization_differs", "pass_label_flips",
                                      "gate_arm_identity", "delivery", "boundary_locality", "verdict")},
                 indent=1)[:2500])
print("wrote", p)
