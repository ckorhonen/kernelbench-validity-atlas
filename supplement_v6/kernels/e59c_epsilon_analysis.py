#!/usr/bin/env python3
"""e59c — epsilon admissible-set / frontier sweep on the REPAIRED instrument (CPU only).

Inputs: lab/results/e59b_budget_ratios.jsonl (per-element budget ratios recorded on spark, e59b) and
lab/results/r1_conv_gated.jsonl (inherited timings, matched by tag/N/H/C/arm).

Objects and kill clause are those of lab/e59_epsilon_sweep_prereg.md, unchanged. Instrument gate: the
candidate-scaled ratio at eps=1e-4 must reproduce the recorded pass_1e4 labels at >=99.5% agreement;
otherwise no eps-ordering claim is made (reported as INSTRUMENT_INVALID).
"""
import json, os, itertools, datetime

LAB = os.path.dirname(os.path.abspath(__file__))
RAT = os.path.join(LAB, "results", "e59b_budget_ratios.jsonl")
R1 = os.path.join(LAB, "results", "r1_conv_gated.jsonl")
OUT = os.path.join(LAB, "results", "e59c_epsilon_sweep.json")
EPS = [1e-4, 1e-3, 1e-2, 1e-1]

def cls(arm):
    if arm == "torch": return "torch"
    if arm == "torch_cmp": return "compile"
    return "hand_" + ("tf32" if "[tf32]" in arm else "ieee")

def main():
    rec = {}
    for line in open(R1):
        if line.strip():
            r = json.loads(line)
            rec[(r["tag"], r["N"], r["H"], r["C"])] = r
    rows = []
    for line in open(RAT):
        if line.strip():
            rows.append(json.loads(line))
    checks = []
    for r in rows:
        key = (r["tag"], r["N"], r["H"], r["C"])
        src = rec[key]
        for arm, a in r["arms"].items():
            if "ratio_cand" not in a: continue
            m = src["variants"].get(arm, {})
            checks.append({"shape": key, "arm": arm, "cls": cls(arm), "ratio": a["ratio_cand"],
                           "ratio_ref": a.get("ratio_ref"), "ms": m.get("med_ms"),
                           "recorded_pass": bool(m.get("pass_1e4"))})
    # instrument validation
    agree = mism = 0
    ex = []
    for c in checks:
        got = c["ratio"] <= 1.0
        if got == c["recorded_pass"]: agree += 1
        else:
            mism += 1
            if len(ex) < 8: ex.append({"shape": c["shape"], "arm": c["arm"], "recon": got,
                                       "recorded": c["recorded_pass"], "ratio": c["ratio"]})
    n = agree + mism
    agreement = agree / n if n else 0.0
    valid = agreement >= 0.995

    shapes = sorted({c["shape"] for c in checks})
    per_eps, per_shape = {}, {}
    for eps in EPS:
        thr = eps * 1e4
        adm = {}
        for c in checks:
            if c["ratio"] <= thr and c["ms"] is not None:
                adm.setdefault(c["shape"], []).append(c)
        class_counts, hand_shapes = {}, 0
        for s, lst in adm.items():
            for k in {c["cls"] for c in lst}:
                class_counts[k] = class_counts.get(k, 0) + 1
            if any(c["cls"].startswith("hand") for c in lst): hand_shapes += 1
        frontier = {}
        for s in shapes:
            lst = adm.get(s, [])
            if not lst: frontier[s] = None; continue
            b = min(lst, key=lambda c: (c["ms"], c["arm"]))
            frontier[s] = {"arm": b["arm"], "cls": b["cls"], "ms": b["ms"]}
        fcount = {}
        for s, f in frontier.items():
            k = f["cls"] if f else "none"
            fcount[k] = fcount.get(k, 0) + 1
        per_eps[f"{eps:g}"] = {"n_arms_admissible": sum(len(v) for v in adm.values()),
                               "n_shapes_any_admissible": len(adm),
                               "n_shapes_with_hand": hand_shapes,
                               "class_shape_counts": class_counts,
                               "frontier_class_counts": fcount}
        per_shape[f"{eps:g}"] = frontier

    base = per_shape[f"{EPS[0]:g}"]
    flips, flip_detail = {}, []
    for eps in EPS[1:]:
        cur = per_shape[f"{eps:g}"]
        n_flip = 0
        for s in shapes:
            b, c = base.get(s), cur.get(s)
            ba, ca = (b["arm"] if b else None), (c["arm"] if c else None)
            if ba != ca:
                n_flip += 1
                flip_detail.append({"shape": list(s), "eps": f"{eps:g}", "frontier_1e4": ba,
                                    "frontier_eps": ca, "cls_1e4": b["cls"] if b else None,
                                    "cls_eps": c["cls"] if c else None,
                                    "ms_1e4": b["ms"] if b else None, "ms_eps": c["ms"] if c else None})
        flips[f"{eps:g}"] = {"n_shapes_flipped": n_flip, "of": len(shapes)}

    hand_only = {}
    for eps in EPS:
        thr = eps * 1e4
        any_hand = beats = 0
        detail = []
        for s in shapes:
            lst = [c for c in checks if c["shape"] == s]
            hands = [c for c in lst if c["cls"].startswith("hand") and c["ratio"] <= thr and c["ms"]]
            nonh = [c for c in lst if not c["cls"].startswith("hand") and c["ms"]]
            if not hands: continue
            any_hand += 1
            bh = min(hands, key=lambda c: (c["ms"], c["arm"]))
            bn = min(nonh, key=lambda c: (c["ms"], c["arm"])) if nonh else None
            if bn and bh["ms"] < bn["ms"]:
                beats += 1
                detail.append({"shape": list(s), "hand": bh["arm"], "hand_ms": bh["ms"],
                               "best_nonhand": bn["arm"], "nonhand_ms": bn["ms"],
                               "speed_ratio": bh["ms"] / bn["ms"]})
        hand_only[f"{eps:g}"] = {"n_shapes_hand_admissible": any_hand,
                                 "n_shapes_hand_faster_than_best_nonhand": beats,
                                 "examples": detail[:15]}

    classes = ["torch", "compile", "hand_ieee", "hand_tf32"]
    dom = {}
    for eps in EPS:
        thr = eps * 1e4
        mat = {}
        for a, b in itertools.permutations(classes, 2):
            cnt = 0
            for s in shapes:
                lst = [c for c in checks if c["shape"] == s]
                aa = any(c["cls"] == a and c["ratio"] <= thr for c in lst)
                bb = any(c["cls"] == b and c["ratio"] <= thr for c in lst)
                if aa and not bb: cnt += 1
            mat[f"{a}>{b}"] = cnt
        dom[f"{eps:g}"] = mat

    total_flips = sum(v["n_shapes_flipped"] for v in flips.values())
    if not valid:
        verdict = "INSTRUMENT_INVALID"
    elif total_flips == 0:
        verdict = "ORDERING_INVARIANT_KILL_FIRED"
    else:
        verdict = "FRONTIER_MOVES"
    first = next((f"{e:g}" for e in EPS[1:] if flips[f"{e:g}"]["n_shapes_flipped"] > 0), None)

    out = {"prereg": "lab/e59_epsilon_sweep_prereg.md", "instrument": "lab/e59b_ratio_record.py",
           "source_ratio": RAT, "source_times": R1,
           "date": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
           "population": {"shapes": len(shapes), "arm_checks": len(checks), "eps": EPS},
           "instrument_validation": {"agreement": round(agreement, 6), "agree": agree,
                                     "mismatch": mism, "valid": valid, "examples": ex},
           "per_eps": per_eps, "frontier_flips": flips, "first_flip_eps": first,
           "flip_detail": flip_detail, "hand_only": hand_only, "class_dominance": dom,
           "verdict": verdict}
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ["instrument_validation", "per_eps", "frontier_flips",
                                          "first_flip_eps", "hand_only", "verdict"]}, indent=1)[:6000])

if __name__ == "__main__":
    main()
