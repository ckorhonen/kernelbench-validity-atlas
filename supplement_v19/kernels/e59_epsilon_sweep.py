#!/usr/bin/env python3
"""e59 — epsilon admissible-set / frontier sweep (CPU only).

Prereg: lab/e59_epsilon_sweep_prereg.md (frozen and committed BEFORE this script ran).
Population: lab/results/r1_conv_gated.jsonl (56 conv shapes x 12 arms, recorded med_ms + max_abs_err +
ref_absmax + pass_1e4). No GPU, no re-measurement: admissibility is recomputed from recorded maxima.

admissible(eps) := max_abs_err <= eps*(1+ref_absmax)   [eps = atol = rtol]
Instrument validation: reconstruct at eps=1e-4 vs recorded pass_1e4 on all 672 arm-checks; VALID iff
agreement >= 95%, else no eps-ordering claim.
"""
import json, hashlib, os, itertools, datetime

SRC = "lab/results/r1_conv_gated.jsonl"
OUT = "lab/results/e59_epsilon_sweep.json"
EPS = [1e-4, 1e-3, 1e-2, 1e-1]

def cls(arm, meta):
    if arm == "torch": return "torch"
    if arm == "torch_cmp": return "compile"
    return "hand_" + (meta.get("precision") or "?")

def main():
    src_sha = hashlib.sha256(open(SRC, "rb").read()).hexdigest()[:16]
    rows = [json.loads(l) for l in open(SRC)]
    checks = []
    for i, r in enumerate(rows):
        ref = r["ref_absmax"]
        for arm, m in r["variants"].items():
            checks.append({"shape": i, "arm": arm, "cls": cls(arm, m), "ms": m.get("med_ms"),
                           "max_abs_err": m.get("max_abs_err"), "pass_1e4": m.get("pass_1e4"),
                           "ref_absmax": ref})
    # ---- instrument validation leg (eps = 1e-4) ----
    agree = mism = 0
    mism_examples = []
    for c in checks:
        if c["max_abs_err"] is None: continue
        rec = c["max_abs_err"] <= 1e-4 * (1 + c["ref_absmax"])
        truth = bool(c["pass_1e4"])
        if rec == truth: agree += 1
        else:
            mism += 1
            if len(mism_examples) < 8:
                mism_examples.append({"shape": c["shape"], "arm": c["arm"], "recon": rec,
                                      "recorded": truth, "max_abs_err": c["max_abs_err"],
                                      "ref_absmax": c["ref_absmax"]})
    n = agree + mism
    agreement = agree / n if n else 0.0
    instrument_valid = agreement >= 0.95

    # ---- frontier + class structure per eps ----
    by_eps = {}
    per_shape = {}
    for eps in EPS:
        adm = {}
        for c in checks:
            if c["max_abs_err"] is None or c["ms"] is None: continue
            if c["max_abs_err"] <= eps * (1 + c["ref_absmax"]):
                adm.setdefault(c["shape"], []).append(c)
        class_counts = {}
        shapes_with_hand = 0
        for s, lst in adm.items():
            cs = sorted({c["cls"] for c in lst})
            for k in cs: class_counts[k] = class_counts.get(k, 0) + 1
            if any(c["cls"].startswith("hand") for c in lst): shapes_with_hand += 1
        frontier = {}
        for s in range(len(rows)):
            lst = adm.get(s, [])
            if not lst:
                frontier[s] = None; continue
            best = min(lst, key=lambda c: (c["ms"], c["arm"]))
            frontier[s] = {"arm": best["arm"], "cls": best["cls"], "ms": best["ms"]}
        by_eps[f"{eps:g}"] = {
            "n_arms_admissible": sum(len(v) for v in adm.values()),
            "n_shapes_any_admissible": len(adm),
            "n_shapes_with_hand": shapes_with_hand,
            "class_shape_counts": class_counts,
            "frontier_class_counts": {},
        }
        for s, f in frontier.items():
            k = f["cls"] if f else "none"
            by_eps[f"{eps:g}"]["frontier_class_counts"][k] = by_eps[f"{eps:g}"]["frontier_class_counts"].get(k, 0) + 1
        per_shape[f"{eps:g}"] = frontier

    # ---- flip counts vs eps = 1e-4 ----
    base = per_shape[f"{EPS[0]:g}"]
    flips = {}
    flip_detail = []
    for eps in EPS[1:]:
        cur = per_shape[f"{eps:g}"]
        n_flip = 0
        detail = []
        for s in range(len(rows)):
            b, c = base.get(s), cur.get(s)
            ba = b["arm"] if b else None
            ca = c["arm"] if c else None
            if ba != ca:
                n_flip += 1
                detail.append({"shape": s, "eps": f"{eps:g}", "frontier_1e4": ba, "frontier_eps": ca,
                               "cls_1e4": b["cls"] if b else None, "cls_eps": c["cls"] if c else None,
                               "ms_1e4": b["ms"] if b else None, "ms_eps": c["ms"] if c else None})
        flips[f"{eps:g}"] = {"n_shapes_flipped": n_flip, "of": len(rows)}
        flip_detail.extend(detail)

    # ---- hand-only stratum: does the fastest admissible hand arm change / beat non-hand ----
    hand_only = {}
    for eps in EPS:
        changed = 0
        hand_beats = 0
        any_hand = 0
        detail = []
        for s in range(len(rows)):
            ref = rows[s]["ref_absmax"]
            hands = [(a, m) for a, m in rows[s]["variants"].items() if a.startswith("imp_")]
            adm_h = [(a, m) for a, m in hands if m.get("max_abs_err") is not None and m["max_abs_err"] <= eps * (1 + ref)]
            nonhand = [(a, m) for a, m in rows[s]["variants"].items() if not a.startswith("imp_") and m.get("med_ms")]
            if not adm_h: continue
            any_hand += 1
            best_h = min(adm_h, key=lambda t: (t[1]["med_ms"], t[0]))
            best_n = min(nonhand, key=lambda t: t[1]["med_ms"]) if nonhand else None
            if best_n and best_h[1]["med_ms"] < best_n[1]["med_ms"]:
                hand_beats += 1
                detail.append({"shape": s, "hand": best_h[0], "hand_ms": best_h[1]["med_ms"],
                               "best_nonhand": best_n[0], "nonhand_ms": best_n[1]["med_ms"],
                               "ratio": best_h[1]["med_ms"] / best_n[1]["med_ms"]})
        # fastest admissible hand arm identity change vs 1e-4
        hand_only[f"{eps:g}"] = {"n_shapes_hand_admissible": any_hand,
                                 "n_shapes_hand_faster_than_best_nonhand": hand_beats,
                                 "examples": detail[:12]}

    # ---- pairwise class dominance ----
    classes = sorted({c["cls"] for c in checks})
    dom = {}
    for eps in EPS:
        mat = {}
        for a, b in itertools.permutations(classes, 2):
            cnt = 0
            for s in range(len(rows)):
                ref = rows[s]["ref_absmax"]
                def any_adm(cl):
                    return any(c["cls"] == cl and c["max_abs_err"] is not None and
                               c["max_abs_err"] <= eps * (1 + ref) for c in checks if c["shape"] == s)
                if any_adm(a) and not any_adm(b): cnt += 1
            mat[f"{a}>{b}"] = cnt
        dom[f"{eps:g}"] = mat

    # ---- verdict per prereg ----
    total_flips = sum(v["n_shapes_flipped"] for v in flips.values())
    if not instrument_valid:
        verdict = "INSTRUMENT_INVALID"
    elif total_flips == 0:
        verdict = "ORDERING_INVARIANT_KILL_FIRED"
    else:
        verdict = "FRONTIER_MOVES"
    first_flip_eps = None
    for eps in EPS[1:]:
        if flips[f"{eps:g}"]["n_shapes_flipped"] > 0:
            first_flip_eps = f"{eps:g}"; break

    out = {
        "prereg": "lab/e59_epsilon_sweep_prereg.md",
        "source": SRC, "source_sha": src_sha,
        "date": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "population": {"rows": len(rows), "arm_checks": len(checks), "eps": EPS},
        "instrument_validation": {"agreement": round(agreement, 6), "agree": agree, "mismatch": mism,
                                  "valid": instrument_valid, "examples": mism_examples},
        "per_eps": by_eps,
        "frontier_flips": flips, "first_flip_eps": first_flip_eps,
        "flip_detail": flip_detail,
        "hand_only": hand_only,
        "class_dominance": dom,
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ["instrument_validation", "per_eps", "frontier_flips",
                                          "first_flip_eps", "verdict"]}, indent=1))
    print("hand_only:", json.dumps(hand_only, indent=1)[:1500])

if __name__ == "__main__":
    main()
