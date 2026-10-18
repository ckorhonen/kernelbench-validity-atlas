"""M1: bt-from-deployed-ctas recheck (plan milestone M1; no GPU).

Pre-registered BEFORE evaluation 2026-10-18 ~07:20Z (this docstring; DIARY same tick).
Question (debt #2 / revival plan M1): the W3 ladder E-family misfire at held-H=14
(fold regret 0.628; bt computed from the FALLBACK variant's ctas fired `hand` where
the deployed refit variant was +114%) -- is it an artifact of the proxy statistic?

Single declared delta vs lab/w3_ladder.py, everything else bit-identical (imports
shared; grids, ties, LOO-H folds, noise rule, oracle, family selection unchanged):
  D1': the bt feature uses the DEPLOYED variant's ctas -- the fold-refit hand
       (hname) that the gate would actually launch; if that variant is absent at a
       point, the fallback imp_B64x32 ctas is used (declared, same rule as the
       launch itself). Applied identically in fit, tie-break count, and gate.

Success criterion (vanishing): held-H=14 E-family test regret improves vs the
proxy run AND the fold's worse-than-torch-1pct count attributable to bt-misfires
drops; enumerated disagreement points (proxy-hand xor deployed-hand) reported.
Falsifier (not vanishing): no improvement in that fold's E regret/worse-count, or
family selection flips so E is no longer the per-fold argmin (comparison moot) --
then the misfire is NOT the proxy; the dispatcher spec keeps bt-implementation-
local as a spec note with this null as its evidence, and the conv object stays
the family-local (H,C) rule. P5: report either way.

Run: python3 m1_bt_deployed.py -> results/m1_bt_deployed.json (+ diff vs w3_ladder.json)
"""
import json, os, itertools, time
from w3_ladder import (build_points, refit_hand, apply_variant, GRIDS,
                       FAMILIES, FEATCOUNT, FALLBACK)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "m1_bt_deployed.json")
ORIG = os.path.join(RES, "w3_ladder.json")


def bt_ok(p, name, b0):
    """bt from the DEPLOYED variant's ctas (D1'); fallback ctas only if absent."""
    _, ctas, _ = apply_variant(p, name)
    return 2 * p["bytes"] / (ctas * 128) >= b0


def fit_family_deployed(family, train, hname):
    if family == "A":
        combos = [(m0, 0, 0.0, 0) for m0 in GRIDS["m0"]]
    elif family == "B":
        combos = [(m0, K0, 0.0, 0) for m0, K0 in itertools.product(GRIDS["m0"], GRIDS["K0"])]
    elif family == "C":
        combos = [(m0, 0, r0, 0) for m0, r0 in itertools.product(GRIDS["m0"], GRIDS["r0"])]
    elif family == "D":
        combos = [(m0, K0, r0, 0) for m0, K0, r0 in
                  itertools.product(GRIDS["m0"], GRIDS["K0"], GRIDS["r0"])]
    else:
        combos = [(m0, 0, 0.0, b0) for m0, b0 in itertools.product(GRIDS["m0"], GRIDS["b0"])]

    def hand_time(p, m0, K0, r0, b0):
        h = p["M"] >= m0 and (not K0 or p["K"] >= K0) and (not r0 or p["r"] >= r0)
        if h and b0:
            h = bt_ok(p, hname, b0)
        if not h:
            return p["t_torch"]
        t, _, nm = apply_variant(p, hname)
        return t

    best = None
    for m0, K0, r0, b0 in combos:
        regs = []
        for p in train:
            t = hand_time(p, m0, K0, r0, b0)
            regs.append((t - p["t_oracle"]) / p["t_oracle"])
        nhand = sum(1 for p in train
                    if p["M"] >= m0 and (not K0 or p["K"] >= K0) and (not r0 or p["r"] >= r0)
                    and (not b0 or bt_ok(p, hname, b0)))
        key = (round(sum(regs) / len(regs), 9), nhand)
        if best is None or key < best[0]:
            best = (key, dict(m0=m0, K0=K0, r0=r0, b0=b0))
    return best[1]


def gate_pred_deployed(fam, th, hname):
    def g(p):
        h = p["M"] >= th["m0"] and (not th["K0"] or p["K"] >= th["K0"]) and \
            (not th["r0"] or p["r"] >= th["r0"])
        if h and th["b0"]:
            h = bt_ok(p, hname, th["b0"])
        if h:
            t, _, nm = apply_variant(p, hname)
            return ("hand", nm)
        return ("torch",)
    return g


def eval_choice_local(choice, pts):
    regs, eq, worse = [], 0, 0
    for p in pts:
        c = choice(p)
        t = apply_variant(p, c[1])[0] if c[0] == "hand" else p["t_torch"]
        regs.append((t - p["t_oracle"]) / p["t_oracle"])
        eq += t <= p["t_oracle"] * 1.001
        worse += t > p["t_torch"] * 1.01
    regs.sort(); n = len(regs)
    return dict(mean_regret=sum(regs) / n, med_regret=regs[n // 2],
                p90_regret=regs[min(n - 1, int(.9 * n))], max_regret=max(regs),
                frac_oracle=eq / n, n_worse_than_torch_1pct=worse, n=n)


def main():
    pts = build_points()
    Hs = sorted({p["H"] for p in pts})
    orig = json.load(open(ORIG))
    orig_by_H = {r["held_H"]: r for r in orig["fold_rows"]}

    fold_rows, pooled_pairs = [], []
    for H in Hs:
        tr = [p for p in pts if p["H"] != H]
        te = [p for p in pts if p["H"] == H]
        hname = refit_hand(tr) or FALLBACK
        fam_res, best = {}, None
        for fam in FAMILIES:
            th = fit_family_deployed(fam, tr, hname)
            g = gate_pred_deployed(fam, th, hname)
            r = eval_choice_local(g, tr)
            fam_res[fam] = (th, r)
            key = (round(r["mean_regret"], 9), FEATCOUNT[fam])
            if best is None or key < best[0]:
                best = (key, fam, th, g)
        _, fam, th, g = best
        te_res = eval_choice_local(g, te)
        # disagreement audit: proxy-hand vs deployed-hand per point (E family, fold thresholds)
        thE = fam_res["E"][0]
        disagreements = []
        gE = gate_pred_deployed("E", thE, hname)
        for p in te:
            a = gE(p)[0] == "hand"
            _, ctas_fb, _ = apply_variant(p, FALLBACK)
            b_proxy = p["M"] >= thE["m0"] and 2 * p["bytes"] / (ctas_fb * 128) >= thE["b0"]
            if a != bool(b_proxy):
                t_dep = apply_variant(p, hname)[0]
                disagreements.append(dict(H=p["H"], M=p["M"], K=p["K"],
                                          proxy_hand=bool(b_proxy), deployed_hand=a,
                                          regret_if_hand=round((t_dep - p["t_oracle"]) / p["t_oracle"], 4)))
        o = orig_by_H[H]
        row = dict(held_H=H, hand_variant=hname, family=fam, thresholds=th,
                   fam_train={f: fam_res[f][1]["mean_regret"] for f in FAMILIES},
                   E_test_deployed=eval_choice_local(gate_pred_deployed("E", fam_res["E"][0], hname), te),
                   E_test_orig=o["test_gate"] if o["family"] == "E" else None,
                   orig_family=o["family"], test_gate=te_res, disagreement_E=disagreements)
        fold_rows.append(row)
        pooled_pairs += [(p, g(p)) for p in te]

    def pooled(pairs):
        regs, worse = [], 0
        for p, c in pairs:
            t = apply_variant(p, c[1])[0] if c[0] == "hand" else p["t_torch"]
            regs.append((t - p["t_oracle"]) / p["t_oracle"])
            worse += t > p["t_torch"] * 1.01
        regs.sort()
        return dict(mean_regret=sum(regs) / len(regs), med_regret=regs[len(regs) // 2],
                    n_worse_than_torch_1pct=worse, n=len(regs))

    res = dict(
        meta=dict(date=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  n_pts=len(pts), delta="D1': bt ctas from deployed (refit) variant",
                  pre_reg="this docstring, before evaluation"),
        fold_rows=fold_rows,
        pooled=dict(gate=pooled(pooled_pairs),
                    orig_gate=orig["pooled_loo"]["gate"],
                    always_torch=orig["always"]["torch"]),
    )
    json.dump(res, open(OUT, "w"), indent=1)
    for r in res["fold_rows"]:
        o = r["E_test_orig"]; n = r["E_test_deployed"]
        print("H=%2d fam=%s/%s E: orig %s deployed %.4f | gate %.4f worse %s->%d disagg %d" % (
            r["held_H"], r["family"], r["orig_family"],
            "n/a" if o is None else "%.4f" % o["mean_regret"],
            n["mean_regret"], r["test_gate"]["mean_regret"],
            (o or {}).get("n_worse_than_torch_1pct", -1), n["n_worse_than_torch_1pct"],
            len(r["disagreement_E"])))
    print("POOLED deployed", {k: round(v, 4) if isinstance(v, float) else v
                              for k, v in res["pooled"]["gate"].items()})
    print("POOLED orig    ", {k: round(v, 4) if isinstance(v, float) else v
                              for k, v in res["pooled"]["orig_gate"].items()})


if __name__ == "__main__":
    main()
