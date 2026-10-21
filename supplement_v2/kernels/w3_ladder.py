"""W3 ladder: ablation A-E on conv-as-gemm -- the decisive 1-vs-2-feature test.

Pre-registered BEFORE evaluation 2026-10-18 ~01:05Z (docstring; collector
w3_conv_gemm.py carries the full pre-registration). Identical to
w4_dispatcher_v1.py except the three declared deltas:
  D1 fold axis = LOO over the 6 (H,W) sizes (shape generalization to unseen
     spatial sizes), instead of LOO-K.
  D2 gate hand arm = per-fold refit variant (fallback imp_B64x32 if absent);
     the bt feature still uses the FALLBACK variant's ctas, exactly v1's
     declared approximation, so family E is bit-identical in form.
  D3 threshold grids extended for conv scales (declared in collector docstring):
     m0 +{128,256,512}; K0 +{128,256,512,1024,2048,4096}; b0 +16. r0 unchanged;
     r = t_torch / launch-floor(x2) with the floor from results/e2b_floor.json
     meta (same instrument as v1; provenance P1).
Adoption bar unchanged: 0 held-out pts >1% worse than torch AND mean regret <
always-torch. torch_cmp (inductor) is context only: reported, never in gate/oracle.
Single evaluation pass; rerun only for harness bugs (none expected -- pure post).

Run: python3 w3_ladder.py -> results/w3_ladder.json + stdout verdict.
"""
import json, os, itertools

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SRC = os.path.join(RES, "w3_conv_gemm.json")
EB = os.path.join(RES, "e2b_floor.json")
OUT = os.path.join(RES, "w3_ladder.json")

VARIANT_META = {  # name -> (BM, BN) for tie-breaks; declared
    "imp_B32x32": (32, 32), "imp_B64x32": (64, 32), "imp_B64x64": (64, 64),
    "imp_B128x64": (128, 64), "imp_B32x64K64": (32, 64)}
FALLBACK = "imp_B64x32"
HANDS = list(VARIANT_META)


def build_points():
    src = json.load(open(SRC))
    t2 = json.load(open(EB))["meta"]["launch_floor_two_ms"]["med_ms"] * 1e3
    pts = []
    for row in src["rows"]:
        var = row["variants"]
        t_torch = var["torch"]["med_ms"]
        hand = {k: v for k, v in var.items() if k in VARIANT_META}
        t_bh, best_name = min((v["med_ms"], k) for k, v in hand.items())
        ti = var["torch"]["iqr_ms"]; bi = hand[best_name]["iqr_ms"]
        noise = max(((ti[1] - ti[0]) + (bi[1] - bi[0])) / 2 / t_torch, 0.01)
        pts.append(dict(
            N=row["N"], H=row["H"], K=row["K"], M=row["M"], bytes=row["bytes"],
            t_torch=t_torch, t_hand=hand[FALLBACK]["med_ms"], _fbname=FALLBACK,
            t_oracle=min(t_torch, t_bh), r=t_torch / t2,
            variants=hand,
            label="hand" if (t_torch - t_bh) / t_torch > noise else
                  ("torch" if (t_torch - t_bh) / t_torch < -noise else "tie")))
    return pts


def refit_hand(train):
    best = None
    for u in HANDS:
        regs, ok = [], True
        for p in train:
            v = p["variants"].get(u)
            if not v:
                ok = False; break
            regs.append((v["med_ms"] - p["t_oracle"]) / p["t_oracle"])
        if not ok:
            continue
        key = (round(sum(regs) / len(regs), 9),) + VARIANT_META[u]
        if best is None or key < best[0]:
            best = (key, u)
    return best[1] if best else None


def apply_variant(p, name):
    v = p["variants"].get(name)
    if v is None:
        v = p["variants"][FALLBACK]; name = FALLBACK
    return v["med_ms"], v["ctas"], name


GRIDS = dict(
    m0=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    K0=[0, 128, 256, 512, 1024, 2048, 4096],
    r0=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0],
    b0=[0, 16, 64, 256, 1024, 4096, 16384, 65536])


def fit_family(family, train, hname):
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
            _, ctas, _ = apply_variant(p, FALLBACK)
            h = 2 * p["bytes"] / (ctas * 128) >= b0
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
        # tie -> smaller hand region: fewer hand assignments preferred
        nhand = sum(1 for p in train
                    if p["M"] >= m0 and (not K0 or p["K"] >= K0) and (not r0 or p["r"] >= r0)
                    and (not b0 or 2 * p["bytes"] / (apply_variant(p, FALLBACK)[1] * 128) >= b0))
        key = (round(sum(regs) / len(regs), 9), nhand)
        if best is None or key < best[0]:
            best = (key, dict(m0=m0, K0=K0, r0=r0, b0=b0))
    return best[1]


def gate_pred(fam, th, hname):
    def g(p):
        h = p["M"] >= th["m0"] and (not th["K0"] or p["K"] >= th["K0"]) and \
            (not th["r0"] or p["r"] >= th["r0"])
        if h and th["b0"]:
            _, ctas, _ = apply_variant(p, FALLBACK)
            h = 2 * p["bytes"] / (ctas * 128) >= th["b0"]
        if h:
            t, _, nm = apply_variant(p, hname)
            return ("hand", nm)
        return ("torch",)
    return g


def eval_choice(choice, pts):
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


FAMILIES = ["A", "B", "C", "D", "E"]
FEATCOUNT = {"A": 1, "B": 2, "C": 2, "D": 3, "E": 2}


def pooled(pairs):
    regs, eq, worse = [], 0, 0
    for p, c in pairs:
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
    fold_rows, te_pairs, fam_pool = [], [], {f: [] for f in FAMILIES}
    for H in Hs:
        tr = [p for p in pts if p["H"] != H]
        te = [p for p in pts if p["H"] == H]
        hname = refit_hand(tr) or FALLBACK
        fam_res, best = {}, None
        for fam in FAMILIES:
            th = fit_family(fam, tr, hname)
            g = gate_pred(fam, th, hname)
            r = eval_choice(g, tr)
            fam_res[fam] = dict(th=th, train=r)
            key = (round(r["mean_regret"], 9), FEATCOUNT[fam])
            if best is None or key < best[0]:
                best = (key, fam, th, g)
        _, fam, th, g = best
        fold_rows.append(dict(held_H=H, hand_variant=hname, family=fam, thresholds=th,
                              fam_train={f: fam_res[f]["train"]["mean_regret"] for f in FAMILIES},
                              fam_th={f: fam_res[f]["th"] for f in FAMILIES},
                              test_gate=eval_choice(g, te),
                              test_torch=eval_choice(lambda p: ("torch",), te),
                              test_hand=eval_choice(lambda p: ("hand", hname), te)))
        for p in te:
            te_pairs.append((p, g(p)))
            for f in FAMILIES:
                fam_pool[f].append((p, gate_pred(f, fam_res[f]["th"], hname)(p)))

    def fam_pool_stats():
        out = {}
        for f, pairs in fam_pool.items():
            regs = []
            for p, c in pairs:
                t = apply_variant(p, c[1])[0] if c[0] == "hand" else p["t_torch"]
                regs.append((t - p["t_oracle"]) / p["t_oracle"])
            regs.sort()
            out[f] = dict(mean_regret=sum(regs) / len(regs), med_regret=regs[len(regs) // 2],
                          n_worse_than_torch_1pct=sum(1 for (p, c) in pairs
                                                      if (apply_variant(p, c[1])[0] if c[0] == "hand" else p["t_torch"]) > p["t_torch"] * 1.01))
        return out

    res = dict(
        meta=dict(date=__import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
                  n_pts=len(pts), folds=Hs, pre_reg="see w3_ladder.py + w3_conv_gemm.py docstrings"),
        fold_rows=[{k: v for k, v in r.items()} for r in fold_rows],
        pooled_loo=dict(gate=pooled(te_pairs),
                        torch=pooled([(p, ("torch",)) for p, _ in te_pairs]),
                        hand_refit=pooled([(p, ("hand", refit_hand([q for q in pts if q["H"] != p["H"]] or pts))) for p, _ in te_pairs])),
        ablation_pooled=fam_pool_stats(),
        always=dict(torch=pooled([(p, ("torch",)) for p in pts])),
    )
    # context: inductor arm, never in gate/oracle
    src = json.load(open(SRC))
    cmp_rows = [r for r in src["rows"] if "med_ms" in r["variants"].get("torch_cmp", {})]
    if cmp_rows:
        faster = sum(1 for r in cmp_rows
                     if r["variants"]["torch_cmp"]["med_ms"] < min(
                         v["med_ms"] for k, v in r["variants"].items() if k != "torch_cmp"))
        res["context_inductor"] = dict(n=len(cmp_rows), beats_all_arms=faster)

    json.dump(res, open(OUT, "w"), indent=1)
    for r in fold_rows:
        print("H=%2d fam=%s th=%s train=%s test gate %.4f torch %.4f hand %.4f w% d" % (
            r["held_H"], r["family"], r["thresholds"],
            {k: round(v, 4) for k, v in r["fam_train"].items()},
            r["test_gate"]["mean_regret"], r["test_torch"]["mean_regret"],
            r["test_hand"]["mean_regret"], r["test_gate"]["n_worse_than_torch_1pct"]))
    print("POOLED gate", {k: round(v, 4) for k, v in res["pooled_loo"]["gate"].items()})
    print("POOLED ablation", {f: round(v["mean_regret"], 4) for f, v in res["ablation_pooled"].items()},
          {f: v["n_worse_than_torch_1pct"] for f, v in res["ablation_pooled"].items()})
    print("always-torch", round(res["always"]["torch"]["mean_regret"], 4))


if __name__ == "__main__":
    main()
