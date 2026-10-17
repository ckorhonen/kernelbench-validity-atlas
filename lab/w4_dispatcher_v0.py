"""W4 dispatcher v0: feasibility of a features->rule selector on existing boundaries.

Data: e2_splitk_v3.json (60 pts, K in 2^12..2^22, M in 1..512, variants torch +
tsplit_S{4,16,64}_KT{16,64}); launch constants from e2b_floor.json (two-launch, P1
provenance: results/e2b_floor.json meta, measured 2026-10-17T15:14Z).

Pre-registered BEFORE evaluation (diary 2026-10-17 ~18:20Z):
  Primary metric: regret = (t_chosen - t_oracle)/t_oracle, t_oracle = min med_ms over
  {torch, all valid hand variants}. Dispatchers:
    D-torch : always torch (the comparator; IDENTITY: honest baseline)
    D-hand  : always hand, variant = KT64 (fallback any) with ctas closest to 64
              (prior from v3 EDA: winners ctas in [32,128], KT=64 dominant; NOT refit
              per fold -- stated limitation, v1 item)
    D-gate  : hand if (M >= m0 and r >= r0) else torch; r = t_torch/t_launch_two.
              (m0,r0) chosen per LOO-K train fold by min mean regret; ties -> smaller
              hand region (parsimony: default to the compiler, IDENTITY comparator).
              Rationale: session 5/e2b -- boundary is hand-advantage vs the launch
              constant, NOT waves alone; M* decreases as r grows.
  Splits: leave-one-K-out over the K values (generalization axis = unseen K at seen
  M grid; M-range extrapolation NOT tested -- stated limitation).
  Secondary: noise-gated label agreement (v3 rule margin vs combined-IQR noise).
  Adoption bar (informal): D-gate never loses to D-torch >1% on held-out points and
  captures most of the hand win; report %points dispatcher == oracle within 0.1%.

Correctness audit (side output): per-variant max_abs_err; err/sqrt(K) comparable scale;
KT16 anomaly (55.03 at K=4096,M=1) flagged for the tolerance axis (IDENTITY #4).

Run: python3 w4_dispatcher_v0.py  (stdlib only) -> results/w4_v0.json
"""
import json, os, time, itertools

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
V3 = os.path.join(RES, "e2_splitk_v3.json")
EB = os.path.join(RES, "e2b_floor.json")
OUT = os.path.join(RES, "w4_v0.json")


def hand_pick(variants):
    """D-hand prior: KT64 else any; ctas closest to 64. Deterministic."""
    cand = {k: v for k, v in variants.items() if k != "torch" and "med_ms" in v}
    if not cand:
        return None, None
    kt64 = {k: v for k, v in cand.items() if "_KT64" in k} or cand
    name, v = min(kt64.items(), key=lambda kv: abs(kv[1]["ctas"] - 64))
    return name, v


def build_points():
    v3 = json.load(open(V3))
    eb = json.load(open(EB))
    t2 = eb["meta"]["launch_floor_two_ms"]["med_ms"] * 1e3  # us
    pts = []
    for row in v3["rows"]:
        K, M, var = row["K"], row["M"], row["variants"]
        if "med_ms" not in var["torch"]:
            continue
        t_torch = var["torch"]["med_ms"]
        hand = {k: v for k, v in var.items() if k != "torch" and "med_ms" in v}
        if not hand:
            continue
        t_bh, best_name = min((v["med_ms"], k) for k, v in hand.items())
        bi, ti = hand[best_name]["iqr_ms"], var["torch"]["iqr_ms"]
        noise = max(((ti[1] - ti[0]) + (bi[1] - bi[0])) / 2 / t_torch, 0.01)
        margin = (t_torch - t_bh) / t_torch
        name, hv = hand_pick(var)
        pts.append(dict(
            K=K, M=M, lgK=K.bit_length() - 1, lgM=M.bit_length(),
            t_torch=t_torch, t_hand=hv["med_ms"], hand_name=name,
            hand_ctas=hv["ctas"], hand_err=hv.get("max_abs_err"),
            best_name=best_name, t_oracle=min(t_torch, t_bh),
            r=t_torch / t2,
            label="hand" if margin > noise else ("torch" if margin < -noise else "tie")))
    return pts, t2


def eval_choices(choice_fn, pts):
    """choice_fn(p) in {'hand','torch'}; returns regret stats vs oracle."""
    regs, eq, worse = [], 0, 0
    for p in pts:
        c = choice_fn(p)
        t = p["t_hand"] if c == "hand" else p["t_torch"]
        regs.append((t - p["t_oracle"]) / p["t_oracle"])
        if t <= p["t_oracle"] * 1.001:
            eq += 1
        if t > p["t_torch"] * 1.01:
            worse += 1
    regs.sort()
    n = len(regs)
    return dict(mean_regret=sum(regs) / n, med_regret=regs[n // 2],
                p90_regret=regs[min(n - 1, int(.9 * n))], max_regret=max(regs),
                frac_oracle=eq / n, n_worse_than_torch_1pct=worse, n=n)


def fit_gate(train):
    """grid (m0,r0): min train mean regret; tie -> smaller hand region."""
    best = None
    for m0, r0 in itertools.product([1, 2, 4, 8, 16, 32, 64],
                                    [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]):
        regs = []
        n_hand = 0
        for p in train:
            hand = p["M"] >= m0 and p["r"] >= r0
            n_hand += hand
            t = p["t_hand"] if hand else p["t_torch"]
            regs.append((t - p["t_oracle"]) / p["t_oracle"])
        key = (round(sum(regs) / len(regs), 6), n_hand)
        if best is None or key < best[0]:
            best = (key, m0, r0)
    return best[1], best[2]


def main():
    pts, t2 = build_points()
    Ks = sorted({p["K"] for p in pts})

    # pooled LOO: fit gate on all K' != K, predict on K
    fold_rows, te_pairs = [], []
    for K in Ks:
        tr = [p for p in pts if p["K"] != K]
        te = [p for p in pts if p["K"] == K]
        m0, r0 = fit_gate(tr)
        gate = lambda p, m0=m0, r0=r0: "hand" if (p["M"] >= m0 and p["r"] >= r0) else "torch"
        row = dict(held_K=K, gate_M_ge=m0, gate_r_ge=r0,
                   train=eval_choices(gate, tr),
                   test_gate=eval_choices(gate, te),
                   test_torch=eval_choices(lambda p: "torch", te),
                   test_hand=eval_choices(lambda p: "hand", te))
        fold_rows.append(row)
        for p in te:
            te_pairs.append((p, gate(p)))

    pooled = {}
    for nm in ("gate", "torch", "hand"):
        cf = {"gate": (lambda t: t[1]), "torch": (lambda p: "torch"),
              "hand": (lambda p: "hand")}[nm]
        sel = (lambda t: t[1]) if nm == "gate" else None
        if nm == "gate":
            pts_sel = [(p, g) for p, g in te_pairs]
            regs, eq, worse = [], 0, 0
            for p, g in pts_sel:
                t = p["t_hand"] if g == "hand" else p["t_torch"]
                regs.append((t - p["t_oracle"]) / p["t_oracle"])
                eq += t <= p["t_oracle"] * 1.001
                worse += t > p["t_torch"] * 1.01
        else:
            fn = (lambda p: "torch") if nm == "torch" else (lambda p: "hand")
            regs, eq, worse = [], 0, 0
            for p, _g in te_pairs:
                t = p["t_hand"] if fn(p) == "hand" else p["t_torch"]
                regs.append((t - p["t_oracle"]) / p["t_oracle"])
                eq += t <= p["t_oracle"] * 1.001
                worse += t > p["t_torch"] * 1.01
        regs.sort()
        n = len(regs)
        pooled[nm] = dict(mean_regret=sum(regs) / n, med_regret=regs[n // 2],
                          p90_regret=regs[min(n - 1, int(.9 * n))], max_regret=max(regs),
                          frac_oracle=eq / n, n_worse_than_torch_1pct=worse, n=n)

    agree = sum(1 for p, g in te_pairs if g == p["label"] or p["label"] == "tie")
    agree = agree / len(te_pairs)

    # correctness audit
    v3 = json.load(open(V3))
    acc = {}
    for row in v3["rows"]:
        for nm, v in row["variants"].items():
            if "med_ms" not in v:
                continue
            a = acc.setdefault(nm, [])
            a.append((v["max_abs_err"], v["max_abs_err"] / row["K"] ** .5))
    audit = {}
    for nm, pairs in acc.items():
        errs = sorted(p[0] for p in pairs)
        rels = sorted(p[1] for p in pairs)
        audit[nm] = dict(med_err=errs[len(errs) // 2], max_err=max(errs),
                         med_rel=rels[len(rels) // 2], n=len(pairs))

    out = dict(
        meta=dict(date=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  op_class="reduction column-sum (K,M) fp32",
                  data=["e2_splitk_v3.json", "e2b_floor.json (launch const)"],
                  t_launch_two_us=t2, n_points=len(pts), Ks=Ks,
                  pre_reg="module docstring; diary 2026-10-17 ~18:20Z"),
        P2=dict(rule="hand(tiled-split KT64 ctas~64) iff M>=m0 and t_torch/t_launch2>=r0 "
                     "else torch; (m0,r0) refit per LOO-K fold",
                domain_predicate="K 2^12..2^22, M 1..512, fp32, GB10 sm_120 column-sum; "
                                 "M-extrapolation and dtype axes untested",
                derived_coords=["lgK", "lgM", "r=t_torch/t_launch2", "ctas/48"],
                boundary_ref=["results/e2b_floor.json", "results/e2_splitk_v3.json"],
                variance="iqr in source rows; n_reps=12"),
        folds=fold_rows, pooled_loo=pooled, label_agreement=agree,
        correctness_audit=audit)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)

    print("t_launch_two=%.3f us | %d pts | %d LOO-K folds" % (t2, len(pts), len(Ks)))
    print("\nPOOLED leave-one-K-out:")
    for nm in ("gate", "torch", "hand"):
        p = pooled[nm]
        print(" %-6s regret mean %.3f med %.3f p90 %.3f max %.3f | ==oracle %.0f%% | worse torch+1%%: %d/%d"
              % (nm, p["mean_regret"], p["med_regret"], p["p90_regret"], p["max_regret"],
                 100 * p["frac_oracle"], p["n_worse_than_torch_1pct"], p["n"]))
    print("\nPer-fold (held-out K): gate(M>=%d? no--) gate | test gate/torch/hand mean regret | worse-than-torch")
    for r in fold_rows:
        print(" K=%8d gate(M>=%2d,r>=%.2f) tr %.4f | te g %.4f t %.4f h %.4f | %d/%d"
              % (r["held_K"], r["gate_M_ge"], r["gate_r_ge"], r["train"]["mean_regret"],
                 r["test_gate"]["mean_regret"], r["test_torch"]["mean_regret"],
                 r["test_hand"]["mean_regret"],
                 r["test_gate"]["n_worse_than_torch_1pct"], r["test_gate"]["n"]))
    print("\nlabel agreement vs noise-gated (ties=agree): %.2f" % agree)
    print("\ncorrectness audit:")
    for nm in sorted(audit):
        a = audit[nm]
        print(" %-18s med_err %9.4f max %10.4f med_rel %.2e n=%d"
              % (nm, a["med_err"], a["max_err"], a["med_rel"], a["n"]))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
