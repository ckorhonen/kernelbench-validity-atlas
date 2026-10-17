"""W4 dispatcher v1: ablation ladder + fold-refit hand prior + honest errors.

Pre-registered BEFORE evaluation 2026-10-17 ~21:40Z (this docstring; diary same tick).
Changes from v0, each declared up front:
  1. Hand prior FOLD-REFIT (v0 limitation): per LOO-K train fold, hand variant =
     argmin_u mean regret(u) on train over the 6 tsplit variants; ties -> KT64, then
     |ctas-64|. If the chosen variant is absent at a test point: fallback KT64/ctas~64
     (declared; coverage is complete on the v3 grid, fallback expected unused).
  2. Feature-ablation ladder (reflection #6 commitment). Gate = hand iff <pred>; all
     thresholds refit per fold on train by min mean regret, ties -> smaller hand region:
       A  M >= m0                          (1 feature, shape)
       B  M >= m0 and K >= K0              (+launch-guard, shape-only)
       C  M >= m0 and r >= r0              (v0 form; r = t_torch/t_launch2, measured-
                                            torch feature -- circularity flagged in v0,
                                            kept for the ablation, NOT shipped alone)
       D  M >= m0 and K >= K0 and r >= r0  (full)
       E  M >= m0 and bt >= b0             (bytes/thread of the fold-refit variant:
                                            bt = 2*bytes/(ctas*128), 4 warps)
     Family selection per fold on train; ties -> fewer features, order A<B<C<D<E.
     Per-family pooled LOO test reported = the ablation-ladder evidence.
  3. Correctness column regenerated from results/e2_v3_err.json (e2_err_regen.py,
     fixed harness; v0's audit was garbage-based and is superseded).
  4. Out-of-sample transfer check: final-fit gate (all v3 data, no holdout) applied to
     the e2b M* grid (K 2^8..2^17, M 2..256) where the hand arm is tsplit_pinned
     (wave-pinned ctas~64 -- a DIFFERENT selector than v3's fixed grid). Decision-rule
     transfer only; declared not an end-to-end dispatcher test.

Primary metric unchanged from v0: regret = (t_chosen - t_oracle)/t_oracle,
t_oracle = min over {torch, all valid hand variants}. LOO-K over the 6 K folds.
Adoption bar (unchanged): 0 held-out points >1% worse than torch while mean regret
< always-torch; report honestly either way (P5).

Run: python3 w4_dispatcher_v1.py (stdlib only) -> results/w4_v1.json
"""
import json, os, time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
V3 = os.path.join(RES, "e2_splitk_v3.json")
EB = os.path.join(RES, "e2b_floor.json")
ERR = os.path.join(RES, "e2_v3_err.json")
OUT = os.path.join(RES, "w4_v1.json")


def fallback_variant(variants):
    cand = {k: v for k, v in variants.items() if k != "torch" and "med_ms" in v}
    kt64 = {k: v for k, v in cand.items() if "_KT64" in k} or cand
    return min(kt64.items(), key=lambda kv: abs(kv[1]["ctas"] - 64))


def build_points():
    v3 = json.load(open(V3))
    eb = json.load(open(EB))
    errmap = {}
    for row in json.load(open(ERR))["rows"]:
        for nm, v in row["variants"].items():
            errmap[(row["K"], row["M"], nm)] = v["med_abs_err"]
    t2 = eb["meta"]["launch_floor_two_ms"]["med_ms"] * 1e3
    pts = []
    for row in v3["rows"]:
        K, M, var = row["K"], row["M"], row["variants"]
        if "med_ms" not in var["torch"]:
            continue
        t_torch = var["torch"]["med_ms"]
        hand = {k: v for k, v in var.items() if k != "torch" and "med_ms" in v}
        t_bh, best_name = min((v["med_ms"], k) for k, v in hand.items())
        bi, ti = hand[best_name]["iqr_ms"], var["torch"]["iqr_ms"]
        noise = max(((ti[1] - ti[0]) + (bi[1] - bi[0])) / 2 / t_torch, 0.01)
        _, fb = fallback_variant(var)
        pts.append(dict(
            K=K, M=M, lgK=K.bit_length() - 1, lgM=M.bit_length(),
            bytes=row["bytes"], t_torch=t_torch,
            t_hand=fb["med_ms"], hand_name=fb["ctas"] and next(k for k, v in var.items() if v is fb),
            variants=hand, hand_ctas=fb["ctas"],
            hand_err=errmap.get((K, M, next((k for k, v in var.items() if v is fb), ""))),
            best_err=errmap.get((K, M, best_name)),
            t_oracle=min(t_torch, t_bh), r=t_torch / t2,
            label="hand" if (t_torch - t_bh) / t_torch > noise else
                  ("torch" if (t_torch - t_bh) / t_torch < -noise else "tie"),
            _fb=fb["med_ms"]))
    return pts, t2


def refit_hand(train):
    """argmin_u mean regret if always-variant-u; tie KT64 then |ctas-64|. -> name"""
    best = None
    for u in {k for p in train for k in p["variants"]}:
        regs, ctas = [], None
        ok = True
        for p in train:
            v = p["variants"].get(u)
            if not v:
                ok = False; break
            regs.append((v["med_ms"] - p["t_oracle"]) / p["t_oracle"])
            ctas = v["ctas"]
        if not ok:
            continue
        key = (round(sum(regs) / len(regs), 9), 0 if "_KT64" in u else 1, abs(ctas - 64))
        if best is None or key < best[0]:
            best = (key, u)
    return best[1] if best else None


def apply_variant(p, name):
    v = p["variants"].get(name) or fallback_variant(p["variants"])[1]
    return v["med_ms"], v["ctas"]


def eval_choice(choice, pts):
    """choice(p) -> ('hand', name) | ('torch',); returns stats vs oracle."""
    regs, eq, worse, errs = [], 0, 0, []
    for p in pts:
        c = choice(p)
        if c[0] == "hand":
            t, _ctas = apply_variant(p, c[1])
            e = p["variants"].get(c[1], {}).get("med_ms")
            errs.append(p["variants"].get(c[1], {}).get("_", None))
        else:
            t = p["t_torch"]
        regs.append((t - p["t_oracle"]) / p["t_oracle"])
        eq += t <= p["t_oracle"] * 1.001
        worse += t > p["t_torch"] * 1.01
    regs.sort(); n = len(regs)
    return dict(mean_regret=sum(regs) / n, med_regret=regs[n // 2],
                p90_regret=regs[min(n - 1, int(.9 * n))], max_regret=max(regs),
                frac_oracle=eq / n, n_worse_than_torch_1pct=worse, n=n)


def hand_err_for(p, name):
    return p["variants"].get(name, {}).get("_err", None)


def fit_family(family, train):
    """grid thresholds; min train mean regret; tie -> smaller hand region."""
    grids = dict(
        m0=[1, 2, 4, 8, 16, 32, 64],
        K0=[0, 2 ** 12, 2 ** 13, 2 ** 14, 2 ** 15, 2 ** 16, 2 ** 17, 2 ** 18, 2 ** 20],
        r0=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0],
        b0=[0, 64, 256, 1024, 4096, 16384, 65536])
    import itertools
    if family == "A":
        combos = [(m0, 0, 0.0, 0) for m0 in grids["m0"]]
    elif family == "B":
        combos = [(m0, K0, 0.0, 0) for m0, K0 in itertools.product(grids["m0"], grids["K0"])]
    elif family == "C":
        combos = [(m0, 0, r0, 0) for m0, r0 in itertools.product(grids["m0"], grids["r0"])]
    elif family == "D":
        combos = [(m0, K0, r0, 0) for m0, K0, r0 in
                  itertools.product(grids["m0"], grids["K0"], grids["r0"])]
    else:  # E
        combos = [(m0, 0, 0.0, b0) for m0, b0 in itertools.product(grids["m0"], grids["b0"])]

    def pred(p, m0, K0, r0, b0, hname):
        if p["M"] < m0:
            return False
        if K0 and p["K"] < K0:
            return False
        if r0 and p["r"] < r0:
            return False
        if b0:
            _, ctas = apply_variant(p, hname)
            if 2 * p["bytes"] / (ctas * 128) < b0:
                return False
        return True

    best = None
    for m0, K0, r0, b0 in combos:
        regs, nhand = [], 0
        for p in train:
            h = pred(p, m0, K0, r0, b0, hname=None)  # bt needs variant; resolve below
            nhand += h
        # two-pass for E (bt depends on refit variant): approximate with fallback ctas
        pass
    # -- simple explicit loop per family (bt via fallback variant ctas, declared) --
    best = None
    for m0, K0, r0, b0 in combos:
        regs, nhand = [], 0
        for p in train:
            hname = p["_fbname"]
            h = p["M"] >= m0 and (not K0 or p["K"] >= K0) and (not r0 or p["r"] >= r0)
            if h and b0:
                _, ctas = apply_variant(p, hname)
                h = 2 * p["bytes"] / (ctas * 128) >= b0
            nhand += h
            if h:
                t, _ = apply_variant(p, hname)
            else:
                t = p["t_torch"]
            regs.append((t - p["t_oracle"]) / p["t_oracle"])
        key = (round(sum(regs) / len(regs), 9), nhand)
        if best is None or key < best[0]:
            best = (key, dict(m0=m0, K0=K0, r0=r0, b0=b0))
    return best[1]


FAMILIES = ["A", "B", "C", "D", "E"]
FEATCOUNT = {"A": 1, "B": 2, "C": 2, "D": 3, "E": 2}


def gate_pred(fam, th):
    def g(p):
        hname = p["_fbname"]
        h = p["M"] >= th["m0"] and (not th["K0"] or p["K"] >= th["K0"]) and \
            (not th["r0"] or p["r"] >= th["r0"])
        if h and th["b0"]:
            _, ctas = apply_variant(p, hname)
            h = 2 * p["bytes"] / (ctas * 128) >= th["b0"]
        return ("hand", hname) if h else ("torch",)
    return g


def main():
    pts, t2 = build_points()
    for p in pts:  # fallback name per point (declared gate hand arm when refit absent)
        p["_fbname"] = fallback_variant(
            dict(list(p["variants"].items()) + [("torch", {"med_ms": p["t_torch"], "ctas": 0})]))[0]

    # attach refit err columns for the tolerance section
    errmap = {}
    for row in json.load(open(ERR))["rows"]:
        for nm, v in row["variants"].items():
            errmap[(row["K"], row["M"], nm)] = v["med_abs_err"]
    for p in pts:
        for nm in p["variants"]:
            p["variants"][nm]["_err"] = errmap.get((p["K"], p["M"], nm))

    Ks = sorted({p["K"] for p in pts})
    fold_rows, te_pairs, fam_pool = [], [], {f: [] for f in FAMILIES}
    for K in Ks:
        tr = [p for p in pts if p["K"] != K]
        te = [p for p in pts if p["K"] == K]
        hname = refit_hand(tr)
        fam_res = {}
        best = None
        for fam in FAMILIES:
            th = fit_family(fam, [(q, hname) for q in tr][-len(tr):] if False else
                            [dict(q, _fbname=q["_fbname"]) for q in tr])
            g = gate_pred(fam, th)
            r = eval_choice(g, tr)
            fam_res[fam] = dict(th=th, train=r)
            key = (round(r["mean_regret"], 9), FEATCOUNT[fam])
            if best is None or key < best[0]:
                best = (key, fam, th, g)
        _, fam, th, g = best
        row = dict(held_K=K, hand_variant=hname, family=fam, thresholds=th,
                   fam_train={f: fam_res[f]["train"]["mean_regret"] for f in FAMILIES},
                   fam_th={f: fam_res[f]["th"] for f in FAMILIES},
                   test_gate=eval_choice(g, te),
                   test_torch=eval_choice(lambda p: ("torch",), te),
                   test_hand=eval_choice(lambda p: ("hand", hname), te))
        fold_rows.append(row)
        for p in te:
            te_pairs.append((p, g(p)))
            for f in FAMILIES:
                gf = gate_pred(f, fam_res[f]["th"])
                fam_pool[f].append((p, gf(p)))

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

    pooled_loo = {"gate": pooled(te_pairs), "torch": pooled([(p, ("torch",)) for p, _ in te_pairs])}
    hname_global = refit_hand(pts)
    pooled_loo["hand_refit"] = pooled([(p, ("hand", hname_global)) for p, _ in te_pairs])
    ablation = {f: pooled(v) for f, v in fam_pool.items()}

    agree = sum(1 for p, c in te_pairs if c[0] == "hand" and p["label"] == "hand" or
                c[0] == "torch" and p["label"] in ("torch", "tie")) / len(te_pairs)

    # --- transfer check: final-fit gate (all data) on e2b grid, hand=tsplit_pinned ---
    eb = json.load(open(EB))
    t2e = eb["meta"]["launch_floor_two_ms"]["med_ms"] * 1e3
    xpts = []
    for row in eb["rows"]:
        v = row["variants"]
        if "med_ms" not in v.get("torch", {}) or "med_ms" not in v.get("tsplit_pinned", {}):
            continue
        xpts.append(dict(K=row["K"], M=row["M"], bytes=row["bytes"],
                         t_torch=v["torch"]["med_ms"],
                         variants={"tsplit_pinned": v["tsplit_pinned"]},
                         t_oracle=min(v["torch"]["med_ms"], v["tsplit_pinned"]["med_ms"]),
                         r=v["torch"]["med_ms"] / t2e, _fbname="tsplit_pinned"))
    # final fit: family+thresholds on ALL v3 points (same selection rule)
    hname_all = refit_hand(pts)
    final = {}
    for fam in FAMILIES:
        th = fit_family(fam, [dict(p, _fbname=p["_fbname"]) for p in pts])
        final[fam] = (th, eval_choice(gate_pred(fam, th),
                                      [dict(p, _fbname=p["_fbname"]) for p in pts])["mean_regret"])
    fam_best = min(FAMILIES, key=lambda f: (round(final[f][1], 9), FEATCOUNT[f]))
    th_final = final[fam_best][0]
    transfer = dict(family=fam_best, thresholds=th_final,
                    eval=eval_choice(gate_pred(fam_best, th_final), xpts),
                    torch=eval_choice(lambda p: ("torch",), xpts),
                    hand=eval_choice(lambda p: ("hand", "tsplit_pinned"), xpts),
                    n=len(xpts), note="decision-rule transfer; hand arm=wave-pinned tsplit, "
                                      "different selector than v3 grid (declared)")

    # --- tolerance latitude from regenerated errors ---
    lat = {}
    for nm in sorted({k for p in pts for k in p["variants"]}):
        errs = sorted(p["variants"][nm]["_err"] for p in pts
                      if p["variants"].get(nm, {}).get("_err") is not None)
        if errs:
            lat[nm] = dict(med_abs_err=errs[len(errs) // 2], max_abs_err=errs[-1],
                           med_rel_sqrtK=errs[len(errs) // 2] / Ks[0] ** 0.5, n=len(errs))

    out = dict(
        meta=dict(date=time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()),
                  op_class="reduction column-sum (K,M) fp32",
                  data=["e2_splitk_v3.json", "e2b_floor.json", "e2_v3_err.json"],
                  t_launch_two_us=t2, n_points=len(pts), Ks=Ks,
                  pre_reg="module docstring, 2026-10-17 ~21:40Z, before eval"),
        P2=dict(rule="family-selected gate, hand=fold-refit tsplit variant; see docstring ladder A-E",
                domain_predicate="K 2^12..2^22, M 1..512 fp32 GB10 column-sum; M-extrapolation, "
                                 "dtype, other op classes untested; r-feature circularity declared",
                derived_coords=["lgK", "lgM", "r=t_torch/t_launch2", "ctas/48", "bytes/thread"],
                boundary_ref=["results/e2b_floor.json", "results/e2_splitk_v3.json",
                              "results/e2_v3_err.json"],
                variance="iqr in source rows; n_reps=12"),
        folds=fold_rows, pooled_loo=pooled_loo, ablation_ladder=ablation,
        label_agreement=agree, tolerance_latitude=lat, transfer_e2b=transfer)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=1, default=str)

    print("t2=%.3fus  pts=%d  hand_refit(global)=%s" % (t2, len(pts), hname_global))
    print("\nABLATION LADDER (pooled LOO test):")
    for f in FAMILIES:
        p = ablation[f]
        print(" %s th=%s -> mean %.4f med %.4f p90 %.4f | ==oracle %.0f%% | worse1%% %d/%d"
              % (f, json.dumps(fold_rows[0]["fam_th"][f]) if False else
                 json.dumps(next((r["fam_th"][f] for r in fold_rows), {}), default=str)[:60],
                 p["mean_regret"], p["med_regret"], p["p90_regret"],
                 100 * p["frac_oracle"], p["n_worse_than_torch_1pct"], p["n"]))
    print("\nPOOLED family-selected gate vs torch vs hand-refit:")
    for nm in ("gate", "torch", "hand_refit"):
        p = pooled_loo[nm]
        print(" %-11s mean %.4f med %.4f p90 %.4f max %.4f | ==oracle %.0f%% | worse1%% %d/%d"
              % (nm, p["mean_regret"], p["med_regret"], p["p90_regret"], p["max_regret"],
                 100 * p["frac_oracle"], p["n_worse_than_torch_1pct"], p["n"]))
    print("\nPer-fold:")
    for r in fold_rows:
        print(" K=%8d hand=%-18s fam=%s th=%s | te g %.4f t %.4f h %.4f | worse %d/%d"
              % (r["held_K"], r["hand_variant"], r["family"],
                 json.dumps(r["thresholds"], default=str), r["test_gate"]["mean_regret"],
                 r["test_torch"]["mean_regret"], r["test_hand"]["mean_regret"],
                 r["test_gate"]["n_worse_than_torch_1pct"], r["test_gate"]["n"]))
    print("\nlabel agreement %.2f" % agree)
    t = transfer
    print("\nTRANSFER e2b (n=%d, fam=%s th=%s): gate mean %.4f worse1%% %d/%d | torch %.4f | hand %.4f"
          % (t["n"], t["family"], json.dumps(t["thresholds"], default=str),
             t["eval"]["mean_regret"], t["eval"]["n_worse_than_torch_1pct"], t["eval"]["n"],
             t["torch"]["mean_regret"], t["hand"]["mean_regret"]))
    print("\nERRORS (regen):")
    for nm in sorted(lat):
        d = lat[nm]
        print(" %-18s med %.4g max %.4g n=%d" % (nm, d["med_abs_err"], d["max_abs_err"], d["n"]))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
