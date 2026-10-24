"""F2/F3/F1 verification for rejection 2CmvAb5SC -- tick #44. CPU only.

F2: recount every disputed pass count on BOTH populations (r1 as-run; corrected
    union per r42_analysis) with the recorded predicates (pass_1e4 default-ref,
    pass32 fp32/ieee-ref); torch_cmp failing rows vs their inclusion in paired d_i.
F3: rider gate_stability.jsonl -- torch unconditional admission, delivered-failing
    counts per anchor, bootstrap-1.0 vs anchor-flip, obsolete shapes, tolerance
    direction realized as rider-vs-primary verdict disagreement.
F1: regenerate breakeven with the correct filter t_b < t_gate, v*=probes/(t_gate-t_b);
    contrast with published (inverted) convention counts.

Output: results/f24_verify.json + printed digest.
"""
import json, os, statistics, time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
WRONG3 = {("w4", 2, 3, 5), ("w4", 2, 3, 12), ("a2", 2, 4, 9), ("a2", 2, 4, 20)}
OUT = os.path.join(RES, "f24_verify.json")


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def key4(r):
    return (r["tag"], r["N"], r["H"], r["C"])


def key3(r):
    return (r["tag"], r["N"], r["H"], r["C"])  # stored order tag,N,H,C


def populations():
    r1 = load(os.path.join(RES, "r1_conv_gated.jsonl"))
    gf = load(os.path.join(RES, "r42_gridfix.jsonl"))
    corr = [r for r in r1 if key4(r) not in WRONG3] + gf
    return r1, corr


def imp_arms(r):
    return {k: v for k, v in r["variants"].items() if k.startswith("imp_")}


def count_passes(rows, flag, prec=None):
    n = 0
    for r in rows:
        for k, v in imp_arms(r).items():
            if prec and not k.endswith("[%s]" % prec):
                continue
            if v.get(flag):
                n += 1
    return n


def adm_rows_by_tag(rows):
    counts, denoms = {}, {}
    for r in rows:
        ok = any(v.get("pass32") for k, v in imp_arms(r).items() if k.endswith("[ieee]"))
        counts[r["tag"]] = counts.get(r["tag"], 0) + (1 if ok else 0)
        denoms[r["tag"]] = denoms.get(r["tag"], 0) + 1
    return counts, denoms


def f2(r1, corr):
    res = dict(
        hand_pass_default={"r1_asrun": count_passes(r1, "pass_1e4"),
                           "corrected": count_passes(corr, "pass_1e4")},
        ieee_pass_default={"r1_asrun": count_passes(r1, "pass_1e4", "ieee"),
                           "corrected": count_passes(corr, "pass_1e4", "ieee")},
        tf32_pass_default={"r1_asrun": count_passes(r1, "pass_1e4", "tf32"),
                           "corrected": count_passes(corr, "pass_1e4", "tf32")},
        ieee_pass_fp32ref={"r1_asrun": count_passes(r1, "pass32", "ieee"),
                           "corrected": count_passes(corr, "pass32", "ieee")},
        tf32_pass_fp32ref={"r1_asrun": count_passes(r1, "pass32", "tf32"),
                           "corrected": count_passes(corr, "pass32", "tf32")},
    )
    a_old, d_old = adm_rows_by_tag(r1)
    a_new, d_new = adm_rows_by_tag(corr)
    res["adm_ieee_fp32ref_rows"] = {"r1_asrun": a_old, "corrected": a_new,
                                    "denoms_r1asrun": d_old, "denoms_corrected": d_new}
    fail = [key4(r) for r in corr
            if r["variants"].get("torch_cmp", {}).get("pass_1e4") is False]
    by_tag = {}
    for t, N, H, C in fail:
        by_tag.setdefault(t, []).append((N, H, C))
    ra = json.load(open(os.path.join(RES, "r42_analysis.json")))
    d_rows = {(p["tag"], p["N"], p["H"], p["C"]) for p in ra["per_row"]
              if p["d_i_pct"] is not None}
    res["torch_cmp_failing"] = {
        "n": len(fail), "by_tag": by_tag, "rows": fail,
        "included_in_paired_d_i": sum(1 for k in fail if k in d_rows)}
    return res


def f3():
    rows = load(os.path.join(RES, "gate_stability.jsonl"))
    summ = json.load(open(os.path.join(RES, "gate_stability_summary.json")))
    n = len(rows)
    torch_fail64 = sum(1 for r in rows if not r["pass_64"].get("torch", True))
    deliv64, delivown = {"def": 0, "ieee": 0}, {"def": 0, "ieee": 0}
    for r in rows:
        for anc in ("def", "ieee"):
            arm = r["gate_%s" % anc]["arm"]
            if not r["pass_64"].get(arm, False):
                deliv64[anc] += 1
            if not r["pass_%s" % anc].get(arm, False):
                delivown[anc] += 1
    obsolete = [key4(r) for r in rows if key4(r) in WRONG3]
    r1 = {key4(r): r for r in load(os.path.join(RES, "r1_conv_gated.jsonl"))}
    dis = agree = 0
    for r in rows:
        prim = r1.get(key4(r))
        if not prim:
            continue
        for arm, pv in r["pass_def"].items():
            if not arm.startswith("imp_"):
                continue
            p1 = prim["variants"].get(arm, {}).get("pass_1e4")
            if p1 is None:
                continue
            if bool(pv) != bool(p1):
                dis += 1
            else:
                agree += 1
    return dict(
        n_rows=n,
        torch_fails_fp64_anchor=torch_fail64,
        anchor_flip_frac_torch=round(torch_fail64 / n, 4),
        gate_delivers_arm_failing_fp64=deliv64,
        gate_delivers_arm_failing_own_ref=delivown,
        bootstrap_torch_flip_median=summ.get("boot_arm_flip_frac_median", {}).get("torch"),
        n_obsolete_shapes=len(obsolete), obsolete_shape_rows=obsolete,
        rider_vs_primary_pass_cells=dict(agree=agree, disagree=dis),
    )


def f1():
    rows = load(os.path.join(RES, "m3_r42.jsonl"))
    BUDGETS = [1, 2, 4, 8]
    out = {}
    for fam in ("reduction", "conv"):
        rs = [r for r in rows if r["family"] == fam]
        old = {b: [] for b in BUDGETS}
        new = {b: [] for b in BUDGETS}
        conv_struct = 0
        for r in rs:
            arms = r["arms"]
            t_g = arms["gate"]["t"]
            if fam == "conv":
                gt = str(arms["gate"].get("choice", "")).split("|")[0]
                bt = [str(arms["b%d" % b].get("choice, " "") if False else
                          arms["b%d" % b].get("choice", "")).split("|")[0] for b in BUDGETS]
                if gt == "torch" and all(c == "torch" for c in bt):
                    conv_struct += 1
            for b in BUDGETS:
                a = arms["b%d" % b]
                if a["probes"] > 0:
                    if a["t"] > t_g:
                        old[b].append(a["probes"] / (a["t"] - t_g))
                    if a["t"] < t_g:
                        new[b].append(a["probes"] / (t_g - a["t"]))
        out[fam] = dict(
            n_rows=len(rs),
            published_conv_t_gt_gate={("b%d" % b): dict(
                n_finite=len(v), median=(round(statistics.median(v), 4) if v else None))
                for b, v in sorted(old.items())},
            corrected_t_lt_gate={("b%d" % b): dict(
                n_finite=len(v), median=(round(statistics.median(v), 4) if v else None))
                for b, v in sorted(new.items())},
            conv_all_torch_struct=conv_struct if fam == "conv" else None,
        )
    out["_totals"] = dict(
        published_style_finite=sum(v["n_finite"] for fam in ("reduction", "conv")
                                   for v in out[fam]["published_conv_t_gt_gate"].values()),
        corrected_finite=sum(v["n_finite"] for fam in ("reduction", "conv")
                             for v in out[fam]["corrected_t_lt_gate"].values()))
    return out


def main():
    r1, corr = populations()
    assert len(corr) == 56, "union must be 56, got %d" % len(corr)
    out = dict(date=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               f2=f2(r1, corr), f3=f3(), f1=f1())
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
