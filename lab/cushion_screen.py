"""Cushion screen -- prereg lab/cushion_screen_prereg.md (thresholds frozen there).

Train: reduction pairs per cf3 CF3 construction. Transfer: rider def-vs-64 arm
verdicts with r1 scalar-err proxy. Sanity: b1. Conv: set-shrinkage corollary.
CPU only. Output results/cushion_screen.json.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
BAR = 1e-4
OUT = os.path.join(RES, "cushion_screen.json")


def R(name):
    return json.load(open(os.path.join(RES, name)))


def auc(xs, ys):
    """Mann-Whitney: P(x_flip > x_noflip) + 0.5 ties. ys 1=flip."""
    f = sorted(x for x, y in zip(xs, ys) if y)
    n = sorted(x for x, y in zip(xs, ys) if not y)
    if not f or not n:
        return None
    g = tie = 0
    for a in f:
        for b in n:
            g += 1 if a > b else 0
            tie += 1 if a == b else 0
    return (g + 0.5 * tie) / (len(f) * len(n))


def train():
    day1 = R("m3_curve.json")["rows"]
    r42 = [json.loads(l) for l in open(os.path.join(RES, "m3_r42.jsonl")) if l.strip()]
    day1_red = {(r["K"], r["M"]): r for r in day1 if r["family"] == "reduction"}
    r42_red = {(r["K"], r["M"]): r for r in r42 if r["family"] == "reduction"}
    xs, ys, x2s = [], [], []
    for key in set(day1_red) & set(r42_red):
        d1, d2 = day1_red[key], r42_red[key]
        for arm, e1 in d1["errs"].items():
            if arm not in d2["adm"]:
                continue
            p1 = e1 <= BAR
            p2 = bool(d2["adm"][arm]["passed"])
            xs.append(e1 / BAR)
            x2s.append(d2["adm"][arm]["err"] / BAR)
            ys.append(1 if p1 != p2 else 0)
    return xs, x2s, ys


def rider():
    r1 = {(r["tag"], r["N"], r["H"], r["C"]): r
          for r in map(json.loads, open(os.path.join(RES, "r1_conv_gated.jsonl"))) if True}
    rows = [json.loads(l) for l in open(os.path.join(RES, "gate_stability.jsonl")) if l.strip()]
    xs, ys, arms = [], [], []
    for r in rows:
        key = (r["tag"], r["N"], r["H"], r["C"])
        prim = r1.get(key)
        if not prim:
            continue
        for arm, pd in r["pass_def"].items():
            if not arm.startswith("imp_"):
                continue
            e = prim["variants"].get(arm, {}).get("max_abs_err")
            if e is None:
                continue
            xs.append(e / BAR)
            ys.append(1 if bool(pd) != bool(r["pass_64"].get(arm)) else 0)
            arms.append(arm)
    return xs, ys


def decile_c(xs, ys):
    xs_s = sorted(set(xs))
    qs = [xs_s[int(len(xs_s) * q / 10)] for q in range(10)]
    best = None
    for x in qs:
        sub = [y for xx, y in zip(xs, ys) if xx <= x]
        if sub and sum(sub) / len(sub) <= 0.05:
            best = x
    return best


def main():
    xs, x2s, ys = train()
    auc_train = auc(xs, ys)
    c_star = decile_c(xs, ys)
    flip_if_below = None
    if c_star is not None:
        sub = [y for x, y in zip(xs, ys) if x <= c_star]
        flip_if_below = sum(sub) / len(sub)
    rx, ry = rider()
    auc_rider = auc(rx, ry)
    rider_below = None
    if c_star is not None:
        sub = [y for x, y in zip(rx, ry) if x <= c_star]
        rider_below = dict(n=len(sub), flip_frac=(sum(sub) / len(sub) if sub else None))
    b1 = R("b1_gatezero_v2.json")
    b1_note = b1.get("verdict", b1) if isinstance(b1, dict) else str(b1)
    conv_x = []
    for r in R("m3_curve.json")["rows"]:
        if r["family"] != "conv":
            continue
        errs = r.get("errs") or {}
        for arm, e in errs.items():
            conv_x.append(e / BAR)
    conv_below = (sum(1 for x in conv_x if x <= c_star), len(conv_x)) if c_star is not None else None
    out = dict(
        date=__import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
        prereg="lab/cushion_screen_prereg.md (thresholds frozen pre-run)",
        train=dict(n=len(ys), flips=sum(ys), auc_x1=auc_train, c_star=c_star,
                   flip_rate_below_c=flip_if_below),
        rider=dict(n=len(ry), flips=sum(ry), auc=auc_rider, below_c=c_star and {
                   "n": len([1 for x in rx if x <= c_star]), **{"detail": None}}),
        rider_below_c=rider_below,
        b1_sanity=str(b1_note)[:200],
        conv_x_below_c=conv_below,
        conv_x_stats=dict(n=len(conv_x), max=(max(conv_x) if conv_x else None)),
    )
    verdict = []
    verdict.append("train AUC %.3f %s 0.70" % (auc_train, ">=" if auc_train and auc_train >= 0.70 else "<"))
    if auc_rider is not None:
        verdict.append("rider AUC %.3f %s 0.60" % (auc_rider, ">=" if auc_rider >= 0.60 else "<"))
    if c_star is not None:
        verdict.append("c*=%g trainP(flip|x<=c)=%s riderBelow=%s" % (
            c_star, None if flip_if_below is None else round(flip_if_below, 4),
            None if not rider_below or rider_below["flip_frac"] is None else round(rider_below["flip_frac"], 4)))
    out["verdict"] = "; ".join(v for v in verdict if v)
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
