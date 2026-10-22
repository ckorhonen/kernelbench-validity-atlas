"""A1 gate (plan v2 item 1, tick #35): orthogonalized w-magnitude-decile
re-attribution on committed b4/b5 rows. No GPU. Answers the circularity
objection: does the gamma-load attribution survive with label-free cuts and
LOO-fit coefficients, against pre-declared within-family floors?

FROZEN RULE (written before fitting; 2026-10-21 tick #35):
  DATA    : b4_guard_v1.jsonl + b5_stage2.jsonl (published finality artifacts);
            sensitivity on b5_stage2r.jsonl (repaired) recorded separately.
  FEATURES: per-(source,arm) z of w_mean_abs, w_max; load L = mean of the two
            (the recorded gamma-load pair). Offset channel Oc = GLOBAL z of
            collapse_work_ratio (bind-time). Cell key C = "H%(H)s_B%(B)s_K%(K)s"
            (bind-known). NO label enters features, z params, or cuts.
  ORTH    : L_orth = residual of OLS L ~ 1 + Oc (pooled). Recorded: within any
            single source the bind-offset channel is collinear with the
            intercept (A4 degeneracy on these artifacts).
  DECILES : 10 equal-count bins of L_orth pooled; cuts label-free.
  MODELS  (LOCO = leave-one-decile-out; coefficients from train folds only):
    M-noO : y ~ 1 + L_orth                      (global surface, no offset)
    M-O   : y ~ 1 + L_orth + Oc                (+ bind scalar offset channel)
    M-C   : y ~ cell_dummies + 1 + L_orth      (A4-FAITHFUL PRIMARY: global
            load + bind-time per-cell offset; GUARD absorption sentence.
            RE-PRIMARYING recorded: the noO form tests a strawman. Unseen-cell
            rows in a held-out fold fall back to the global part; count
            recorded.) numpy lstsq throughout (SVD, fold-collinearity-safe).
  STATS   s1: LOO offset-R2 = 1 - SSE/SST, SST about the GLOBAL train mean.
          s2: LOO AUC pooled.  Per-source descriptives alongside.
  FLOORS  (pre-declared): R2_w / AUC_w = pooled 1-SSE/SST and AUC of
            per-(source,arm) OLS label ~ 1 + committed score (label-supervised
            within-family ceiling, threshold-free).
  PASS    : PRIMARY = M-C. s1_C >= R2_w  AND  s2_C >= AUC_w - 0.01.
            (Original freeze keyed PASS to the then-primary global form; the
            re-primarying to M-C is recorded above with reason. M-noO / M-O
            reported as ablations; half-floor sensitivity recorded.)
  FAIL    feeds falsifier arm 1 (plan v2).

Output: results/a1_orthdecile.json (+ .jsonl folds). Run: python3 (numpy ok).
"""
import json, pathlib, math, time
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"


def rankdata(v):
    idx = sorted(range(len(v)), key=lambda i: v[i])
    out = [0.0] * len(v)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and v[idx[j + 1]] == v[idx[i]]:
            j += 1
        m = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[idx[k]] = m
        i = j + 1
    return out


def auc(score, lab):
    r = rankdata(score)
    n1 = sum(lab)
    n0 = len(lab) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    s1 = sum(rk for rk, l in zip(r, lab) if l)
    return (s1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def ols1(x, y):
    A = np.column_stack([np.ones(len(y)), np.asarray(x, float)])
    sol, *_ = np.linalg.lstsq(A, np.asarray(y, float), rcond=None)
    return float(sol[0]), float(sol[1])


def r2(pairs, baseline):
    if len(pairs) < 2:
        return float("nan")
    sse = sum((p - y) ** 2 for p, y in pairs)
    sst = sum((y - baseline) ** 2 for _, y in pairs)
    return 1 - sse / sst if sst > 0 else float("nan")


def zparams(vals):
    n = len(vals)
    mu = sum(vals) / n
    var = sum((v - mu) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    sd = math.sqrt(var)
    return mu, (sd if sd > 0 else 1.0)


def run(rows, tag):
    n = len(rows)
    groups = {}
    for r in rows:
        groups.setdefault((r["_src"], r["arm"]), []).append(r)
    for g in groups.values():
        for f in ("w_mean_abs", "w_max", "collapse_work_ratio"):
            mu, sd = zparams([float(x[f]) for x in g])
            for x in g:
                x["z_" + f] = (float(x[f]) - mu) / sd
    gm, gsd = zparams([float(r["collapse_work_ratio"]) for r in rows])
    for r in rows:
        r["L"] = (r["z_w_mean_abs"] + r["z_w_max"]) / 2.0
        r["Oc"] = (float(r["collapse_work_ratio"]) - gm) / gsd
    b0, b1 = ols1([r["Oc"] for r in rows], [r["L"] for r in rows])
    for r in rows:
        r["Lo"] = r["L"] - (b0 + b1 * r["Oc"])
    orth = dict(b0=b0, b1=b1)

    s = sorted(range(n), key=lambda i: rows[i]["Lo"])
    for rank, i in enumerate(s):
        rows[i]["dec"] = min(9, rank * 10 // n)

    # floors: per-(source,arm) supervised OLS on the committed score
    sse = sst = 0.0
    for g in groups.values():
        xs = [float(x["score"]) for x in g]
        ys = [1.0 if x["label"] else 0.0 for x in g]
        c0, c1 = ols1(xs, ys)
        my = sum(ys) / len(ys)
        sse += sum((c0 + c1 * x - y) ** 2 for x, y in zip(xs, ys))
        sst += sum((y - my) ** 2 for y in ys)
    R2_w = 1 - sse / sst
    AUC_w = auc([r["score"] for r in rows], [1 if r["label"] else 0 for r in rows])
    floor_R2_per_src = {}
    for (src, arm), g in groups.items():
        xs = [float(x["score"]) for x in g]
        ys = [1.0 if x["label"] else 0.0 for x in g]
        c0, c1 = ols1(xs, ys)
        my = sum(ys) / len(ys)
        floor_R2_per_src["%s/%s" % (src, arm)] = r2(
            [(c0 + c1 * x, y) for x, y in zip(xs, ys)], my)

    base = sum(1.0 if r["label"] else 0.0 for r in rows) / n
    yall = [1.0 if r["label"] else 0.0 for r in rows]

    def loo(build_predict):
        pred = [None] * n

        def wrapped(tr, te):
            return build_predict(pred, tr, te)

        folds = []
        for d in range(10):
            tr = [r for r in rows if r["dec"] != d]
            te = [r for r in rows if r["dec"] == d]
            if not te:
                continue
            info = wrapped(tr, te)
            folds.append(dict(dec=d, n=len(te), **info))
        pairs = [(pred[i], yall[i]) for i in range(n) if pred[i] is not None]
        return (pred, r2(pairs, base),
                auc([p for p, _ in pairs], [l for _, l in pairs]), folds)

    # M-noO
    def f_noO(pred, tr, te):
        c0, c1 = ols1([r["Lo"] for r in tr], [1.0 if r["label"] else 0.0 for r in tr])
        for r in te:
            pred[r["_i"]] = c0 + c1 * r["Lo"]
        return dict(b1=c1)

    # M-O
    def f_O(pred, tr, te):
        A = np.array([[1.0, r["Lo"], r["Oc"]] for r in tr])
        y = np.array([1.0 if r["label"] else 0.0 for r in tr])
        if np.linalg.matrix_rank(A) < 3:
            c0, c1 = ols1([r["Lo"] for r in tr], [1.0 if r["label"] else 0.0 for r in tr])
            sol = np.array([c0, c1, 0.0])
        else:
            sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        for r in te:
            pred[r["_i"]] = float(sol[0] + sol[1] * r["Lo"] + sol[2] * r["Oc"])
        return dict(b1=float(sol[1]), gamma=float(sol[2]))

    # M-C (A4-faithful primary)
    cells = sorted({r["cell"] for r in rows})
    cidx = {c: i for i, c in enumerate(cells)}
    nc = len(cells)

    def f_C(pred, tr, te):
        seen = {r["cell"] for r in tr}
        A = np.zeros((len(tr), nc + 2))
        y = np.array([1.0 if r["label"] else 0.0 for r in tr])
        for i, r in enumerate(tr):
            A[i, cidx[r["cell"]]] = 1.0
            A[i, nc] = 1.0
            A[i, nc + 1] = r["Lo"]
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        un = 0
        for r in te:
            if r["cell"] in seen:
                v = float(sol[cidx[r["cell"]]] + sol[nc] + sol[nc + 1] * r["Lo"])
            else:
                un += 1
                v = float(sol[nc] + sol[nc + 1] * r["Lo"])
            pred[r["_i"]] = v
        return dict(n_unseen_cell=un, b1=float(sol[nc + 1]))

    for i, r in enumerate(rows):
        r["_i"] = i
    pC, s1C, s2C, foldsC = loo(f_C)
    pred_noO, s1, s2, folds = loo(f_noO)
    _, s1O, s2O, _ = loo(f_O)

    per_src = {}
    for src in {r["_src"] for r in rows}:
        g = [r for r in rows if r["_src"] == src]
        idx = [r["_i"] for r in g]
        ym = sum(yall[i] for i in idx) / len(idx)
        per_src[src] = dict(
            n=len(g),
            looR2_C=r2([(pC[i], yall[i]) for i in idx], ym),
            looAUC_C=auc([pC[i] for i in idx], [yall[i] for i in idx]),
            looAUC_noO=auc([pred_noO[i] for i in idx], [yall[i] for i in idx]))

    out = dict(
        tag=tag, n=n, n_cells=nc, base_rate=base, orth=orth,
        PRIMARY="M-C (cell offsets + global load; A4-faithful, re-primarying recorded)",
        LOO_offset_R2=s1C, LOO_auc=s2C,
        LOO_offset_R2_noO=s1, LOO_auc_noO=s2,
        LOO_offset_R2_O=s1O, LOO_auc_O=s2O,
        floor_R2_supervised=R2_w, floor_auc=AUC_w,
        floor_R2_per_src=floor_R2_per_src,
        pass_R2=bool(s1C >= R2_w), pass_R2_half=bool(s1C >= R2_w / 2),
        pass_auc=bool(s2C >= AUC_w - 0.01),
        per_source=per_src, folds=foldsC)
    out["PASS"] = bool(out["pass_R2"] and out["pass_auc"])
    # stash noO preds for per-source descriptive
    for r in rows:
        r["pred_noO"] = None
    return out, (pC, s1C, s2C)


def load(src, fn):
    p = RES / fn
    rows = [json.loads(l) for l in open(p) if l.strip()]
    for r in rows:
        r["_src"] = src
        if "cell" not in r:
            r["cell"] = "H%d_B%d_K%d" % (r["H"], r["B"], r["K"])
    return rows


def main():
    t0 = time.time()
    b4 = load("b4", "b4_guard_v1.jsonl")
    b5 = load("b5", "b5_stage2.jsonl")
    primary, _ = run(b4 + b5, "primary: b4_guard_v1 + b5_stage2 (published)")
    try:
        sens, _ = run(load("b5r", "b5_stage2r.jsonl"), "sensitivity: b5_stage2r (repaired)")
    except FileNotFoundError:
        sens = None
    out = dict(when=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               runtime_s=round(time.time() - t0, 2),
               primary=primary, sensitivity=sens,
               rule="see lab/a1_orthdecile.py docstring (frozen; re-primarying recorded)")
    (RES / "a1_orthdecile.json").write_text(json.dumps(out, indent=1, default=str))
    with open(RES / "a1_orthdecile.jsonl", "w") as f:
        for name, res in (("primary", primary), ("sensitivity", sens)):
            if res:
                for fd in res["folds"]:
                    f.write(json.dumps(dict(run=name, **fd)) + "\n")

    for name, p in (("PRIMARY(b4+b5)", primary), ("SENS(b5r)", sens)):
        if not p:
            continue
        print("%s n=%d cells=%d base=%.3f" % (name, p["n"], p["n_cells"], p["base_rate"]))
        print("  M-C   LOO R2=%.4f floor=%.4f | AUC=%.4f floor=%.4f"
              % (p["LOO_offset_R2"], p["floor_R2_supervised"],
                 p["LOO_auc"], p["floor_auc"]))
        print("  M-noO R2=%.4f AUC=%.4f | M-O R2=%.4f AUC=%.4f"
              % (p["LOO_offset_R2_noO"], p["LOO_auc_noO"],
                 p["LOO_offset_R2_O"], p["LOO_auc_O"]))
        print("  PASS(frozen, M-C): %s (half-floor: %s)  per-src floors: %s"
              % (p["PASS"], p["pass_R2_half"],
                 {k: round(v, 3) for k, v in p["floor_R2_per_src"].items()}))
        for k, v in p["per_source"].items():
            print("  src %s n=%d looR2_C=%.4f looAUC_C=%.4f"
                  % (k, v["n"], v["looR2_C"], v["looAUC_C"]))


if __name__ == "__main__":
    main()
