"""D1 — cell-affine decomposition posthoc on COMMITTED b4/b5 artifacts (no GPU).

PRE-REGISTRATION written 2026-10-20 ~17:10Z, BEFORE the run (plan v2 item 2;
primary read predeclared there: cell-level affine R^2 >= 0.5). Pure posthoc on
committed results; no measurement, no GPU, no new data.

DATA (read-only): results/b4_guard_v1.jsonl (816 held-out rows), 
results/b5_stage2.jsonl (288 rows), orientation/kept from results/guard_v1.json.
Sanity gate: signed-z mean must reproduce `score` on every row (abort otherwise).

UNIT u = (src, cell, arm); src in {b4,b5} is the family (b4=Linear-fold, b5=LN).
Signed per-row feature z: z_signed = (+1 if orient else -1) * z_raw[f], kept f.

t*_u (unit-optimal threshold): max Youden J = TPR-FPR over thresholds on the
unit's scores, positive class label==True; ties -> midpoint of the maximal-J
plateau (convention fixed now). Units lacking both classes are EXCLUDED and
counted (expected: some b5 arms; b5 randn_p1000 single-class recorded in JSON).

Covariate: m_{u,f} = mean signed z of feature f over the unit's rows;
d_u = mean_f [ m_{u,f} - mean over units of same family ]  (family-centered
unit score shift; the "constant part" is absorbed by the family intercept).

MODELS.
  M1 (PRIMARY)  t*_u ~ a + b5-indicator + c*d_u   (global affine law + family
                offset; the plan-v2 "threshold as surface = global fit + bind-
                time family offset" instantiation, 3 params, OLS, unweighted).
  PRIMARY READ: R2_M1 = 1 - SSres/SStot >= 0.5  => gate GREEN, items 3-4 unlock.
                R2 < 0.5 => the predeclared A4 killer objection fires:
                per-family-refit reframe (recorded, lane continues per plan).
  DESCRIPTIVE (non-gating): R2 of intercept/family-offset-only model; M2 with
  per-family slopes; n-weighted sensitivity R2; leave-one-unit-out R2 spread;
  full per-unit table (n, pos prev, t*, d_u, residual).

OUTPUT: results/d1_cellaffine.json (+ stdout table). Diary + commit; ATLAS A4
status line updated with the number, either branch.
"""
import json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

def load(name):
    return [json.loads(l) for l in open(os.path.join(RES, name))]

def youden_tstar(scores, labels):
    pairs = sorted(zip(scores, labels))
    n = len(pairs); P = sum(labels); N = n - P
    if P == 0 or N == 0:
        return None, None
    # candidate cuts between distinct consecutive scores; J evaluated with
    # accept = s >= t. Efficient sweep.
    # sweep all admissible cuts (between distinct sorted values); keep the
    # maximal-J plateau, threshold = plateau midpoint
    Js = []; idxs = []
    for i in range(-1, n):
        if i >= 0 and i < n - 1 and pairs[i][0] == pairs[i + 1][0]:
            continue
        acc = pairs[i + 1:]
        tp = sum(1 for s, l in acc if l); fp = len(acc) - tp
        J = (tp / P) - (fp / N)
        Js.append(J); idxs.append(i)
    m = max(Js)
    cuts = [idxs[k] for k, j in enumerate(Js) if abs(j - m) <= 1e-12]
    los = [pairs[c + 1][0] for c in cuts if c + 1 < n]
    his = [pairs[c][0] for c in cuts if c >= 0]
    lo = min(los) if los else (his[0] if his else 0.0)
    hi = max(his) if his else lo
    t = (lo + hi) / 2.0
    return t, m

def ols(X, y):
    # X: list of rows (lists); normal equations, gaussian elim; pure stdlib
    p = len(X[0])
    A = [[sum(X[r][i] * X[r][j] for r in range(len(X))) for j in range(p)] for i in range(p)]
    b = [sum(X[r][i] * y[r] for r in range(len(X))) for i in range(p)]
    # augment & solve
    M = [A[i][:] + [b[i]] for i in range(p)]
    for c in range(p):
        piv = max(range(c, p), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        if abs(pv) < 1e-12:
            return None, None
        M[c] = [v / pv for v in M[c]]
        for r in range(p):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [a - f * bb for a, bb in zip(M[r], M[c])]
    beta = [M[i][p] for i in range(p)]
    def pred(row): return sum(x * w for x, w in zip(row, beta))
    yhat = [pred(r) for r in X]
    ybar = sum(y) / len(y)
    ss_res = sum((yi - yh) ** 2 for yi, yh in zip(y, yhat))
    ss_tot = sum((yi - ybar) ** 2 for yi in y) or 1e-30
    return beta, 1.0 - ss_res / ss_tot

def main():
    guard = json.load(open(os.path.join(RES, "guard_v1.json")))
    kept, orient = guard["kept_features"], guard["orientation"]
    rows = []
    for src, fname in (("b4", "b4_guard_v1.jsonl"), ("b5", "b5_stage2.jsonl")):
        for r in load(fname):
            zs = {f: (1.0 if orient[f] else -1.0) * r["z_raw"][f] for f in kept}
            m = sum(zs.values()) / len(kept)
            if abs(m - r["score"]) > 1e-9:
                sys.exit("score reconstruction mismatch in %s row %s" % (src, r.get("cell")))
            rows.append(dict(src=src, cell=r.get("cell", "L1/40(ln)"), arm=r["arm"],
                             score=r["score"], label=bool(r["label"]), zmean=m))

    units = defaultdict(list)
    for r in rows:
        units[(r["src"], r["cell"], r["arm"])].append(r)

    tab, excluded = [], []
    for key, rs in sorted(units.items()):
        t, J = youden_tstar([r["score"] for r in rs], [1.0 if r["label"] else 0.0 for r in rs])
        if t is None:
            excluded.append(dict(unit=list(key), n=len(rs),
                                 n_pos=sum(1 for r in rs if r["label"])))
            continue
        tab.append(dict(src=key[0], cell=key[1], arm=key[2], n=len(rs),
                        n_pos=sum(1 for r in rs if r["label"]),
                        youden_J=J, t_star=t,
                        d_raw=sum(r["zmean"] for r in rs) / len(rs)))

    # family-center the covariate
    fams = defaultdict(list)
    for row in tab:
        fams[row["src"]].append(row["d_raw"])
    for row in tab:
        fm = sum(fams[row["src"]]) / len(fams[row["src"]])
        row["d_cen"] = row["d_raw"] - fm

    y = [r["t_star"] for r in tab]
    X1 = [[1.0, 1.0 if r["src"] == "b5" else 0.0, r["d_cen"]] for r in tab]
    beta1, r2_m1 = ols(X1, y)
    X0 = [[1.0, 1.0 if r["src"] == "b5" else 0.0] for r in tab]
    _, r2_offset = ols(X0, y)
    # M2: per-family slopes (b4 slope, b5 slope) + b5 intercept
    X2 = [[1.0, r["d_cen"] if r["src"] == "b4" else 0.0,
           1.0 if r["src"] == "b5" else 0.0,
           r["d_cen"] if r["src"] == "b5" else 0.0] for r in tab]
    beta2, r2_m2 = ols(X2, y)
    # n-weighted sensitivity for M1
    W = [r["n"] for r in tab]
    Xw = [[1.0, 1.0 if r["src"] == "b5" else 0.0, r["d_cen"]] for r in tab]
    Xw2 = [[c for c in row] for row in Xw]; yw = []; 
    for w, row, yi in zip(W, Xw, y):
        for _ in range(int(w)):
            yw.append(yi)
    Xrep = []
    for w, row in zip(W, Xw):
        Xrep += [row] * int(w)
    _, r2_m1_w = ols(Xrep, yw)

    # leave-one-unit-out stability of the PRIMARY read
    loos = []
    for i in range(len(tab)):
        sub = [k for k in range(len(tab)) if k != i]
        _, r2 = ols([X1[k] for k in sub], [y[k] for k in sub])
        if r2 is not None:
            loos.append(r2)

    gate = dict(primary_R2=r2_m1, threshold=0.5, green=(r2_m1 is not None and r2_m1 >= 0.5))
    out = dict(
        when=__import__("datetime").datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        prereg="docstring; written 2026-10-20 ~17:10Z BEFORE run; plan v2 item 2",
        inputs=dict(b4_rows=len(load("b4_guard_v1.jsonl")), b5_rows=len(load("b5_stage2.jsonl")),
                    kept=kept, orientation=orient,
                    corpus_sha256=guard["corpus_sha256"], frozen_t=guard["threshold"]),
        n_units_total=len(units), n_units_used=len(tab), n_excluded=len(excluded),
        excluded_single_class=excluded,
        gate=gate,
        M1_primary=dict(beta=beta1, labels=["const", "b5_ind", "d_cen"], R2=r2_m1),
        descriptive=dict(R2_offset_only=r2_offset, M2_per_family_slope=dict(beta=beta2, R2=r2_m2),
                         R2_M1_n_weighted=r2_m1_w,
                         LOO_R2_min=min(loos) if loos else None,
                         LOO_R2_max=max(loos) if loos else None),
        table=tab,
    )
    with open(os.path.join(RES, "d1_cellaffine.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)

    print("units used %d / %d (excluded %d)" % (len(tab), len(units), len(excluded)))
    print("PRIMARY  R2(M1 affine+offset) = %.4f  gate>=0.5 -> %s" % (r2_m1, gate["green"]))
    print("beta M1 %s" % ([round(b, 4) for b in beta1],))
    m2_note = ("b5 d_cen degenerate (family-constant z; within-family slope "
               "unidentifiable) — the B5 mechanism, confirmed in the covariate"
               ) if beta2 is None else ""
    print("desc: offset-only R2=%s  M2 R2=%s %s weighted=%.4f  LOO[%.3f,%.3f]"
          % ("n/a" if r2_offset is None else round(r2_offset, 4),
             "n/a" if r2_m2 is None else round(r2_m2, 4), m2_note,
             r2_m1_w, min(loos), max(loos)))
    for r in tab:
        print("  %-3s %-18s %-11s n=%3d pos=%2d t*=%+.4f J=%.3f d=%+.3f"
              % (r["src"], r["cell"], r["arm"], r["n"], r["n_pos"], r["t_star"], r["youden_J"], r["d_cen"]))
    for e in excluded:
        print("  excluded:", e)

if __name__ == "__main__":
    main()
