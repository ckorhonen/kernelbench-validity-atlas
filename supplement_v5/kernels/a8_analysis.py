"""A8 analysis — compression of (rho_max, frac_over) onto derived coordinates.

Prereg: lab/a8_compress_prereg.md @ c56c0a0, frozen BEFORE the run. Verdict rule applied verbatim.
Models M2 (class-median), M3 (class + derived), M4 (derived only, no class). LOO out-of-fold.
"""
import json, os, itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "results", "a8_compress.json")
OUT = os.path.join(HERE, "results", "a8_analysis.json")

CLASSES = ["W", "TF32", "GEMM", "OTHER"]


def feats(r):
    """Derived coordinates per prereg: x1..x4 (log10 k_eff, log10 n_elem, log10 C_eff, log10 H)."""
    keff = max(r.get("k_eff") or 1, 1)
    n = max(r.get("n_elem") or 1, 1)
    ceff = max(r.get("C_eff") or r.get("C") or r.get("M") or 1, 1)
    H = r.get("H") or 1
    return [np.log10(keff), np.log10(n), np.log10(ceff), np.log10(H)]


def onehot(r):
    return [1.0 if r["engine"] == c else 0.0 for c in CLASSES]


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else 0.0


def fit_ols(X, y):
    X = np.asarray(X, float); y = np.asarray(y, float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def predict(X, coef):
    return np.asarray(X, float) @ np.asarray(coef, float)


def loo(rows, mode):
    """mode: 'class_median' | 'class_derived' | 'derived_only'."""
    preds_rho, preds_frac = [], []
    for i, held in enumerate(rows):
        train = [r for j, r in enumerate(rows) if j != i]
        if mode == "class_median":
            same = [r for r in train if r["engine"] == held["engine"]]
            pool = same if same else train
            pr = float(np.median([np.log10(max(r["rho_max"], 1e-9)) for r in pool]))
            pf = float(np.median([r["frac_over"] for r in pool]))
        else:
            if mode == "class_derived":
                Xtr = [onehot(r) + feats(r) for r in train]
                Xte = [onehot(held) + feats(held)]
            else:  # derived_only
                Xtr = [feats(r) for r in train]
                Xte = [feats(held)]
            Xtr = [[1.0] + x for x in Xtr]; Xte = [[1.0] + x for x in Xte]
            cr = fit_ols(Xtr, [np.log10(max(r["rho_max"], 1e-9)) for r in train])
            cf = fit_ols(Xtr, [r["frac_over"] for r in train])
            pr = float(predict(Xte, cr)[0])
            pf = float(np.clip(predict(Xte, cf)[0], 0.0, 1.0))
        preds_rho.append(pr); preds_frac.append(pf)
    meas_rho = [np.log10(max(r["rho_max"], 1e-9)) for r in rows]
    meas_frac = [r["frac_over"] for r in rows]
    bulk_true = [1 if f > 0.5 else 0 for f in meas_frac]
    bulk_pred = [1 if f > 0.5 else 0 for f in preds_frac]
    acc = float(np.mean([1 if a == b else 0 for a, b in zip(bulk_true, bulk_pred)]))
    return {"spearman_logrho": spearman(preds_rho, meas_rho),
            "bulk_acc": acc,
            "frac_mae": float(np.mean([abs(a - b) for a, b in zip(preds_frac, meas_frac)])),
            "preds": [{"row": r["row"], "pred_logrho": p, "meas_logrho": m,
                       "pred_frac": pf, "meas_frac": mf}
                      for r, p, m, pf, mf in zip(rows, preds_rho, meas_rho, preds_frac, meas_frac)]}


def within_class_spans(rows):
    out = {}
    for c in CLASSES:
        sub = [r for r in rows if r["engine"] == c]
        if not sub:
            continue
        rr = [np.log10(max(r["rho_max"], 1e-9)) for r in sub]
        ff = [r["frac_over"] for r in sub]
        out[c] = {"n": len(sub), "logrho_min": float(min(rr)), "logrho_max": float(max(rr)),
                  "logrho_span_decades": float(max(rr) - min(rr)),
                  "frac_min": float(min(ff)), "frac_max": float(max(ff))}
    return out


def dispatch_condition(rows):
    """Which recorded coordinates separate the engine classes? Report per (k,s,d) the classes seen."""
    groups = {}
    for r in rows:
        if r["family"] != "conv2d":
            continue
        key = f"k{r['k']}_s{r['s']}_d{r['d']}"
        groups.setdefault(key, {}).setdefault(r["engine"], []).append((r["C"], r["H"], round(r["rho_max"], 2)))
    return groups


def main():
    d = json.load(open(SRC))
    rows = d["rows"]
    results = {}
    for mode in ("class_median", "class_derived", "derived_only"):
        results[mode] = loo(rows, mode)
        print(mode, "rho_spearman", round(results[mode]["spearman_logrho"], 4),
              "bulk_acc", round(results[mode]["bulk_acc"], 4),
              "frac_mae", round(results[mode]["frac_mae"], 4))

    spans = within_class_spans(rows)
    cross = {}
    for a, b in itertools.combinations([c for c in CLASSES if c in spans], 2):
        cross[f"{a}_vs_{b}"] = spans[a]["logrho_min"] - spans[b]["logrho_max"]
    disp = dispatch_condition(rows)

    m3 = results["class_derived"]; m4 = results["derived_only"]; m2 = results["class_median"]
    if m3["spearman_logrho"] >= 0.80 and m3["bulk_acc"] >= 0.90 and m4["spearman_logrho"] < 0.60:
        verdict = "COMPRESSION_SUPPORTED_CATEGORICAL"
    elif m3["spearman_logrho"] >= 0.80 and m4["spearman_logrho"] >= 0.80 and m3["bulk_acc"] >= 0.90:
        verdict = "COMPRESSION_SUPPORTED_CONTINUOUS"
    elif m3["spearman_logrho"] >= 0.80 and (m3["bulk_acc"] < 0.90 or m4["spearman_logrho"] < 0.80):
        verdict = "COMPRESSION_PARTIAL"
    else:
        verdict = "NO_COMPRESSION"

    # falsifier: within-class span < 1 decade while cross-class span > 1 decade => class sufficient
    max_within = max([s["logrho_span_decades"] for s in spans.values()]) if spans else 0.0
    min_cross = min(cross.values()) if cross else 0.0
    falsifier = {"max_within_class_span_decades": max_within,
                 "min_cross_class_gap_decades": min_cross,
                 "class_alone_sufficient": bool(max_within < 1.0 < min_cross)}

    out = {"header": {"prereg": "lab/a8_compress_prereg.md @ c56c0a0",
                      "src": "lab/results/a8_compress.json",
                      "runtime_facts": d["header"]["runtime_facts"],
                      "n_rows": len(rows), "loo": True},
           "models": {k: {kk: vv for kk, vv in v.items() if kk != "preds"} for k, v in results.items()},
           "preds": {k: v["preds"] for k, v in results.items()},
           "within_class_spans": spans, "cross_class_gaps": cross,
           "dispatch_by_ksd": disp,
           "falsifier": falsifier, "verdict": verdict,
           "controls": d["controls"]}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("VERDICT", verdict)
    print("falsifier", falsifier)


if __name__ == "__main__":
    main()
