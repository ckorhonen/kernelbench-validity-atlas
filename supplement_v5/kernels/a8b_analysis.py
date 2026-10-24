"""A8b — corrected engine classification + class-predictability sub-analysis.

DECLARED POST-HOC CORRECTION (not a prereg bar change): the frozen map in a8_analysis.py keys on the
literal substrings 'winograd'/'tf32'/'sgemm'/'gemm'. Raw kernel names in results/a8_compress.json show
that conv 1x1 rows at C=256 run `cutlass_80_tensorop_s1688gemm_*` — tensor-core TF32 hardware whose
name carries no literal 'tf32' — while C=16 1x1 rows run `implicit_convolve_sgemm` (true fp32 SIMT).
The frozen map therefore merges two physically different arithmetic classes into 'GEMM', which
inflates the within-class rho span to 4.54 decades. This script reclassifies from the stored raw names
and reports BOTH verdicts; the frozen one stands as the preregistered reading.

Corrected classes (from a6/a7 physics: s1688 tensorop = TF32 tensor core):
  W          = any kernel name containing 'winograd'
  TF32       = contains 'tensorop_s1688' or 'tf32'
  GEMM_FP32  = contains 'simt_sgemm' or 'implicit_convolve_sgemm' or plain 'gemm' without tensorop
  OTHER      = anything else

Second question added here (descriptive, no bar): is the engine CLASS itself predictable from derived
coordinates (LOO nearest-centroid on the same x1..x4)? If not, the certificate needs the class
measured per shape, which is a stronger statement than rho needing measurement.
"""
import json, os, itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "results", "a8_compress_v2.json")
OUT = os.path.join(HERE, "results", "a8b_analysis.json")

CLASSES = ["W", "TF32", "GEMM_FP32", "OTHER"]


def classify(names):
    s = " ".join(names).lower()
    if "winograd" in s:
        return "W"
    if "tensorop_s1688" in s or "tf32" in s:
        return "TF32"
    if "simt_sgemm" in s or "implicit_convolve_sgemm" in s or "gemm" in s:
        return "GEMM_FP32"
    return "OTHER"


def feats(r):
    keff = max(r.get("k_eff") or 1, 1)
    n = max(r.get("n_elem") or 1, 1)
    ceff = max(r.get("C_eff") or r.get("C") or r.get("M") or 1, 1)
    H = r.get("H") or 1
    return [np.log10(keff), np.log10(n), np.log10(ceff), np.log10(H)]


def onehot(r, cls):
    return [1.0 if cls == c else 0.0 for c in CLASSES]


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


def loo(rows, mode):
    pr_l, pf_l = [], []
    for i, held in enumerate(rows):
        train = [r for j, r in enumerate(rows) if j != i]
        if mode == "class_median":
            same = [r for r in train if r["cls"] == held["cls"]]
            pool = same if same else train
            pr = float(np.median([np.log10(max(r["rho_max"], 1e-9)) for r in pool]))
            pf = float(np.median([r["frac_over"] for r in pool]))
        else:
            if mode == "class_derived":
                Xtr = [onehot(r, r["cls"]) + feats(r) for r in train]
                Xte = [onehot(held, held["cls"]) + feats(held)]
            else:
                Xtr = [feats(r) for r in train]; Xte = [feats(held)]
            Xtr = [[1.0] + x for x in Xtr]; Xte = [[1.0] + x for x in Xte]
            cr = fit_ols(Xtr, [np.log10(max(r["rho_max"], 1e-9)) for r in train])
            cf = fit_ols(Xtr, [r["frac_over"] for r in train])
            pr = float((np.asarray(Xte, float) @ cr)[0])
            pf = float(np.clip((np.asarray(Xte, float) @ cf)[0], 0, 1))
        pr_l.append(pr); pf_l.append(pf)
    mr = [np.log10(max(r["rho_max"], 1e-9)) for r in rows]
    mf = [r["frac_over"] for r in rows]
    bt = [1 if f > 0.5 else 0 for f in mf]; bp = [1 if f > 0.5 else 0 for f in pf_l]
    return {"spearman_logrho": spearman(pr_l, mr),
            "bulk_acc": float(np.mean([1 if a == b else 0 for a, b in zip(bt, bp)])),
            "frac_mae": float(np.mean([abs(a - b) for a, b in zip(pf_l, mf)])),
            "preds": [{"row": r["row"], "cls": r["cls"], "pred_logrho": p, "meas_logrho": m,
                       "pred_frac": f, "meas_frac": g} for r, p, m, f, g in zip(rows, pr_l, mr, pf_l, mf)]}


def class_predict(rows):
    """LOO nearest-centroid on derived feats; accuracy vs chance (majority class)."""
    correct = 0; maj = 0
    preds = []
    for i, held in enumerate(rows):
        train = [r for j, r in enumerate(rows) if j != i]
        cents = {}
        for c in CLASSES:
            sub = [feats(r) for r in train if r["cls"] == c]
            if sub:
                cents[c] = np.mean(np.asarray(sub, float), axis=0)
        x = np.asarray(feats(held), float)
        best = min(cents, key=lambda c: float(np.linalg.norm((x - cents[c]) / (np.std(np.asarray([feats(r) for r in train], float), axis=0) + 1e-9))))
        ok = best == held["cls"]
        correct += int(ok)
        preds.append({"row": held["row"], "true": held["cls"], "pred": best, "ok": ok})
    from collections import Counter
    maj_cls = Counter(r["cls"] for r in rows).most_common(1)[0][0]
    maj = sum(1 for r in rows if r["cls"] == maj_cls)
    return {"loo_accuracy": correct / len(rows), "majority_baseline": maj / len(rows),
            "majority_class": maj_cls, "preds": preds}


def main():
    d = json.load(open(SRC))
    rows = []
    for r in d["rows"]:
        r = dict(r)
        r["cls"] = classify(r["kernels"]) if r.get("kernels") else r["engine"]
        rows.append(r)
    from collections import Counter
    print("corrected class counts", Counter(r["cls"] for r in rows))
    print("frozen-map counts", Counter(r["engine"] for r in rows))

    results = {m: loo(rows, m) for m in ("class_median", "class_derived", "derived_only")}
    for m, v in results.items():
        print(m, "rho_sp", round(v["spearman_logrho"], 4), "bulk", round(v["bulk_acc"], 4),
              "mae", round(v["frac_mae"], 4))

    spans = {}
    for c in CLASSES:
        sub = [r for r in rows if r["cls"] == c]
        if not sub:
            continue
        rr = [np.log10(max(r["rho_max"], 1e-9)) for r in sub]
        spans[c] = {"n": len(sub), "min": float(min(rr)), "max": float(max(rr)),
                    "span_decades": float(max(rr) - min(rr)),
                    "frac_min": float(min(r["frac_over"] for r in sub)),
                    "frac_max": float(max(r["frac_over"] for r in sub)),
                    "rows": [r["row"] for r in sub]}
        print(c, spans[c]["n"], "span", round(spans[c]["span_decades"], 3))

    m3, m4 = results["class_derived"], results["derived_only"]
    if m3["spearman_logrho"] >= 0.80 and m3["bulk_acc"] >= 0.90 and m4["spearman_logrho"] < 0.60:
        v2 = "COMPRESSION_SUPPORTED_CATEGORICAL"
    elif m3["spearman_logrho"] >= 0.80 and m4["spearman_logrho"] >= 0.80 and m3["bulk_acc"] >= 0.90:
        v2 = "COMPRESSION_SUPPORTED_CONTINUOUS"
    elif m3["spearman_logrho"] >= 0.80:
        v2 = "COMPRESSION_PARTIAL"
    else:
        v2 = "NO_COMPRESSION"
    print("CORRECTED VERDICT", v2)

    # dispatch table: class by (k,s,d) x (C,H)
    disp = {}
    for r in rows:
        if r["family"] != "conv2d":
            continue
        disp.setdefault(f"k{r['k']}_s{r['s']}_d{r['d']}", {}).setdefault(f"C{r['C']}_H{r['H']}", []).append(
            {"cls": r["cls"], "rho": round(r["rho_max"], 3), "frac": round(r["frac_over"], 4)})

    cp = class_predict(rows)
    print("class LOO accuracy", round(cp["loo_accuracy"], 4), "majority", round(cp["majority_baseline"], 4))

    out = {"header": {"prereg": "lab/a8_compress_prereg.md @ c56c0a0",
                      "correction": "engine map corrected post-hoc from stored raw kernel names; frozen verdict stands as preregistered reading",
                      "runtime_facts": d["header"]["runtime_facts"], "n_rows": len(rows)},
           "models_corrected": {k: {kk: vv for kk, vv in v.items() if kk != "preds"} for k, v in results.items()},
           "preds_corrected": {k: v["preds"] for k, v in results.items()},
           "within_class_spans_corrected": spans,
           "dispatch_table": disp,
           "class_predictability": {k: v for k, v in cp.items() if k != "preds"},
           "class_preds": cp["preds"],
           "verdict_corrected": v2,
           "controls": d["controls"]}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
