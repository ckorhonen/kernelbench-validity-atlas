#!/usr/bin/env python3
"""r57: candidate-level admission compressibility. Implements lab/r57_candidate_admission_prereg.md
verbatim. CPU only; reads committed records. Run only after the prereg commit."""
import json, math, hashlib, sys, datetime
import numpy as np

REPO = "/Users/ckorhonen/rstation/kernelbench"
SRC = f"{REPO}/lab/results/r1_conv_gated.jsonl"
PREREG = f"{REPO}/lab/r57_candidate_admission_prereg.md"
OUT = f"{REPO}/lab/results/r57_candidate_admission.json"

def fam(arm):
    return "torch" if arm == "torch" else ("compile" if arm == "torch_cmp" else "hand")

rows = [json.loads(l) for l in open(SRC)]
recs = []
for r in rows:
    refdev = abs(r["ref_absmax"] - r["ref32_absmax"]) / (1e-4 + 1e-4 * abs(r["ref_absmax"]))
    for a, v in r["variants"].items():
        if v.get("pass_1e4") is None:
            continue
        prec = v.get("precision")
        recs.append(dict(
            row_key=(r["tag"], r["C"], r["H"], r["W"], r["N"], r["K"]),
            arm=a, fam=fam(a), prec=prec,
            y=1 if v["pass_1e4"] else 0,
            feats=[
                math.log2(r["C"]), math.log2(r["K"]), math.log2(r["H"] * r["W"]), float(r["N"]),
                math.log2(r["bytes"]), refdev,
                1.0 if fam(a) == "compile" else 0.0,
                1.0 if fam(a) == "hand" else 0.0,
                1.0 if prec == "tf32" else 0.0,
                (1.0 if fam(a) == "hand" else 0.0) * refdev,
            ],
        ))

n = len(recs); pos = sum(r["y"] for r in recs)
row_keys = sorted({r["row_key"] for r in recs})
print(f"n_armchecks={n} pos={pos} rows={len(row_keys)}")

X = np.array([r["feats"] for r in recs], dtype=float)
y = np.array([r["y"] for r in recs], dtype=float)
mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
Z = (X - mu) / sd

def auc(scores, labels):
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    # tie handling: average ranks for equal scores
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    p = labels.sum(); q = len(labels) - p
    if p == 0 or q == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - p * (p + 1) / 2.0) / (p * q))

pred = np.full(n, np.nan)
for rk in row_keys:
    test = np.array([r["row_key"] == rk for r in recs])
    train = ~test
    if train.sum() < 20:
        continue
    beta, *_ = np.linalg.lstsq(np.c_[np.ones(train.sum()), Z[train]], y[train], rcond=None)
    pred[test] = np.c_[np.ones(test.sum()), Z[test]] @ beta

ok = ~np.isnan(pred)
base_acc = max(y.mean(), 1 - y.mean())
acc = float(((pred[ok] >= 0.5).astype(float) == y[ok]).mean())
a_all = auc(pred[ok], y[ok])

# P2: hand-ieee stratum only, same folds/model spec (refit on stratum)
hi = np.array([r["fam"] == "hand" and r["prec"] == "ieee" for r in recs])
pred2 = np.full(n, np.nan)
for rk in row_keys:
    test = np.array([r["row_key"] == rk for r in recs]) & hi
    train = (~np.array([r["row_key"] == rk for r in recs])) & hi
    if test.sum() == 0 or train.sum() < 20:
        continue
    beta, *_ = np.linalg.lstsq(np.c_[np.ones(train.sum()), Z[train]], y[train], rcond=None)
    pred2[test] = np.c_[np.ones(test.sum()), Z[test]] @ beta
ok2 = ~np.isnan(pred2) & hi
a_hi = auc(pred2[ok2], y[ok2])
acc_hi = float(((pred2[ok2] >= 0.5).astype(float) == y[ok2]).mean())
base_hi = max(y[hi].mean(), 1 - y[hi].mean())

# P3: mixed verdict rows
mixed = 0
for rk in row_keys:
    ys = {r["y"] for r in recs if r["row_key"] == rk}
    if len(ys) > 1:
        mixed += 1

p1 = dict(auc=a_all, accuracy=acc, majority_baseline=float(base_acc), bar_acc=float(base_acc) + 0.05,
          verdict=None)
p1["verdict"] = "PASS" if (a_all >= 0.85 and acc >= base_acc + 0.05) else "KILL"
p2 = dict(auc=a_hi, accuracy=acc_hi, n=int(hi.sum()), positives=int(y[hi].sum()),
          majority_baseline=float(base_hi), bar=0.80)
p2["verdict"] = "PASS" if a_hi >= 0.80 else "FAIL"
p3 = dict(mixed_rows=mixed, total_rows=len(row_keys))

# descriptive: arm-class-only rule (no fitting) for context, not primary
res = dict(
    prereg=f"lab/r57_candidate_admission_prereg.md sha256:{hashlib.sha256(open(PREREG,'rb').read()).hexdigest()[:16]}",
    source=SRC, source_sha=hashlib.sha256(open(SRC,'rb').read()).hexdigest()[:16],
    date=datetime.datetime.utcnow().isoformat() + "Z",
    n_armchecks=n, positives=int(pos), rows=len(row_keys),
    P1=p1, P2=p2, P3=p3,
    feature_names=["log2C","log2K","log2HW","N","log2bytes","refdev","is_compile","is_hand","prec_tf32","hand_x_refdev"],
    method="linear probability OLS, LOO grouped by row, standardized features, no tuning",
)
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
