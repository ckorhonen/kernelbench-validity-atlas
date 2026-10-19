"""B3 stage-1: error-budget ownership decomposition + C-N classifier pre-test.

Pre-registration (written 2026-10-19 ~17:55Z BEFORE first run; implements the
stage-1 committed in meta-reflection #17 and the C-N test committed in #16.
No threshold below was edited after seeing results.)

STAGE 1 -- ownership decomposition (L2/14 arm, scale 1, seeds 0-4, n=5120 elts)
  T = fold32 output (fp32, grammar candidate); R = task reference output (fp32);
  E = fp64 evaluation of the fold path (E_f64); fp64 ref path E_r64 recorded;
  max|E_f64-E_r64| recorded as path-agreement diagnostic (expected << tol).
  Per element: t_i=|T-E|, r_i=|R-E|, d_i=|T-R| (the budgeted quantity).
  PRIMARY (decides the #17 branch): transform share on L2 norm, pooled n=5120,
    share_T2 = sqrt(sum t_i^2) / (sqrt(sum t_i^2) + sqrt(sum r_i^2)).
  Secondary (descriptive): median per-element share t/(t+r) (zero-total elts
    excluded, count recorded); per-seed shares; failing-element conditional
    frac(t_i > limit | d_i > limit)  [would the fold fail vs a perfect ref?].
  VERDICT: share_T2 >= 0.80 => transform-owned: grammar MAY DROP the
    version-pinned reference-error model for this family (record, not ship).
    share_T2 < 0.80 => reference-error model is load-bearing; keep it.
  Known confound (named in #17): single-family evidence; cross-family
    generality is exactly what the stage-2 corpus test addresses. No
    cross-family claim from stage-1 alone.

STAGE 2 -- C-N classifier pre-test on B2 corpus (preregistered BEFORE stage-1)
  Data: b2_failure_corpus.jsonl, method==fold32 rows, label=fold32_pass.
  Derived-only features: [collapse_work_ratio, fan_in_sum, w_max, w_mean_abs,
    w_rel_spread, B, K, H].  Plus-arm adds log10(scale)  [C-N delta test].
  PRIMARY: best univariate rank-AUC (Mann-Whitney, ties midrank) among the 8
    derived features on all rows. Grammar proceeds iff >= 0.80; else the
    degraded tier is declared (kill per plan B3).
  SECONDARY (C-N conjecture check, non-killing): AUC of combined score = mean
    of per-feature standardized ranks. Orientation per feature fixed
    deterministically from data (pass-class rank-mean higher), applied
    identically to both arms. delta = AUC_plus - AUC_derived.
    delta <= 0.05 => C-N supported (scale adds nothing beyond derived coords);
    delta > 0.05 => record "scale feature required" (grammar alive, conjecture
    not supported on this corpus).
  Descriptive if sklearn present: 5-fold stratified-CV logistic AUC both arms,
    seed 2026 (variance check; never replaces the rank-AUC primary).
  No feature selection after seeing labels beyond the fixed sets above.

Provenance: spark GB10 sm_120, torch 2.14.1+cu130, TF32 off, grad off;
correctness only. Results -> results/b3_stage1.json.
"""
import json, math, pathlib, importlib.util
import torch

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
ATOL = RTOL = 1e-4

ROOT = pathlib.Path("/home/ckorhonen/kb-lab")
RESULTS = ROOT / "results"; RESULTS.mkdir(exist_ok=True)

def load_task(n):
    p = next(pathlib.Path("/home/ckorhonen/kernelbench/KernelBench/KernelBench/level2").glob(f"{n}_*.py"))
    spec = importlib.util.spec_from_file_location(f"task{n}", p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

# ---------------- stage 1: arm A regenerated exactly as b2 ----------------
m14 = load_task(14)
per_seed = []; sums = {"t2": 0.0, "r2": 0.0}
for seed in range(5):
    torch.manual_seed(seed)
    model = m14.Model(*m14.get_init_inputs()).cuda().eval()
    x = m14.get_inputs()[0].cuda()
    W = model.weight; sf = model.scaling_factor
    T = (x @ W.sum(0)).unsqueeze(1) * (sf / 2)          # fold32, fp32 (as B2)
    R = model(x)                                        # task reference, fp32
    E_fold = (x.double() @ W.double().sum(0)).unsqueeze(1) * (float(sf) / 2.0)
    E_ref  = ((x.double() @ W.double().T) / 2.0).sum(1, keepdim=True) * float(sf)
    Td, Rd = T.double(), R.double()
    t = (Td - E_fold).abs(); r = (Rd - E_fold).abs(); d = (Td - Rd).abs()
    limit = ATOL + RTOL * Rd.abs()
    st2 = float((t ** 2).sum()); sr2 = float((r ** 2).sum())
    sums["t2"] += st2; sums["r2"] += sr2
    share_seed = math.sqrt(st2) / (math.sqrt(st2) + math.sqrt(sr2))
    tot = t + r; nz = tot > 0
    med_share = float((t[nz] / tot[nz]).median()) if bool(nz.any()) else float("nan")
    fail = d > limit; n_fail = int(fail.sum())
    frac_tf = float((t[fail] > limit[fail]).float().mean()) if n_fail else float("nan")
    row = dict(seed=seed, n=int(t.numel()),
        path_agreement_max=float((E_fold - E_ref).abs().max()),
        share_T2_seed=share_seed, median_elem_share=med_share,
        zero_total=int((~nz).sum()), n_fail=n_fail,
        frac_towner_on_fail=frac_tf, d_max=float(d.max()),
        budget_ratio_max=float((d / limit).max()), t_max=float(t.max()), r_max=float(r.max()))
    per_seed.append(row)
    print(f"seed {seed}: share_T2={share_seed:.4f} med={med_share:.4f} n_fail={n_fail} "
          f"frac_t@fail={frac_tf:.3f} path_agree={row['path_agreement_max']:.2e}")
    del model, x, T, R, E_fold, E_ref, Td, Rd, t, r, d, limit
torch.cuda.empty_cache()

share_T2 = math.sqrt(sums["t2"]) / (math.sqrt(sums["t2"]) + math.sqrt(sums["r2"]))
stage1 = dict(n_pooled=sum(p["n"] for p in per_seed), share_T2_pooled=share_T2,
              verdict="transform_owned_may_drop_ref_model" if share_T2 >= 0.80
                      else "ref_model_load_bearing",
              per_seed=per_seed)
print("STAGE1 share_T2 pooled:", round(share_T2, 4), "verdict:", stage1["verdict"])

# ---------------- stage 2: C-N rank-AUC on B2 corpus ----------------
def rankdata(v):
    idx = sorted(range(len(v)), key=lambda i: v[i]); out = [0.0] * len(v); i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and v[idx[j + 1]] == v[idx[i]]: j += 1
        m = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1): out[idx[k]] = m
        i = j + 1
    return out

def auc(score, lab):
    r = rankdata(score); n1 = sum(lab); n0 = len(lab) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    s1 = sum(rk for rk, l in zip(r, lab) if l)
    return (s1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)

rows = [json.loads(l) for l in open(ROOT / "results" / "b2_failure_corpus.jsonl")]
rows = [r for r in rows if r.get("method") == "fold32"]
DERIVED = ["collapse_work_ratio", "fan_in_sum", "w_max", "w_mean_abs", "w_rel_spread", "B", "K", "H"]
lab = [1 if r.get("fold32_pass", r.get("label")) else 0 for r in rows]  # arm A vs B/C key

def featvals(f):
    return [math.log10(r["scale"]) if f == "log10_scale" else float(r[f]) for r in rows]

uni = {}
for f in DERIVED + ["log10_scale"]:
    a = auc(featvals(f), lab)
    rk = rankdata(featvals(f))
    mp = sum(x for x, l in zip(rk, lab) if l) / max(sum(lab), 1)
    mf = sum(x for x, l in zip(rk, lab) if not l) / max(len(lab) - sum(lab), 1)
    uni[f] = dict(auc=a, orient_pass_higher=bool(mp >= mf))
    print(f"feature {f}: AUC={a:.4f} pass_higher={mp >= mf}")

def combined(features):
    n = len(rows); cols = []
    for f in features:
        rk = rankdata(featvals(f))
        z = [(v - (n + 1) / 2) / ((n ** 2 - 1) / 12.0) ** 0.5 for v in rk]
        if not uni[f]["orient_pass_higher"]: z = [-v for v in z]
        cols.append(z)
    return [sum(c[i] for c in cols) / len(cols) for i in range(n)]

auc_der = auc(combined(DERIVED), lab)
auc_plus = auc(combined(DERIVED + ["log10_scale"]), lab)
stage2 = dict(n_rows=len(rows), n_pass=sum(lab),
    univariate=uni, best_derived_auc=max(uni[f]["auc"] for f in DERIVED),
    best_derived_feature=max(DERIVED, key=lambda f: uni[f]["auc"]),
    combined_auc_derived=auc_der, combined_auc_plus=auc_plus, delta=auc_plus - auc_der)
stage2["grammar_proceeds"] = bool(stage2["best_derived_auc"] >= 0.80)
stage2["cn_supported"] = bool(stage2["delta"] <= 0.05)
print("STAGE2 best derived AUC:", round(stage2["best_derived_auc"], 4),
      "feature:", stage2["best_derived_feature"],
      "| combined der/plus:", round(auc_der, 4), round(auc_plus, 4),
      "delta:", round(auc_plus - auc_der, 4))

try:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    def cv_auc(feats):
        X = np.array([[math.log10(r["scale"]) if f == "log10_scale" else float(r[f]) for f in feats] for r in rows])
        y = np.array(lab); aucs = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=2026).split(X, y):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr])
            aucs.append(roc_auc_score(y[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
        return float(np.mean(aucs)), [round(a, 4) for a in aucs]
    m1, f1 = cv_auc(DERIVED); m2, f2 = cv_auc(DERIVED + ["log10_scale"])
    stage2["cv_logistic_derived"] = m1; stage2["cv_logistic_plus"] = m2
    stage2["cv_folds_derived"] = f1; stage2["cv_folds_plus"] = f2
    print("CV logistic AUC der/plus:", round(m1, 4), round(m2, 4))
except Exception as e:
    stage2["sklearn_note"] = f"sklearn CV skipped: {type(e).__name__}: {e}"
    print("sklearn CV skipped:", e)

out = dict(when="2026-10-19T17:55Z", prereg="docstring written before run",
           stage1=stage1, stage2=stage2,
           env=dict(gpu="GB10 sm_120", torch="2.14.1+cu130", tf32=False))
(RESULTS / "b3_stage1.json").write_text(json.dumps(out, indent=1))
print("saved", RESULTS / "b3_stage1.json")
