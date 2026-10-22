"""B3 legs 2-3: family-controlled CV + surrogate-exposure validation.

Pre-registration (written 2026-10-19 ~21:05Z BEFORE first run; implements the
plan legs committed in #16/#18 and restated in the #18 follow-up. No threshold
below was edited after seeing results.)

Data: b2_failure_corpus.jsonl, method==fold32 rows (175). Families:
  l2_14_fold (5 rows, SINGLE-CLASS: all fail), l2_51_cancel (120),
  l2_51_ordinary_mag (50).

LEG 2 -- family-controlled CV of the derived-only joint rank score
  Score: identical functional form to stage-2 (mean over features of
  orientation-signed standardized ranks; ties midrank; parameter-free normal
  approx (n+1)/2, sqrt((n^2-1)/12)). The ONLY fitted content -- per-feature
  orientation (pass-class rank-mean higher) -- is computed on TRAIN families
  only; ranks/standardization are within the evaluation set (declared
  convention: monotone per feature, no labels touched).
  LOFO: for each family h: train = other two families; eval on h.
  PRIMARY: held-out AUC >= 0.80 for EVERY family with both classes present
    (l2_51_cancel, l2_51_ordinary_mag). l2_14_fold is single-class => AUC
    undefined there; descriptive placement only (see below).
  SECONDARY: family-collapse flag := any two-class held-out AUC < 0.70
    (recorded; feeds the B4 tier decision). Directed pairwise transfer AUCs
    (train=cancel/eval=ordmag and reverse) recorded, descriptive.
  C-N under family controls: per held-out family
    delta = AUC(derived+log10scale) - AUC(derived), same LOFO machinery.
    max|delta| > 0.05 anywhere => 'scale feature required under family
    controls' recorded (revises the train-level C-N support of stage-2).
  Arm-A descriptive (non-killing): pooled-rank basis (arm A + its train
    families ranked together), orientation from train rows only; median score
    of arm-A rows vs train fail/pass medians (sign-consistency check).

LEG 3 -- surrogate-exposure validation
  Surrogate (the ONLY one tested): phi_cand = fraction of the fp32 candidate
  output T with |T| < 1e-4 (tol-matched; computed from the guarded path's own
  output; no reference touched => runtime-computable).
  Exact: phi_ref = fraction |R| < 1e-4 on the task reference. Arm B/C
  phi_ref cross-checked against the corpus `phi` field (max abs diff
  recorded; sanity only). Arm A fills the corpus's phi=None.
  Validity PRIMARY: Spearman rank correlation rho(phi_ref, phi_cand) pooled
    over all 175 rows >= 0.80.
  Utility PRIMARY: LOFO-CV AUC (leg-2 convention, two-class families):
    phi_cand validates iff AUC(derived+phi_cand) >= AUC(derived+phi_ref) - 0.05
    in BOTH held-out families AND pooled rho >= 0.80.
  Recorded regardless: AUC(derived+phi_ref) - AUC(derived) per held-out
    family (is the exposure feature load-bearing at all on this corpus?).
  VERDICT RULES: pass => exposure feature admitted to the grammar in surrogate
    form for B4. fail => per plan change-course: degraded tier -- B4 ships
    empirical-envelope-only (no exposure feature); surrogate failure recorded
    as a negative result.

Regeneration: T and R for every fold32 cell recomputed verbatim from the B2
arm code (same seeds, same call order, same ops) on spark; phi captured at
each cell. No thresholds, no labels used in regeneration.

Provenance: spark GB10 sm_120, torch 2.14.1+cu130, TF32 off, grad off;
correctness only. Results -> results/b3_leg23.json.
"""
import os
import json, math, pathlib, importlib.util
import torch

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
ATOL = 1e-4

ROOT = pathlib.Path(__file__).resolve().parent  # de-absolutized #42b (R4): lab layout root
RESULTS = ROOT / "results"; RESULTS.mkdir(exist_ok=True)

def load_task(n):
    p = next(pathlib.Path(os.path.expanduser("~/kernelbench/KernelBench/KernelBench/level2")).glob(f"{n}_*.py"))
    spec = importlib.util.spec_from_file_location(f"task{n}", p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

# ---------------- phi regeneration (verbatim B2 arm logic) ----------------
phi = {}   # key -> dict(phi_ref, phi_cand); key=(family,seed,scale,draw)
xchk = []  # corpus-phi cross-checks

def cap(fam, s, sc, dr, ref, T):
    phi[(fam, s, sc, dr)] = dict(
        phi_ref=float((ref.abs() < ATOL).float().mean()),
        phi_cand=float((T.abs() < ATOL).float().mean()))

# arm A (L2/14, seeds 0-4) -- verbatim b2 lines 76-84 tensors
m14 = load_task(14)
for seed in range(5):
    torch.manual_seed(seed)
    model = m14.Model(*m14.get_init_inputs()).cuda().eval()
    x = m14.get_inputs()[0].cuda()
    W = model.weight; sf = model.scaling_factor
    ref = model(x)
    T = (x @ W.sum(0)).unsqueeze(1) * (sf / 2)
    cap("l2_14_fold", seed, 1.0, None, ref, T)
    del model, x, ref, T, W
torch.cuda.empty_cache()

# arm B (seed-43 constructions) -- verbatim b2 lines 104-138 tensors
m51 = load_task(51)
torch.manual_seed(43)
m = m51.Model(*m51.get_init_inputs()).cuda().eval()
base_w = m.gemm.weight.detach().clone(); base_b = m.gemm.bias.detach().clone()
K51 = m51.get_inputs()[0].shape[1]
scales = [1.0, 100.0, 1e3, 1e4, 1e5, 1e6]
for si, scale in enumerate(scales):
    m.gemm.weight.copy_(base_w * scale); m.gemm.bias.copy_(base_b)
    for i in range(20):
        torch.manual_seed(4300 + si * 100 + i)
        r_ = torch.rand(1, K51, device="cuda"); r_[0, 0] = 0
        x = r_.expand(2048, K51).contiguous()
        m.subtract.copy_(m.gemm(x)[0])
        ref = m(x)
        v = m.gemm.weight.mean(0); c = m.gemm.bias.mean() - m.subtract.mean()
        pre = (x @ v + c)
        T = torch.nn.functional.gelu(pre.unsqueeze(1)) + x
        cap("l2_51_cancel", 43, scale, i, ref, T)
    del x
torch.cuda.empty_cache()

# arm C (ordinary magnification) -- verbatim b2 lines 145-167 tensors
for seed in (44, 45):
    torch.manual_seed(seed)
    m = m51.Model(*m51.get_init_inputs()).cuda().eval()
    x = m51.get_inputs()[0].cuda()
    base_w = m.gemm.weight.detach().clone(); base_b = m.gemm.bias.detach().clone()
    for si in range(25):
        scale = float(10 ** (6.0 * si / 24))
        m.gemm.weight.copy_(base_w * scale); m.gemm.bias.copy_(base_b)
        ref = m(x)
        v = m.gemm.weight.mean(0); c = m.gemm.bias.mean() - m.subtract.mean()
        pre = x @ v + c
        T = torch.nn.functional.gelu(pre.unsqueeze(1)) + x
        cap("l2_51_ordinary_mag", seed, scale, None, ref, T)
    del m, x
    torch.cuda.empty_cache()
print(f"phi regenerated for {len(phi)} cells")

# ---------------- corpus + phi merge ----------------
rows = [json.loads(l) for l in open(ROOT / "results" / "b2_failure_corpus.jsonl")]
rows = [r for r in rows if r.get("method") == "fold32"]
def key(r): return (r["family"], r["seed"], float(r["scale"]), r.get("draw"))
missing = [key(r) for r in rows if key(r) not in phi]
assert not missing, f"phi regen missing {len(missing)} keys e.g. {missing[:3]}"
for r in rows:
    p = phi[key(r)]
    r["phi_ref"] = p["phi_ref"]; r["phi_cand"] = p["phi_cand"]
    if r.get("phi") is not None:
        xchk.append(abs(p["phi_ref"] - r["phi"]))
xchk_max = max(xchk) if xchk else None
print("corpus-phi cross-check max abs diff:", xchk_max)

lab = [1 if r.get("fold32_pass", r.get("label")) else 0 for r in rows]
FAMS = sorted({r["family"] for r in rows})
TWO_CLASS = [f for f in FAMS if 0 < sum(l for l, r in zip(lab, rows) if r["family"] == f) < sum(1 for r in rows if r["family"] == f)]
print("families:", {f: sum(1 for r in rows if r["family"] == f) for f in FAMS}, "two-class:", TWO_CLASS)

# ---------------- rank machinery (stage-2 verbatim + train-orientation) ----
def rankdata(v):
    idx = sorted(range(len(v)), key=lambda i: v[i]); out = [0.0] * len(v); i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and v[idx[j + 1]] == v[idx[i]]: j += 1
        m = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1): out[idx[k]] = m
        i = j + 1
    return out

def auc(score, lab_):
    r = rankdata(score); n1 = sum(lab_); n0 = len(lab_) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    s1 = sum(rk for rk, l in zip(r, lab_) if l)
    return (s1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)

def featvals(rr, f):
    return [math.log10(x["scale"]) if f == "log10_scale" else float(x[f]) for x in rr]

def orientation(train, f):
    rk = rankdata(featvals(train, f))
    n1 = sum(1 for r in train if r["_lab"])
    mp = sum(x for x, r in zip(rk, train) if r["_lab"]) / max(n1, 1)
    mf = sum(x for x, r in zip(rk, train) if not r["_lab"]) / max(len(train) - n1, 1)
    return mp >= mf   # pass-higher

def score_set(eval_rows, features, orient):
    n = len(eval_rows); cols = []
    for f in features:
        rk = rankdata(featvals(eval_rows, f))
        z = [(v - (n + 1) / 2) / ((n ** 2 - 1) / 12.0) ** 0.5 for v in rk]
        if not orient[f]: z = [-v for v in z]
        cols.append(z)
    return [sum(c[i] for c in cols) / len(cols) for i in range(n)]

DERIVED = ["collapse_work_ratio", "fan_in_sum", "w_max", "w_mean_abs", "w_rel_spread", "B", "K", "H"]
for r in rows: r["_lab"] = bool(1 if r.get("fold32_pass", r.get("label")) else 0)

def lofo(variant):
    """returns {heldout_family: auc} for two-class families + aux"""
    out = {}
    for h in TWO_CLASS:
        train = [r for r in rows if r["family"] != h]
        evalr = [r for r in rows if r["family"] == h]
        orient = {f: orientation(train, f) for f in variant}
        s = score_set(evalr, variant, orient)
        out[h] = auc(s, [r["_lab"] for r in evalr])
    return out

leg2 = {}
leg2["lofo_derived"] = lofo(DERIVED)
leg2["lofo_derived_plus_scale"] = lofo(DERIVED + ["log10_scale"])
leg2["lofo_delta_cn"] = {h: leg2["lofo_derived_plus_scale"][h] - leg2["lofo_derived"][h] for h in TWO_CLASS}
# directed pairwise transfer (descriptive)
def xfer(tr_fam):
    train = [r for r in rows if r["family"] == tr_fam]
    return {h: auc(score_set([r for r in rows if r["family"] == h],
                             DERIVED, {f: orientation([r for r in rows if r["family"] == tr_fam], f) for f in DERIVED}),
                   [r["_lab"] for r in rows if r["family"] == h])
            for h in TWO_CLASS if h != tr_fam}
leg2["pairwise_transfer"] = {t: xfer(t) for t in TWO_CLASS}

# arm-A descriptive placement (pooled-rank basis, train orientation)
armA = [r for r in rows if r["family"] == "l2_14_fold"]
tr = [r for r in rows if r["family"] != "l2_14_fold"]
orient = {f: orientation(tr, f) for f in DERIVED}
pool = armA + tr
s_all = score_set(pool, DERIVED, orient)
s_map = dict()
for r, s in zip(pool, s_all): s_map[id(r)] = s
a_scores = [s_map[id(r)] for r in armA]
t_fail = [s_map[id(r)] for r in tr if not r["_lab"]]
t_pass = [s_map[id(r)] for r in tr if r["_lab"]]
leg2["armA_descriptive"] = dict(
    median_armA=sorted(a_scores)[len(a_scores) // 2],
    median_train_fail=sorted(t_fail)[len(t_fail) // 2],
    median_train_pass=sorted(t_pass)[len(t_pass) // 2],
    frac_armA_below_train_fail_median=sum(1 for s in a_scores if s < sorted(t_fail)[len(t_fail) // 2]) / len(a_scores))

leg2["primary_pass"] = all(v >= 0.80 for v in leg2["lofo_derived"].values())
leg2["family_collapse"] = {h: v < 0.70 for h, v in leg2["lofo_derived"].items() if v < 0.70}
leg2["cn_max_abs_delta"] = max(abs(v) for v in leg2["lofo_delta_cn"].values())
leg2["cn_scale_required_under_family_controls"] = bool(leg2["cn_max_abs_delta"] > 0.05)
print("LEG2 lofo derived:", {h: round(v, 4) for h, v in leg2["lofo_derived"].items()},
      "primary_pass:", leg2["primary_pass"])
print("LEG2 cn deltas:", {h: round(v, 4) for h, v in leg2["lofo_delta_cn"].items()})

# ---------------- leg 3: surrogate exposure ----------------
def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    n = len(a); ma = sum(ra) / n; mb = sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = (sum((x - ma) ** 2 for x in ra)) ** 0.5; db = (sum((y - mb) ** 2 for y in rb)) ** 0.5
    if da == 0.0 or db == 0.0: return float("nan")  # constant vector: undefined
    return num / (da * db)

leg3 = {}
rho = spearman([r["phi_ref"] for r in rows], [r["phi_cand"] for r in rows])
leg3["spearman_pooled"] = rho
leg3["spearman_per_family"] = {f: spearman([r["phi_ref"] for r in rows if r["family"] == f],
                                           [r["phi_cand"] for r in rows if r["family"] == f]) for f in FAMS}
leg3["lofo_derived_plus_phiref"] = lofo(DERIVED + ["phi_ref"])
leg3["lofo_derived_plus_phicand"] = lofo(DERIVED + ["phi_cand"])
leg3["phi_ref_gain"] = {h: leg3["lofo_derived_plus_phiref"][h] - leg2["lofo_derived"][h] for h in TWO_CLASS}
leg3["phi_cand_gap"] = {h: leg3["lofo_derived_plus_phiref"][h] - leg3["lofo_derived_plus_phicand"][h] for h in TWO_CLASS}
leg3["validity_pass"] = bool(rho >= 0.80)
leg3["utility_pass"] = bool(leg3["validity_pass"] and all(v <= 0.05 for v in leg3["phi_cand_gap"].values()))
leg3["verdict"] = ("exposure_surrogate_admitted" if leg3["utility_pass"]
                   else "degraded_tier_empirical_envelope_only")
print("LEG3 spearman pooled:", round(rho, 4), "per-family:",
      {k: round(v, 4) for k, v in leg3["spearman_per_family"].items()})
print("LEG3 phi_cand_gap:", {h: round(v, 4) for h, v in leg3["phi_cand_gap"].items()},
      "phi_ref_gain:", {h: round(v, 4) for h, v in leg3["phi_ref_gain"].items()},
      "verdict:", leg3["verdict"])

out = dict(when="2026-10-19T21:05Z", prereg="docstring written before run",
           n_rows=len(rows), phi_crosscheck_max_abs=xchk_max,
           leg2=leg2, leg3=leg3,
           phi_table={f"{k[0]}|s{k[1]}|sc{k[2]}|d{k[3]}": v for k, v in sorted(phi.items(), key=lambda kv: str(kv[0]))},
           env=dict(gpu="GB10 sm_120", torch=torch.__version__, tf32=False, atol=ATOL))
(RESULTS / "b3_leg23.json").write_text(json.dumps(out, indent=1))
print("saved", RESULTS / "b3_leg23.json")
