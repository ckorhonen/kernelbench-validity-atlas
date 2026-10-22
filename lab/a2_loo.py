"""A2 LOO offset-recovery probe -- PREREG lab/a2_loo_prereg.md (freeze commit
eb973d9, file sha256 356b2f25...). SINGLE evaluation pass after freeze; rerun
only for implementation bugs (logged in impl_notes).

Machinery identical to a1_orthdecile (imported): per-(src,arm) z of w_mean_abs /
w_max -> L; global-z cwr -> Oc; Lo = residual of OLS L ~ 1 + Oc. M-C fold fit
y ~ cell_dummies + 1 + Lo (numpy lstsq min-norm, a1 verbatim). Train-cell offset
o_c := its dummy coefficient (prereg, literal). Held-out row prediction =
intercept + b1*Lo + g(phi(cell)); g = ridge(lambda=1, correlation scale,
intercept via centering) from cell feature vector phi -> o_c.

phi blocks (prereg-locked):
  struct: cell-mean of log1p H/B/K/HBK, cwr, fan_in_sum, scale, w_mean_abs,
          w_rel_spread, w_max, mean|z_raw|, std(z_raw), log1p n_rows
  marg:   cell-mean of per-row input {mean,std,kurt,max|.|} and weight {same 4};
          tensors regenerated verbatim per b4_guard_v1.py / b5_stage2.py
          (row seed/arm/scale; cuda RNG where the originals used it).
          Kurtosis = m4/m2^2, biased moments, float64 accumulation.
  src:    one-hot (b4, b5); unseen level -> zero column (recorded per fold).
  cwr:    cell-mean collapse_work_ratio alone (ablation c).
  z_raw:  b4 rows carry the committed dict; b5 has none -- recomputed with the
          a1 group-z recipe on the same three fields (declared, label-free).

Ablations (descriptive; bar .30 applies to (a) on primary data only):
  a full | b struct+src | c cwr | d marg+src.
PASS: mean LOO offset-R2 >= 0.30, ablation (a), primary b4+b5. Sensitivity
b4+b5_stage2r (repaired b5), same rule, reported not pass-bearing.

Run: spark venv python. Output results/a2_loo.json + .jsonl per-fold.
"""
import os
import json, pathlib, time, hashlib, math
import numpy as np
import torch

import a1_orthdecile as A

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"
PREREG = HERE / "a2_loo_prereg.md"

STRUCT = ["log1p_H", "log1p_B", "log1p_K", "log1p_HBK", "collapse_work_ratio",
          "fan_in_sum", "scale", "w_mean_abs", "w_rel_spread", "w_max",
          "zraw_absmean", "zraw_std", "log1p_nrows"]
MARG = ["in_mean", "in_std", "in_kurt", "in_maxabs",
        "w_mean", "w_std", "w_kurt", "w_maxabs"]
NOTES = []
KEEP = (("a_full", ("struct", "marg", "src")),
        ("b_no_marg", ("struct", "src")),
        ("c_cwr_only", ("cwr",)),
        ("d_marg_only", ("marg", "src")))


# ---------------- verbatim tensor regeneration ----------------
_CACHE = {"b4_base": {}, "b4_x": {}, "b5_x": {}}


def b4_base(seed, H, K):
    key = ("b4_base", seed, H, K)
    if key not in _CACHE:
        torch.manual_seed(seed)
        lin = torch.nn.Linear(K, H)
        _CACHE[key] = (lin.weight.detach().clone(), lin.bias.detach().clone())
    return _CACHE[key]


def b4_row_tensors(r):
    H, B, K, seed = int(r["H"]), int(r["B"]), int(r["K"]), int(r["seed"])
    W0, _b0 = b4_base(seed, H, K)
    scale = float(r["scale"])
    W = W0 * scale
    if r["arm"] == "ordmag":
        key = ("b4_x", "o", seed, B, K)
        if key not in _CACHE:
            _CACHE[key] = torch.rand(B, K, generator=torch.Generator().manual_seed(seed + 2))
        x = _CACHE[key]
    else:  # cancel: ci from scale, per-draw cuda RNG (verbatim b4 v1)
        ci = 0 if abs(scale - 1e4) < 1e-9 else 1
        draw = int(r.get("draw", 0))
        key = ("b4_x", "c", ci, draw, K)
        if key not in _CACHE:
            torch.manual_seed(4800 + ci * 100 + draw)
            if torch.cuda.is_available():
                x = torch.rand(1, K, device="cuda").cpu()
            else:
                NOTES.append("cuda unavailable: b4 cancel r_ drawn on CPU (device deviation)")
                x = torch.rand(1, K)
            _CACHE[key] = x
        x = _CACHE[key].expand(B, K)
    return x.contiguous(), W


def b5_globals(repo=os.path.expanduser("~/kernelbench/KernelBench/KernelBench/level1")):
    key = ("b5g", repo)
    if key in _CACHE:
        return _CACHE[key]
    import importlib.util
    p = next(pathlib.Path(repo).glob("40_*.py"))
    spec = importlib.util.spec_from_file_location("task40", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.Model(*mod.get_init_inputs()).cuda().eval()
    ln = [m for m in model.modules() if isinstance(m, torch.nn.LayerNorm)][0]
    torch.manual_seed(123)
    shape = tuple(mod.get_inputs()[0].shape)
    g_task = ln.weight.detach().clone()
    gg = torch.Generator().manual_seed(1234)
    g_rand = torch.randn(ln.weight.shape, generator=gg)
    out = (shape, g_task, g_rand)
    _CACHE[key] = out
    return out


def b5_row_tensors(r, shape, g_task, g_rand):
    didx = {"randu": 0, "randn": 1, "randn_p1000": 2}[r["arm"]]
    seed = int(r["seed"])
    key = ("b5_x", seed, didx)
    if key not in _CACHE:
        torch.manual_seed(9000 + seed * 10 + didx)
        if torch.cuda.is_available():
            if r["arm"] == "randu":
                x = torch.rand(shape, device="cuda")
            else:
                x = torch.randn(shape, device="cuda")
        else:
            NOTES.append("cuda unavailable: b5 input drawn on CPU (device deviation)")
            x = torch.rand(shape) if r["arm"] == "randu" else torch.randn(shape)
        if r["arm"] == "randn_p1000":
            x = x + 1000.0
        _CACHE[key] = x.cpu()
    x = _CACHE[key]
    g_base = g_task if r["gamma0"] == "task_default" else g_rand
    return x, g_base * float(r["scale"])


def tstats(t):
    v = t.detach().double().flatten()
    mu = v.mean()
    m2 = float((v - mu).pow(2).mean())
    m4 = float((v - mu).pow(4).mean())
    kurt = (m4 / m2 ** 2) if m2 > 0 else 1.0  # degenerate (e.g. LN ones gamma): convention 1.0, recorded
    return [float(v.mean()), float(v.std(unbiased=True)), kurt,
            float(v.abs().max())]


def marg_for(rows):
    need_b5 = any(r["_src"].startswith("b5") for r in rows)
    b5g = b5_globals() if need_b5 else None
    out = {}
    for r in rows:
        if r["_src"].startswith("b4"):
            x, w = b4_row_tensors(r)
        else:
            x, w = b5_row_tensors(r, *b5g)
        out[id(r)] = tuple(tstats(x) + tstats(w))
    return out


# ---------------- a1-verbatim preproc + zraw pairs ----------------
def prep(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["_src"], r["arm"]), []).append(r)
    for g in groups.values():
        for f in ("w_mean_abs", "w_max", "collapse_work_ratio"):
            mu, sd = A.zparams([float(x[f]) for x in g])
            for x in g:
                x["z_" + f] = (float(x[f]) - mu) / sd
    gm, gsd = A.zparams([float(r["collapse_work_ratio"]) for r in rows])
    for r in rows:
        r["L"] = (r["z_w_mean_abs"] + r["z_w_max"]) / 2.0
        r["Oc"] = (float(r["collapse_work_ratio"]) - gm) / gsd
    b0, b1 = A.ols1([r["Oc"] for r in rows], [r["L"] for r in rows])
    for r in rows:
        r["Lo"] = r["L"] - (b0 + b1 * r["Oc"])
    # zraw pair: committed dict (b4) or a1-recipe recompute (b5)
    for r in rows:
        z = r.get("z_raw")
        if isinstance(z, dict) and z:
            v = [float(x) for x in z.values()]
            r["_zraw_pair"] = (sum(abs(x) for x in v) / len(v), float(np.std(v)))
        else:
            v = [r["z_w_mean_abs"], r["z_w_max"], r["z_collapse_work_ratio"]]
            r["_zraw_pair"] = (sum(abs(x) for x in v) / len(v), float(np.std(v)))
    return dict(orth_b0=b0, orth_b1=b1)


# ---------------- blocks / ridge / ablation ----------------
def build_blocks(rows):
    cells = sorted({r["cell"] for r in rows})
    per = {c: dict(n=0, s={}, s4=0, s5=0) for c in cells}
    for r in rows:
        d = per[r["cell"]]
        d["n"] += 1
        if r["_src"].startswith("b4"):
            d["s4"] += 1
        if r["_src"].startswith("b5"):
            d["s5"] += 1
        vals = dict(log1p_H=math.log1p(r["H"]), log1p_B=math.log1p(r["B"]),
                    log1p_K=math.log1p(r["K"]),
                    log1p_HBK=math.log1p(r["H"] * r["B"] * r["K"]),
                    collapse_work_ratio=float(r["collapse_work_ratio"]),
                    fan_in_sum=float(r["fan_in_sum"]), scale=float(r["scale"]),
                    w_mean_abs=float(r["w_mean_abs"]),
                    w_rel_spread=float(r["w_rel_spread"]), w_max=float(r["w_max"]),
                    zraw_absmean=r["_zraw_pair"][0], zraw_std=r["_zraw_pair"][1])
        for k, v in vals.items():
            d["s"][k] = d["s"].get(k, 0.0) + float(v)
    out = {}
    for c in cells:
        d, n = per[c], per[c]["n"]
        struct = [d["s"][k] / n for k in STRUCT if k != "log1p_nrows"]
        struct.insert(STRUCT.index("log1p_nrows"), math.log1p(n))
        out[c] = dict(struct=np.array(struct, float),
                      cwr=np.array([d["s"]["collapse_work_ratio"] / n], float),
                      src=np.array([1.0 if d["s4"] > 0 else 0.0,
                                    1.0 if d["s5"] > 0 else 0.0]))
    # marg filled per run (needs tensors); attached by caller as out[c]["marg"]
    return out, cells


def attach_marg(blocks, rows, marg):
    per = {r["cell"]: [] for r in rows}
    for r in rows:
        per[r["cell"]].append(marg[id(r)])
    for c, arr in per.items():
        blocks[c]["marg"] = np.array(arr, float).mean(0)
    return blocks


def ridge_fit(X, y, lam=1.0):
    X = np.asarray(X, float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xc = (X - mu) / sd
    yc = y - y.mean()
    beta = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ yc)
    return dict(beta=beta, mu=mu, sd=sd, b0=float(y.mean()))


def ridge_pred(m, x):
    xc = (np.asarray(x, float) - m["mu"]) / m["sd"]
    return float(m["b0"] + xc @ m["beta"])


def run_ablation(rows, cells, cidx, blocks, keep):
    for i, r in enumerate(rows):
        r["_i"] = i
    yall = [1.0 if r["label"] else 0.0 for r in rows]
    base = sum(yall) / len(rows)
    nc = len(cells)
    pred = [None] * len(rows)
    folds = []
    for hc in cells:
        tr = [r for r in rows if r["cell"] != hc]
        te = [r for r in rows if r["cell"] == hc]
        M = np.zeros((len(tr), nc + 2))
        yy = np.array([1.0 if r["label"] else 0.0 for r in tr])
        for i, r in enumerate(tr):
            M[i, cidx[r["cell"]]] = 1.0
            M[i, nc] = 1.0
            M[i, nc + 1] = r["Lo"]
        sol, *_ = np.linalg.lstsq(M, yy, rcond=None)
        Xtr = [np.concatenate([blocks[c][b] for b in keep])
               for c in cells if c != hc]
        ytr = np.array([float(sol[cidx[c]]) for c in cells if c != hc])
        Xte = np.concatenate([blocks[hc][b] for b in keep])
        gm = ridge_fit(Xtr, ytr)
        b0, b1 = float(sol[nc]), float(sol[nc + 1])
        gheld = ridge_pred(gm, Xte)
        tr_srcs = {c[:2] for c in cells if c != hc}
        unseen_src = int(hc[:2] not in tr_srcs)
        sse = sst = 0.0
        for r in te:
            v = b0 + b1 * r["Lo"] + gheld
            pred[r["_i"]] = v
            e = v - (1.0 if r["label"] else 0.0)
            sse += e * e
            sst += ((1.0 if r["label"] else 0.0) - base) ** 2
        folds.append(dict(cell=hc, n=len(te), sse=sse, sst=sst,
                          r2=(1 - sse / sst) if sst > 0 else None,
                          n_ridge_feats=int(Xte.size),
                          unseen_src=unseen_src))
    ok = [f for f in folds if f["r2"] is not None]
    r2s = [f["r2"] for f in ok]
    pairs = [(pred[i], yall[i]) for i in range(len(rows)) if pred[i] is not None]
    print("   fold R2s:", {f["cell"]: (round(f["r2"], 3) if f["r2"] is not None else None)
                            for f in folds}, flush=True)
    return dict(mean_loo_r2=float(np.mean(r2s)) if r2s else float("nan"),
                per_fold=folds, pooled_r2=A.r2(pairs, base),
                pooled_auc=A.auc([p for p, _ in pairs], [l for _, l in pairs]),
                n_folds=len(folds), n_r2_folds=len(ok))


def n_rows(rows):
    return len(rows)


def main():
    t0 = time.time()
    b4 = A.load("b4", "b4_guard_v1.jsonl")
    b5 = A.load("b5", "b5_stage2.jsonl")
    b5r = A.load("b5r", "b5_stage2r.jsonl")
    print("rows b4=%d b5=%d b5r=%d" % (len(b4), len(b5), len(b5r)), flush=True)

    out = dict(
        when=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        runtime_s=0,
        prereg_sha256=hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        prereg_commit="eb973d9",
        torch=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        device=("cuda" if torch.cuda.is_available() else "cpu"),
        impl_notes=NOTES + [
            "rerun log (prereg allows implementation-bug reruns): v1 nan kurt on degenerate tensors -> convention 1.0; "
            "src one-hot keyed off cell name (always-zero) -> now from row _src counts; log1p_nrows KeyError. Final run v2.",
            "b5-heldout fold: LN cell sits ~1e3 sd outside the b4 covator hull (raw input/weight marginals); "
            "frozen lambda=1 ridge extrapolates linearly -> its fold R2 dominates the mean. Per-fold table recorded; "
            "within-b4 12-fold mean is descriptive only, not an acceptance.",
        ],
        metric="mean over 13 leave-one-cell-out folds of 1-SSE/SST, SST about "
               "the global train-row mean (a1 s1 verbatim); PASS bar .30 on "
               "ablation (a) primary",
    )
    results = {}
    for name, data in (("primary", b4 + b5), ("sensitivity_b5r", b4 + b5r)):
        prep(data)
        marg = marg_for(data)
        blocks, cells = build_blocks(data)
        blocks = attach_marg(blocks, data, marg)
        cidx = {c: i for i, c in enumerate(cells)}
        res = {}
        for ab, keep in KEEP:
            res[ab] = run_ablation(data, cells, cidx, blocks, keep)
            print("%s %s meanR2=%.4f pooledR2=%.4f auc=%.4f" % (
                name[:4], ab, res[ab]["mean_loo_r2"], res[ab]["pooled_r2"],
                res[ab]["pooled_auc"]), flush=True)
        results[name] = dict(n=len(data), n_cells=len(cells), ablations=res)

    p = results["primary"]["ablations"]["a_full"]
    out["PASS"] = bool(p["mean_loo_r2"] >= 0.30)
    out["results"] = results
    out["runtime_s"] = round(time.time() - t0, 1)
    RES.mkdir(exist_ok=True)
    (RES / "a2_loo.json").write_text(json.dumps(out, indent=1, default=str))
    with open(RES / "a2_loo.jsonl", "w") as f:
        for name, r in results.items():
            for ab, res in r["ablations"].items():
                for fd in res["per_fold"]:
                    f.write(json.dumps(dict(run=name, ablation=ab, **fd)) + "\n")
    print("PASS(bar .30, a, primary):", out["PASS"],
          "| meanR2=%.4f" % p["mean_loo_r2"])
    print("sensitivity b4+b5r a_full meanR2=%.4f" %
          results["sensitivity_b5r"]["ablations"]["a_full"]["mean_loo_r2"])


if __name__ == "__main__":
    main()
