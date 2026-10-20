"""Direction audit: legacy (ref-scaled) vs evaluator-exact (cand-scaled) tolerance.

WHY (2026-10-20 ~07:25Z, post sylve reply caeAjrvj5oQTQcCW): pinned eval.py
(423217d, lines 804-806) calls torch.allclose(output, output_new, atol=t, rtol=t)
with output=reference model, output_new=candidate -> per torch docs the rtol term
scales |candidate|. ALL our harnesses (b2/b3/b4v0/b4v1) used limit=ATOL+RTOL*|ref|
(legacy direction). sylve's own audit found 0/1531 ratios in the ambiguous band
[0.999,1.001]; this is the analogous check on OUR artifacts. No freeze is refit:
train labels feed only the already-frozen guard (finality per b4v1 prereg); we
report label flips + held-out metrics under both directions, frozen t and basis
UNTOUCHED. This is a labeled sensitivity, not a v2.

Parts:
  A. b2 corpus regeneration (arms A/B/C verbatim) -> label flips legacy/exact,
     max ratio deltas, ambiguous-band counts. DESCRIPTIVE.
  B. b4 held-out regeneration (cells/seeds/arms verbatim) -> scores from frozen
     results/guard_v1.json (assert per-row equality with b4_guard_v1.jsonl
     legacy labels/scores first), then full metric block under BOTH label sets.
     The alive-verdict sensitivity is the deliverable.

Provenance: spark GB10 sm_120, torch 2.14.1+cu130, TF32 off, matmul highest.
Results -> results/b2_direction_audit.json, results/b4_direction_audit.json.
"""
import json, math, pathlib, hashlib, time
import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
ATOL = RTOL = 1e-4
ROOT = pathlib.Path("/home/ckorhonen/kb-lab")
RES = ROOT / "results"; RES.mkdir(exist_ok=True)

def both(a, b):
    """a=reference, b=candidate. Legacy: lim=AT+RT|a|; exact: lim=AT+RT|b|."""
    d = (a - b).abs()
    lr = ATOL + RTOL * a.abs(); lc = ATOL + RTOL * b.abs()
    return dict(pass_ref=bool((d <= lr).all()), pass_cand=bool((d <= lc).all()),
                mbr_ref=float((d / lr).max()), mbr_cand=float((d / lc).max()),
                n_flip_elem=int(((d <= lr) != (d <= lc)).sum()))

def load_task(n):
    import importlib.util
    p = next(pathlib.Path("/home/ckorhonen/kernelbench/KernelBench/KernelBench/level2").glob(f"{n}_*.py"))
    spec = importlib.util.spec_from_file_location(f"task{n}", p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

# ============================= PART A: b2 corpus =============================
summaries = []  # (tag, both-dict)

def summarizeA(tag, a, b):
    s = both(a, b); s["tag"] = tag; summaries.append(s); return s

m14 = load_task(14)
for seed in range(5):
    torch.manual_seed(seed)
    model = m14.Model(*m14.get_init_inputs()).cuda().eval()
    x = m14.get_inputs()[0].cuda()
    W = model.weight; sf = model.scaling_factor
    ref = model(x)
    f32 = (x @ W.sum(0)).unsqueeze(1) * (sf / 2)
    f64 = (x.double() @ W.sum(0).double()).unsqueeze(1) * (sf / 2)
    summarizeA(f"A_seed{seed}_fold32", ref, f32)
    summarizeA(f"A_seed{seed}_fold64", ref, f64)
    del model, x, ref, f32, f64
torch.cuda.empty_cache()

m51 = load_task(51)
torch.manual_seed(43)
m = m51.Model(*m51.get_init_inputs()).cuda().eval()
base_w = m.gemm.weight.detach().clone(); base_b = m.gemm.bias.detach().clone()
K = m51.get_inputs()[0].shape[1]
scales = [1.0, 100.0, 1e3, 1e4, 1e5, 1e6]
for si, scale in enumerate(scales):
    m.gemm.weight.copy_(base_w * scale); m.gemm.bias.copy_(base_b)
    for i in range(20):
        torch.manual_seed(4300 + si * 100 + i)
        r_ = torch.rand(1, K, device="cuda"); r_[0, 0] = 0
        x = r_.expand(2048, K).contiguous()
        m.subtract.copy_(m.gemm(x)[0])
        ref = m(x)
        v = m.gemm.weight.mean(0); c = m.gemm.bias.mean() - m.subtract.mean()
        pre = (x @ v + c)
        f32 = torch.nn.functional.gelu(pre.unsqueeze(1)) + x
        f64 = torch.nn.functional.gelu((x.double() @ m.gemm.weight.double().mean(0)
              + (m.gemm.bias.double().mean() - m.subtract.double().mean())).float().unsqueeze(1)) + x
        summarizeA(f"B_s{scale:g}_d{i}_fold32", ref, f32)
        summarizeA(f"B_s{scale:g}_d{i}_fold64", ref, f64)
        del x
torch.cuda.empty_cache()
del m

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
        f32 = torch.nn.functional.gelu(pre.unsqueeze(1)) + x
        f64 = torch.nn.functional.gelu((x.double() @ m.gemm.weight.double().mean(0)
              + (m.gemm.bias.double().mean() - m.subtract.double().mean())).float().unsqueeze(1)) + x
        summarizeA(f"C_s{seed}_i{si}_fold32", ref, f32)
        summarizeA(f"C_s{seed}_i{si}_fold64", ref, f64)
    del m, x
    torch.cuda.empty_cache()

n_flip_rows = sum(1 for s in summaries if s["pass_ref"] != s["pass_cand"])
amb = [s for s in summaries if 0.999 <= min(s["mbr_ref"], s["mbr_cand"]) and max(s["mbr_ref"], s["mbr_cand"]) <= 1.001]
maxd = max(abs(s["mbr_ref"] - s["mbr_cand"]) / max(s["mbr_ref"], 1e-12) for s in summaries)
partA = dict(n_summaries=len(summaries), n_label_flips=n_flip_rows,
             flips=[s["tag"] for s in summaries if s["pass_ref"] != s["pass_cand"]],
             n_ambiguous_band=len(amb), tags_ambiguous=[s["tag"] for s in amb],
             max_rel_mbr_delta=maxd,
             elem_flips_total=sum(s["n_flip_elem"] for s in summaries),
             note="legacy=ref-scaled (our harnesses); exact=cand-scaled (evaluator 423217d L804)")
print("PART A:", json.dumps({k: partA[k] for k in ("n_summaries", "n_label_flips", "n_ambiguous_band", "max_rel_mbr_delta", "elem_flips_total")}))
(RES / "b2_direction_audit.json").write_text(json.dumps(
    dict(when=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), partA=partA,
         rows=summaries), indent=1, default=float))

# ===================== PART B: b4 held-out relabel ==========================
def ref_fwd(x, W, b, s):
    x0 = x
    y = F.linear(x, W, b) - s
    y = torch.mean(y, dim=1, keepdim=True)
    y = torch.logsumexp(y, dim=1, keepdim=True)
    y = F.gelu(y)
    return y + x0

def fac_fwd(x, W, b, s):
    x0 = x
    y = torch.matmul(x, W.mean(dim=0)) + (b - s).mean()
    y = y.reshape(-1, 1)
    y = torch.logsumexp(y, dim=1, keepdim=True)
    y = F.gelu(y)
    return y + x0

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

def featval(r, f):
    return math.log10(r["scale"]) if f == "log10_scale" else float(r[f])

DERIVED = ["collapse_work_ratio", "fan_in_sum", "w_max", "w_mean_abs", "w_rel_spread", "B", "K", "H"]
fz = json.loads((RES / "guard_v1.json").read_text())
ORIENT, COLS, T_L, KEPT = fz["orientation"], {k: list(v) for k, v in fz["corpus_columns"].items()}, fz["threshold"], fz["kept_features"]

def deployed_scores(eval_rows):
    out = []; n = len(next(iter(COLS.values()))) + 1
    sd = ((n ** 2 - 1) / 12.0) ** 0.5; mu = (n + 1) / 2
    for r in eval_rows:
        zs = []
        for f in KEPT:
            rk = rankdata(COLS[f] + [featval(r, f)])[-1]
            z = (rk - mu) / sd
            if not ORIENT[f]: z = -z
            zs.append(z)
        out.append(sum(zs) / len(zs))
    return out

CELLS = [(h, b, k) for h in (64, 512, 4096) for b in (256, 2048) for k in (1024, 8192)]
heldout = []
for (H, B, K) in CELLS:
    for seed in (48, 49):
        torch.manual_seed(seed)
        lin = torch.nn.Linear(K, H)
        W0 = lin.weight.detach().float().cuda(); b0 = lin.bias.detach().float().cuda()
        g = torch.Generator().manual_seed(seed + 1)
        s0 = torch.randn(H, generator=g).float().cuda()
        x_ord = torch.rand(B, K, generator=torch.Generator().manual_seed(seed + 2)).float().cuda().cuda()
        base = dict(cell=f"H{H}_B{B}_K{K}", seed=seed, B=B, K=K, H=H,
                    collapse_work_ratio=B * K * H / (H * K + K), fan_in_sum=K)
        for si in range(24):
            scale = float(10 ** (6.0 * si / 23)); W = W0 * scale
            ref = ref_fwd(x_ord, W, b0, s0); T = fac_fwd(x_ord, W, b0, s0)
            s_ = both(ref, T); a = W.abs()
            heldout.append(dict(base, arm="ordmag", scale=scale,
                                lab_ref=s_["pass_ref"], lab_cand=s_["pass_cand"],
                                mbr_ref=s_["mbr_ref"], mbr_cand=s_["mbr_cand"],
                                w_max=float(a.max()), w_mean_abs=float(a.mean()),
                                w_rel_spread=float(W.std() / a.mean())))
        for ci, scale in ((0, 1e4), (1, 1e6)):
            W = W0 * scale
            for i in range(5):
                torch.manual_seed(4800 + ci * 100 + i)
                r_ = torch.rand(1, K, device="cuda"); r_[0, 0] = 0
                x = r_.expand(B, K).contiguous()
                s_0 = F.linear(x, W, b0)[0].contiguous()
                ref = ref_fwd(x, W, b0, s_0); T = fac_fwd(x, W, b0, s_0)
                s_ = both(ref, T); a = W.abs()
                heldout.append(dict(base, arm="cancel", scale=scale, draw=i,
                                    lab_ref=s_["pass_ref"], lab_cand=s_["pass_cand"],
                                    mbr_ref=s_["mbr_ref"], mbr_cand=s_["mbr_cand"],
                                    w_max=float(a.max()), w_mean_abs=float(a.mean()),
                                    w_rel_spread=float(W.std() / a.mean())))
        del W0, b0, s0, x_ord
        torch.cuda.empty_cache()
assert len(heldout) == 816, len(heldout)

# sanity: legacy labels/mbr must reproduce the frozen run's jsonl
orig = [json.loads(l) for l in open(RES / "b4_guard_v1.jsonl")]
assert len(orig) == len(heldout)
mismatch = 0
for o, n_ in zip(orig, heldout):
    if o["label"] != n_["lab_ref"] or abs(o["max_budget_ratio"] - n_["mbr_ref"]) > 1e-9 * max(1, abs(o["max_budget_ratio"])):
        mismatch += 1
print("legacy reproduction mismatches:", mismatch)

hs = deployed_scores(heldout)
for r, s in zip(heldout, hs): r["score"] = s

def metrics(labkey):
    lab = [bool(r[labkey]) for r in heldout]
    pooled = auc([r["score"] for r in heldout], lab)
    by_arm = {a: auc([r["score"] for r in heldout if r["arm"] == a],
                     [l for r, l in zip(heldout, lab) if r["arm"] == a])
              for a in ("ordmag", "cancel")}
    acc = [r for r, l in zip(heldout, lab) if r["score"] >= T_L]
    prec = (sum(1 for r in acc if r[labkey == "lab_cand" and "lab_cand" or "lab_ref"] if False) ,)
    prec = (sum(1 for r, l in zip(heldout, lab) if r["score"] >= T_L and l) / len(acc)) if acc else None
    cells = {c: auc([r["score"] for r in heldout if r["cell"] == c],
                    [l for r, l in zip(heldout, lab) if r["cell"] == c])
             for c in {r["cell"] for r in heldout}}
    return dict(n_pass=sum(lab), n_fail=len(lab) - sum(lab), pooled_auc=pooled,
                auc_by_arm=by_arm, accepted_n=len(acc), accepted_precision=prec,
                accepted_frac=len(acc) / len(heldout),
                cells_auc_min=min(v for v in cells.values() if v == v),
                cells_auc_max=max(cells.values()), collapse_flags={a: v for a, v in by_arm.items() if v < 0.70})

m_ref = metrics("lab_ref"); m_cand = metrics("lab_cand")
n_flip = sum(1 for r in heldout if r["lab_ref"] != r["lab_cand"])
alive = lambda m_: (m_["pooled_auc"] >= 0.80) and (not m_["collapse_flags"]) and (m_["accepted_frac"] >= 0.20)
partB = dict(n_heldout=816, legacy_reproduction_mismatches=mismatch, n_label_flips=n_flip,
             legacy=m_ref, exact_cand_scaled=m_cand,
             alive_legacy=alive(m_ref), alive_exact=alive(m_cand),
             frozen_t=T_L, kept=KEPT,
             note="scores/features/limit-threshold frozen from guard_v1.json; ONLY pass labels re-derived")
print("PART B:", json.dumps({k: partB[k] for k in ("n_label_flips", "legacy_reproduction_mismatches",
      "alive_legacy", "alive_exact", "frozen_t")}, default=str))
print("legacy pooled/arm:", m_ref["pooled_auc"], m_ref["auc_by_arm"], "acc", m_ref["accepted_n"], m_ref["accepted_precision"])
print("exact  pooled/arm:", m_cand["pooled_auc"], m_cand["auc_by_arm"], "acc", m_cand["accepted_n"], m_cand["accepted_precision"])
(RES / "b4_direction_audit.json").write_text(json.dumps(partB, indent=1, default=float))
print("done")
