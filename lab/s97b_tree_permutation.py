#!/usr/bin/env python3
"""s97b (post-hoc, declared after s97): first point of the order-permutation cost curve.

Pure-PyTorch proxy for E2: hold the shape and the output math fixed, vary ONLY the reduction tree used for
the LayerNorm statistics (chunked sequential accumulation at several chunk sizes, plus a pairwise binary tree),
and measure the evaluator ratio r=|A-X|/(1e-4+1e-4|X|) extremum and frac_over at the shipped 1e-2 gate against
the true fp32 reference (F.layer_norm). fp64 truth is recorded for attribution. Declared post-hoc: this is not
the CUDA pair of E2; it tests whether a tree permutation has a measurable, monotone cost at all.
"""
import json, os, statistics, itertools
import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False
SHAPE = (16, 64, 256, 256)
DRAWS = 4
CHUNKS = [1048576, 262144, 65536, 16384, 4096]
OUT = "results/s97b_tree_permutation.jsonl"
rows = []

def stats_chunked(x, chunk):
    # sequential accumulation over flattened row, fp32, declared order = chunk-major sequential
    flat = x.reshape(x.shape[0], -1)
    n = flat.shape[1]
    acc = torch.zeros(flat.shape[0], device=x.device, dtype=torch.float32)
    acc2 = torch.zeros_like(acc)
    for i in range(0, n, chunk):
        c = flat[:, i:i+chunk]
        acc = acc + c.sum(dim=1)
        acc2 = acc2 + (c*c).sum(dim=1)
    mean = acc / n
    var = acc2 / n - mean*mean
    return mean, var, n

def stats_pairwise(x, chunk):
    # pairwise binary tree over chunk-level partials (fan-in 2), same chunk boundaries
    flat = x.reshape(x.shape[0], -1)
    n = flat.shape[1]
    parts = []
    parts2 = []
    for i in range(0, n, chunk):
        c = flat[:, i:i+chunk]
        parts.append(c.sum(dim=1)); parts2.append((c*c).sum(dim=1))
    def tree(ps):
        while len(ps) > 1:
            nxt = []
            for a, b in zip(ps[0::2], ps[1::2]):
                nxt.append(a+b)
            if len(ps) % 2: nxt.append(ps[-1])
            ps = nxt
        return ps[0]
    mean = tree(parts) / n
    var = tree(parts2) / n - mean*mean
    return mean, var, n

for d in range(DRAWS):
    torch.manual_seed(9700 + d)
    x = torch.randn(SHAPE, device="cuda", dtype=torch.float32)
    w = torch.randn(SHAPE[1:], device="cuda", dtype=torch.float32)
    b = torch.randn(SHAPE[1:], device="cuda", dtype=torch.float32)
    ref = F.layer_norm(x, SHAPE[1:], w, b, 1e-5)
    ref64 = F.layer_norm(x.double(), SHAPE[1:], w.double(), b.double(), 1e-5)
    # reference's own error vs fp64 truth, in evaluator units
    e_ref = (ref.double() - ref64).abs()
    rho_ref = float((e_ref / (1e-4 + 1e-4*ref64.abs())).max())
    for label, fn, chunk in ([("seq", stats_chunked, c) for c in CHUNKS] + [("pair", stats_pairwise, c) for c in CHUNKS]):
        mean, var, n = fn(x, chunk)
        y = (x - mean.view(-1,1,1,1)) * torch.rsqrt(var.clamp_min(0).view(-1,1,1,1) + 1e-5) * w + b
        e = (y - ref).abs()
        r = e / (1e-4 + 1e-4*ref.abs())
        rho = float(r.max()); idx = int(r.argmax())
        xat = float(ref.flatten()[idx])
        frac_over = float((r > 1).float().mean())
        rel = float((e/ref.abs().clamp_min(1e-30)).flatten()[idx])
        rows.append({"draw": d, "seed": 9700+d, "variant": label, "chunk": chunk,
                     "n_chunks": (n + chunk - 1)//chunk,
                     "rho": rho, "frac_over": frac_over, "absX_at_argmax": xat,
                     "rel_at_argmax": rel, "rho_ref_vs_fp64": rho_ref,
                     "max_abs_err": float(e.max())})
        print(f"d{d} {label} c={chunk}: rho={rho:.4f} frac={frac_over:.3e} |X|={xat:.3e}", flush=True)

os.makedirs("results", exist_ok=True)
with open(OUT, "w") as f:
    for r in rows: f.write(json.dumps(r)+"\n")

# analysis: does rho grow with tree distance (levels = n_chunks-1 for seq, log2 for pair)?
summ = {"by_variant": {}, "draws": DRAWS, "shape": list(SHAPE)}
for label in ["seq","pair"]:
    for chunk in CHUNKS:
        rs=[r for r in rows if r["variant"]==label and r["chunk"]==chunk]
        summ["by_variant"][f"{label}_{chunk}"] = {
            "n_chunks": rs[0]["n_chunks"],
            "rho_mean": statistics.mean(r["rho"] for r in rs),
            "rho_min": min(r["rho"] for r in rs), "rho_max": max(r["rho"] for r in rs),
            "frac_over_mean": statistics.mean(r["frac_over"] for r in rs),
            "absX_at_argmax_range": [min(r["absX_at_argmax"] for r in rs), max(r["absX_at_argmax"] for r in rs)],
            "rho_ref_mean": statistics.mean(r["rho_ref_vs_fp64"] for r in rs),
        }
# spearman between log2(n_chunks) and rho, within variant
def spearman(a,b):
    ra={v:i for i,v in enumerate(sorted(a))}; rb={v:i for i,v in enumerate(sorted(b))}
    import math
    n=len(a); da=[ra[v] for v in a]; db=[rb[v] for v in b]
    ma=sum(da)/n; mb=sum(db)/n
    num=sum((x-ma)*(y-mb) for x,y in zip(da,db))
    den=math.sqrt(sum((x-ma)**2 for x in da)*sum((y-mb)**2 for y in db))
    return num/den if den else 0.0
for label in ["seq","pair"]:
    xs=[]; ys=[]
    for chunk in CHUNKS:
        rs=[r for r in rows if r["variant"]==label and r["chunk"]==chunk]
        xs += [rs[0]["n_chunks"]]*len(rs); ys += [r["rho"] for r in rs]
    summ[f"spearman_nchunks_rho_{label}"] = spearman(xs, ys)
summ["verdict_note"] = "descriptive first point; no kill clause applied (s97b is post-hoc)"
with open("results/s97b_tree_permutation_summary.json","w") as f: json.dump(summ,f,indent=1)
print(json.dumps(summ)[:1200])
