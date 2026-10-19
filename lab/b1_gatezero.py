"""B1 gate-zero (exploitation_plan.md): is the L2/51 fold still alive under compile?

Pre-registered BEFORE run 2026-10-19 ~11:35Z (this docstring; DIARY same tick).
Family: task-51 linear-mean reassociation (sylve L2/51). Reference graph EXACT per
KernelBench 423217d level2/51: Linear -> x - subtract(H) -> mean(dim1,keepdim)
-> logsumexp(dim1) [identity, kept for graph parity] -> gelu -> + x.
Factored: gelu(x@mean(W,0) + (b-s).mean()) -> logsumexp(1) -> + x.  FP32 strict
(float32_matmul_precision highest, TF32 off) both arms; no_grad; inference.

Grid: c := out-features H (mean fan-in = fold shrink factor). K=1024 fixed,
H in {8,128,2048,8192} x B in {64,512,4096} (12 cells) + ANCHOR (B,K,H)=
(2048,8192,8192) = the school showcase. Params from nn.Linear init; inputs
torch.rand (benchmark distribution); per-cell seed 1000+i.

PRIMARY (pre-registered): per cell, 15 paired reps (flush 256MB between reps,
per-rep arm order shuffled, seed 1237+i): d_r=(t_cref-r - t_cfac-r)/t_cref-r;
cell PASSES iff mean(d)>0.05 and paired t>=2 (n=15, house convention).
LANE VERDICT: ALIVE iff ANCHOR passes AND >=4/12 grid cells pass;
F1-TIE otherwise (safety-half-only per plan). Passing cells delimit the
grammar's domain either way -- recorded per cell, no third state.

Also recorded: compile walls (NOT charged), correctness maxabs factored-vs-ref
and ref-vs-fp64 (reference-relative gate 1e-4 before timing; a correctness FAIL
marks the cell and it cannot pass).

Run on spark: python3 b1_gatezero.py -> results/b1_gatezero.jsonl (incr, rows
skipped on rerun) + results/b1_gatezero.json (final, verdict).
"""
import json, os, time, random, math
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUTL = os.path.join(RES, "b1_gatezero.jsonl")
OUT = os.path.join(RES, "b1_gatezero.json")
DATE = time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())
SEED = 1237
FLUSH_BYTES = 256 * 2**20
REPS, WARMUP = 15, 3

GRID = [(h, b, 1024) for h in (8, 128, 2048, 8192) for b in (64, 512, 4096)]
ANCHOR = (8192, 2048, 8192)  # (H, B, K)
CELLS = GRID + [ANCHOR]


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


def make_cell(H, B, K, seed):
    torch.manual_seed(seed)
    lin = torch.nn.Linear(K, H)
    W = lin.weight.detach().float().cuda()
    b = lin.bias.detach().float().cuda()
    g = torch.Generator().manual_seed(seed + 1)
    s = torch.randn(H, generator=g).float().cuda()
    x = torch.rand(B, K, generator=torch.Generator().manual_seed(seed + 2)).float().cuda()
    return x, W, b, s


def timed(fns, flush, reps, rng):
    for _ in range(WARMUP):
        for f in fns:
            f()
    torch.cuda.synchronize()
    ts = {}
    for r in range(reps):
        order = [0, 1]
        rng.shuffle(order)
        for arm in order:
            if flush is not None:
                flush.zero_()
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); fns[arm](); e.record(); torch.cuda.synchronize()
            ts.setdefault(r, {})[arm] = s.elapsed_time(e)
    return ts


def paired(ts, reps):
    d = []
    for r in range(reps):
        a, cf = ts[r][0], ts[r][1]
        d.append((a - cf) / a if a > 0 else 0.0)
    n = len(d)
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    sd = math.sqrt(var)
    if sd > 0:
        t = m / (sd / math.sqrt(n))
    else:
        t = float("inf") if m > 0 else 0.0
    return m, t


def main():
    os.makedirs(RES, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.float32, device="cuda")
    done = {}
    if os.path.exists(OUTL):
        for ln in open(OUTL):
            ln = ln.strip()
            if ln:
                r = json.loads(ln)
                done[r["cell_key"]] = r
    out = open(OUTL, "a")
    rows = []
    for i, (H, B, K) in enumerate(CELLS):
        key = "H%d_B%d_K%d" % (H, B, K)
        if key in done:
            rows.append(done[key])
            print("skip", key, flush=True)
            continue
        x, W, b, s = make_cell(H, B, K, 1000 + i)
        with torch.no_grad():
            r64 = ref_fwd(x.double(), W.double(), b.double(), s.double())
            t_ref = ref_fwd(x, W, b, s)
            t_fac = fac_fwd(x, W, b, s)
            err_fr = (t_fac - t_ref).abs().max().item()
            err_64 = (t_ref.double() - r64).abs().max().item()
            ok = err_fr <= 1e-4
            t0 = time.time(); c_ref = torch.compile(ref_fwd); c_ref(x, W, b, s); w_ref = time.time() - t0
            t0 = time.time(); c_fac = torch.compile(fac_fwd); c_fac(x, W, b, s); w_fac = time.time() - t0
            fns = [(lambda: c_ref(x, W, b, s)), (lambda: c_fac(x, W, b, s))]
            ts = timed(fns, flush, REPS, random.Random(SEED + i))
        m, t = paired(ts, REPS)
        med = lambda k: sorted(ts[r][k] for r in range(REPS))[REPS // 2]
        row = dict(cell_key=key, H=H, B=B, K=K, date=DATE,
                   correct=int(ok), err_factored_vs_ref=err_fr, err_ref_vs_fp64=err_64,
                   cref_ms=med(0), cfac_ms=med(1),
                   rel_mean=m, t_stat=t,
                   pass_=bool(ok and m > 0.05 and t >= 2),
                   compile_wall_s=[round(w_ref, 2), round(w_fac, 2)],
                   times={str(k): v for k, v in ts.items()})
        out.write(json.dumps(row) + "\n"); out.flush()
        rows.append(row)
        print("done", key, "cref=%.4f cfac=%.4f rel=%.4f t=%.2f pass=%s err=%.2e"
              % (row["cref_ms"], row["cfac_ms"], m, t, row["pass_"], err_fr), flush=True)
    out.close()
    bykey = {r["cell_key"]: r for r in rows}
    akey = "H%d_B%d_K%d" % ANCHOR
    anch = bykey.get(akey)
    grid = [bykey[k] for k in bykey if k != akey]
    npass = sum(1 for r in grid if r["pass_"])
    verdict = "ALIVE" if (anch is not None and anch["pass_"] and npass >= 4) else "F1_TIE"
    res = dict(date=DATE, verdict=verdict, anchor_pass=bool(anch and anch["pass_"]),
               grid_pass=npass, grid_n=len(grid),
               cells=[{k: r[k] for k in ("cell_key", "cref_ms", "cfac_ms", "rel_mean",
                                          "t_stat", "pass_", "correct",
                                          "err_factored_vs_ref")} for r in rows])
    json.dump(res, open(OUT, "w"), indent=1)
    print("VERDICT", verdict, "anchor_pass", anch["pass_"] if anch else None,
          "grid", npass, "/", len(grid), flush=True)


if __name__ == "__main__":
    main()
