"""E2 v1: contested boundary -- split-K vs flat on small-out reductions (sum over K).
Claim under test: 'split-K wins when the output is too small to fill the machine'.
Protocol (identity: monotonicity check BEFORE bisection):
  lattice over (K, M), bytes capped; variants: torch, torch.compile (spot), triton flat
  (grid over M-blocks, loop K), triton split-K (grid M-blocks x S, partials + reduce).
  Record derived coords: ctas_flat=M/BM, ctas_split=M/BM*S, waves=ctas/SM.
  v1 output: per-point argmin table + regime boundary m*(K) + collapse check in
  programs-per-SM coords. Bisection refinement only after monotonicity verdict.
Numerics: fp32; split-K reorders sums -- max|diff| vs torch recorded per point.
"""
import json, os, time
import torch, triton, triton.language as tl
import torch._dynamo

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "e2_splitk.json")
DATE = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
BYTE_CAP = 128 * 2**20
FLUSH_BYTES = 256 * 2**20

@triton.jit
def reduce_flat(X, O, K, M, BM: tl.constexpr, KU: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BM + tl.arange(0, BM)
    acc = tl.zeros([BM], dtype=tl.float32)
    for k in range(0, K, KU):
        for u in tl.static_range(KU):
            kk = k + u
            row = tl.load(X + kk * M + offs, mask=offs < M, other=0.0)
            acc += row
    tl.store(O + offs, acc, mask=offs < M)

@triton.jit
def split_partial(X, P, K, M, S, BM: tl.constexpr):
    pid_m = tl.program_id(0); pid_s = tl.program_id(1)
    k_per = K // S
    k0 = pid_s * k_per
    offs = pid_m * BM + tl.arange(0, BM)
    acc = tl.zeros([BM], dtype=tl.float32)
    for k in range(k0, k0 + k_per):
        row = tl.load(X + k * M + offs, mask=offs < M, other=0.0)
        acc += row
    tl.store(P + pid_s * M + offs, acc, mask=offs < M)

@triton.jit
def split_reduce(P, O, M, S, BM: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BM + tl.arange(0, BM)
    acc = tl.zeros([BM], dtype=tl.float32)
    for s in range(S):
        acc += tl.load(P + s * M + offs, mask=offs < M, other=0.0)
    tl.store(O + offs, acc, mask=offs < M)

def evt_time(fn, warmup=2, reps=10, flush=None):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(reps):
        if flush is not None: flush.zero_()
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort(); n = len(ts)
    return {"med_ms": ts[n//2], "iqr_ms": [ts[int(.25*n)], ts[min(n-1,int(.75*n))]], "n_reps": n}

def main():
    dev = torch.device("cuda:0")
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    flush = torch.empty(FLUSH_BYTES//4, dtype=torch.float32, device=dev)
    Ks = [2**k for k in (12,14,16,18,20,22)]
    Ms = [1,2,4,8,16,32,64,128,256,512,1024,2048,4096]
    BM, BMsmall = 128, 8
    Ss = [2,4,8,16,32,64]
    maxB = BYTE_CAP // 4  # largest element count any (K,M) point needs
    X = torch.randn(maxB, dtype=torch.float32, device=dev)
    Pmax = max(Ss)*max(Ms)
    P = torch.empty(Pmax, dtype=torch.float32, device=dev)

    points = []
    for K in Ks:
        for M in Ms:
            if K*M*4 > BYTE_CAP: continue
            points.append((K,M))
    print("points:", len(points))
    results = {"meta": {"date": DATE, "sm_count": sm, "byte_cap": BYTE_CAP,
                        "BM": [BM, BMsmall], "Ss": Ss}, "rows": []}

    def variants(K, M, x, o):
        out = {}
        out["torch"] = lambda: torch.sum(x, dim=0, out=o)
        g1 = (triton.cdiv(M, BM),)
        out["flat128"] = lambda: reduce_flat[g1](x, o, K, M, BM=BM, KU=4, num_warps=4)
        if M > 4:
            g2 = (triton.cdiv(M, BMsmall),)
            out["flat8"] = lambda: reduce_flat[g2](x, o, K, M, BM=BMsmall, KU=8, num_warps=4)
        for S in Ss:
            if K % S: continue
            gm = triton.cdiv(M, BMsmall if M <= 256 else 256)
            BMu = BMsmall if M <= 256 else 256
            def sk(BMu=BMu, S=S, gm=gm):
                split_partial[(gm, S)](x, P, K, M, S, BM=BMu, num_warps=4)
                split_reduce[(triton.cdiv(M, 256),)](P, o, M, S, BM=256, num_warps=4)
            out["splitS%d" % S] = sk
        return out

    for K, M in points:
        x = X[:K*M].view(K, M).contiguous()
        o = torch.empty(M, dtype=torch.float32, device=dev)
        ref = torch.sum(x, dim=0)
        bytes_mv = K*M*4 + M*4
        row = {"K": K, "M": M, "bytes": bytes_mv,
               "derived": {"ctas_flat128": triton.cdiv(M, BM),
                            "ctas_flat8": triton.cdiv(M, BMsmall),
                            "waves_per SM": None}, "variants": {}}
        for name, fn in variants(K, M, x, o).items():
            try:
                r = evt_time(fn, flush=flush)
                if name not in ("torch",):
                    o2 = torch.empty_like(o); fn(); torch.cuda.synchronize()
                    err = float((o2-ref).abs().max()) if M else 0.0
                else:
                    err = 0.0
                r["max_abs_err"] = err
                r["GBps"] = round(bytes_mv/(r["med_ms"]/1e3)/1e9, 1)
                row["variants"][name] = r
            except Exception as ex:
                row["variants"][name] = {"error": str(ex)[:120]}
        results["rows"].append(row)

    # torch.compile comparator at spot shapes (kept last; failures nonfatal)
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 64
    for K, M in points[:: max(1, len(points)//6)]:
        x = X[:K*M].view(K, M).contiguous(); o = torch.empty(M, dtype=torch.float32, device=dev)
        row = next(r for r in results["rows"] if r["K"] == K and r["M"] == M)
        try:
            t0 = time.time()
            cf = torch.compile(lambda xx: torch.sum(xx, dim=0), dynamic=False)
            cf(x); torch.cuda.synchronize()
            r = evt_time(lambda: cf(x), warmup=1, reps=10, flush=flush)
            r["compile_s"] = round(time.time()-t0, 1)
            r["GBps"] = round(row["bytes"]/(r["med_ms"]/1e3)/1e9, 1)
            row["variants"]["torch_compile"] = r
        except Exception as ex:
            row["variants"]["torch_compile"] = {"error": str(ex)[:120]}

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=1)

    # analysis: regime map + collapse check
    print("\n%-7s %-6s %9s  %-28s %s" % ("K","M","GB/s","winner(ms)","runner-up"))
    for row in results["rows"]:
        good = {k: v for k, v in row["variants"].items() if "med_ms" in v}
        if not good: continue
        w = min(good, key=lambda k: good[k]["med_ms"]); s = sorted(good, key=lambda k: good[k]["med_ms"])
        d = row["derived"]
        print("%7d %-6d %9.1f  %-28s %s" % (row["K"], row["M"], good[w]["GBps"],
              "%s %.4f" % (w, good[w]["med_ms"]),
              " ".join("%s %.4f" % (k, good[k]["med_ms"]) for k in s[1:3])))
    # collapse probe: for each K, smallest M where flat128 beats best split
    print("\nBoundary m*(K): min M where flat128 <= min(splitS*); ctas at that M:")
    for row in results["rows"]:
        v = row["variants"]; 
        if not all(k in v and "med_ms" in v[k] for k in ("flat128",)): continue
        splits = {k: v[k]["med_ms"] for k in v if k.startswith("split") and "med_ms" in v[k]}
        if not splits: continue
        bs = min(splits.values())
        if v["flat128"]["med_ms"] <= bs:
            print("K=%8d M=%5d flat wins  ctas_flat=%d waves=%.2f" %
                  (row["K"], row["M"], row["derived"]["ctas_flat128"], row["derived"]["ctas_flat128"]/sm))
    print("wrote", OUT)

if __name__ == "__main__":
    main()
