"""E2 v2: contested boundary -- split-K vs flat on small-out reductions (sum over K).
v1 lesson: 1D-column flat loop is latency-bound (512B/CTA/iter) -- not a fair flat arm.
v2 flat arm: 2D tile (KT x BM), tl.sum axis=0; KT in {16,32,64,128}; BM=128.
  Per-iter bytes/CTA = KT*BM*4 (2..64KB) -- bandwidth-shaped, comparable to split-K partials.
Claim under test: 'split-K wins when the output is too small to fill the machine'.
Protocol: monotonicity check BEFORE bisection; boundary m*(K); collapse probe in
  programs-per-SM coords. Split variants carried from v1 for the comparison.
Numerics: fp32; split-K/tiled reorders sums -- max|diff| vs torch recorded per point.
"""
import json, os, time
import torch, triton, triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "e2_splitk_v2.json")
DATE = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
BYTE_CAP = 128 * 2**20
FLUSH_BYTES = 256 * 2**20

@triton.jit
def reduce_tiled(X, O, K, M, BM: tl.constexpr, KT: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BM + tl.arange(0, BM)
    mmask = offs_m < M
    acc = tl.zeros([BM], dtype=tl.float32)
    for k0 in range(0, K, KT):
        offs_k = k0 + tl.arange(0, KT)
        kmask = offs_k < K
        ptr = X + offs_k[:, None] * M + offs_m[None, :]
        tile = tl.load(ptr, mask=kmask[:, None] & mmask[None, :], other=0.0)
        acc += tl.sum(tile, axis=0)
    tl.store(O + offs_m, acc, mask=mmask)

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
    BM = 128
    KTs = [16,32,64,128]
    Ss = [2,4,8,16,32,64]
    maxB = BYTE_CAP // 4
    X = torch.randn(maxB, dtype=torch.float32, device=dev)
    Pmax = max(Ss)*max(Ms)
    P = torch.empty(Pmax, dtype=torch.float32, device=dev)

    points = []
    for K in Ks:
        for M in Ms:
            if K*M*4 > BYTE_CAP: continue
            points.append((K,M))
    print("points:", len(points))
    results = {"meta": {"date": DATE, "sm_count": sm, "byte_cap": BYTE_CAP, "BM": BM,
                        "KTs": KTs, "Ss": Ss,
                        "note": "v2 flat = 2D tile KT x BM, tl.sum axis=0"}, "rows": []}

    def variants(K, M, x, o):
        out = {}
        out["torch"] = lambda: torch.sum(x, dim=0, out=o)
        for KT in KTs:
            if KT > max(K, 1)//1: pass
            g = (triton.cdiv(M, BM),)
            out["tile%d" % KT] = lambda KT=KT: reduce_tiled[g](x, o, K, M, BM=BM, KT=KT, num_warps=4)
        for S in Ss:
            if K % S: continue
            BMu = 8 if M <= 256 else 256
            gm = triton.cdiv(M, BMu)
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
               "derived": {"ctas_tile": triton.cdiv(M, BM)}, "variants": {}}
        for name, fn in variants(K, M, x, o).items():
            try:
                r = evt_time(fn, flush=flush)
                if name != "torch":
                    o2 = torch.empty_like(o); fn(); torch.cuda.synchronize()
                    err = float((o2-ref).abs().max())
                else:
                    err = 0.0
                r["max_abs_err"] = err
                r["GBps"] = round(bytes_mv/(r["med_ms"]/1e3)/1e9, 1)
                row["variants"][name] = r
            except Exception as ex:
                row["variants"][name] = {"error": str(ex)[:120]}
        results["rows"].append(row)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=1)

    print("\n%-7s %-6s %9s  %-24s %s" % ("K","M","GB/s","winner(ms)","runner-up"))
    for row in results["rows"]:
        good = {k: v for k, v in row["variants"].items() if "med_ms" in v}
        if not good: continue
        s = sorted(good, key=lambda k: good[k]["med_ms"])
        w = s[0]
        print("%7d %-6d %9.1f  %-24s %s" % (row["K"], row["M"], good[w]["GBps"],
              "%s %.4f" % (w, good[w]["med_ms"]),
              " ".join("%s %.4f" % (k, good[k]["med_ms"]) for k in s[1:3])))
    # monotonicity + boundary: per K, min M where best tile <= best split
    print("\nBoundary m*(K): min M where best tile <= best split (ctas, waves/SM, bestKT):")
    for row in results["rows"]:
        v = row["variants"]
        tiles = {k: v[k]["med_ms"] for k in v if k.startswith("tile") and "med_ms" in v[k]}
        splits = {k: v[k]["med_ms"] for k in v if k.startswith("split") and "med_ms" in v[k]}
        if not tiles or not splits: continue
        bt, bs = min(tiles.values()), min(splits.values())
        if bt <= bs:
            bkt = min(tiles, key=tiles.get)
            ctas = row["derived"]["ctas_tile"]
            print("K=%8d M=%5d tile %.4f split %.4f ctas=%d waves=%.2f %s" %
                  (row["K"], row["M"], bt, bs, ctas, ctas/sm, bkt))
    print("wrote", OUT)

if __name__ == "__main__":
    main()
