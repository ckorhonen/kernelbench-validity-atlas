"""E2 v3: the missing corner -- tiled-split. v2 finding: tile grid keys on M only, so
M<=128 degenerates to 1 CTA; v2 split arm is 1D-loop (32B/CTA/iter) at small M.
v3: split_partial with 2D tile loads (KT x BM per iter) -- grid (M-blocks, S), both axes
shapeable. This completes the fair 2D design space (grid_m x grid_k x access shape).
Question: does ANY hand variant beat torch beyond noise at M<=128, K>=2^14? If no ->
W2 no-collapse verdict + P5 falsifier; if yes -> boundary mapping in (ctas_m, ctas_k).
"""
import json, os, time
import torch, triton, triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "e2_splitk_v3.json")
DATE = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
FLUSH_BYTES = 256 * 2**20

@triton.jit
def split_tiled(X, P, K, M, S, BM: tl.constexpr, KT: tl.constexpr):
    pid_m = tl.program_id(0); pid_s = tl.program_id(1)
    k_per = K // S
    k0 = pid_s * k_per
    offs_m = pid_m * BM + tl.arange(0, BM)
    mmask = offs_m < M
    acc = tl.zeros([BM], dtype=tl.float32)
    for ki in range(k0, k0 + k_per, KT):
        offs_k = ki + tl.arange(0, KT)
        tile = tl.load(X + offs_k[:, None] * M + offs_m[None, :],
                       mask=mmask[None, :], other=0.0)  # k_per,KT pow2; KT<=k_per enforced host-side
        acc += tl.sum(tile, axis=0)
    tl.store(P + pid_s * M + offs_m, acc, mask=mmask)

@triton.jit
def split_reduce(P, O, M, S, BM: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BM + tl.arange(0, BM)
    acc = tl.zeros([BM], dtype=tl.float32)
    for s in range(S):
        acc += tl.load(P + s * M + offs, mask=offs < M, other=0.0)
    tl.store(O + offs, acc, mask=offs < M)

def evt_time(fn, warmup=2, reps=12, flush=None):
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
    Ms = [1,2,4,8,16,32,64,128,256,512]
    BM = 32                      # small-M focus: BM=32 -> more M-CTAs
    Ss = [4,16,64]
    KTs = [16,64]
    Xmax = max(Ks)*max(Ms)
    X = torch.randn(Xmax, dtype=torch.float32, device=dev)
    P = torch.empty(max(Ss)*max(Ms)*2, dtype=torch.float32, device=dev)

    results = {"meta": {"date": DATE, "sm_count": sm, "BM": BM, "Ss": Ss, "KTs": KTs,
                        "note": "v3 tiled-split: grid (Mblocks,S), 2D tile loads; small-M focus"}, "rows": []}
    for K in Ks:
        for M in Ms:
            x = X[:K*M].view(K, M).contiguous()
            o = torch.empty(M, dtype=torch.float32, device=dev)
            ref = torch.sum(x, dim=0)
            row = {"K": K, "M": M, "bytes": K*M*4 + M*4, "variants": {}}
            row["variants"]["torch"] = evt_time(lambda: torch.sum(x, dim=0, out=o), flush=flush)
            row["variants"]["torch"]["max_abs_err"] = 0.0
            for S in Ss:
                kper = K // S
                for KT in KTs:
                    if KT > kper: continue
                    name = "tsplit_S%d_KT%d" % (S, KT)
                    gm = triton.cdiv(M, BM)
                    def run(S=S, KT=KT, gm=gm):
                        split_tiled[(gm, S)](x, P, K, M, S, BM=BM, KT=KT, num_warps=4)
                        split_reduce[(triton.cdiv(M, 256),)](P, o, M, S, BM=256, num_warps=4)
                    try:
                        r = evt_time(run, flush=flush)
                        o2 = torch.empty_like(o); run(); torch.cuda.synchronize()
                        r["max_abs_err"] = float((o2-ref).abs().max())
                        r["ctas"] = gm*S
                        row["variants"][name] = r
                    except Exception as ex:
                        row["variants"][name] = {"error": str(ex)[:120]}
            results["rows"].append(row)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=1)

    print("%-8s %-5s %10s  %s" % ("K", "M", "torch us", "best hand (ratio, ctas, noise-beat?)"))
    wins = []
    for r in results["rows"]:
        v = r["variants"]
        t = v["torch"]["med_ms"]; ti = v["torch"]["iqr_ms"]
        hand = {k: x for k, x in v.items() if k != "torch" and "med_ms" in x}
        if not hand: continue
        bk = min(hand, key=lambda k: hand[k]["med_ms"]); b = hand[bk]
        margin = (t - b["med_ms"]) / t
        noise = max(((ti[1]-ti[0]) + (b["iqr_ms"][1]-b["iqr_ms"][0]))/2/t, 0.01)
        tag = "BEAT" if margin > noise else ("tie" if margin > -noise else "lose")
        if tag == "BEAT": wins.append((r["K"], r["M"], bk, margin, b["ctas"]))
        print("%8d %-5d %10.2f  %-16s %.2f ctas=%5d %s" %
              (r["K"], r["M"], t*1e3, bk, t/b["med_ms"], b["ctas"], tag))
    print("\nhand BEATS torch beyond noise: %d points" % len(wins))
    for w in wins: print("  ", w)
    print("wrote", OUT)

if __name__ == "__main__":
    main()
