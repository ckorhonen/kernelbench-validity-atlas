"""e2 debug2: the kernel is clean on fresh tensors (e2_debug). Localize the v3 harness
error: giant-X view? shared P? ref staleness? For suspect configs, run the EXACT v3
sequence and compare three errors:
  err_v3    : |o2 - ref| with ref computed BEFORE the timed loop (v3 as shipped)
  err_fresh : |o2 - ref2|, ref2 recomputed AFTER the loop
  err_rerun : fresh kernel rerun into o3 vs ref2
Plus P-dump diff vs emulation at the same moment. Whichever jumps to ~55/1000s localizes it.
"""
import torch, triton, triton.language as tl, json, os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "e2_debug2.json")
torch.manual_seed(11)

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
                       mask=mmask[None, :], other=0.0)
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

def evt_like(run, flush, reps=12):
    for _ in range(2): run()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    for _ in range(reps):
        flush.zero_()
        s.record(); run(); e.record(); torch.cuda.synchronize()
    return None

def main():
    dev = "cuda:0"
    BM = 32
    flush = torch.zeros(256 * 2**20 // 4, dtype=torch.float32, device=dev)
    # giant X exactly as v3
    X = torch.randn(2**22 * 512, dtype=torch.float32, device=dev)
    P = torch.empty(64 * 512 * 2, dtype=torch.float32, device=dev)
    P.fill_(float("nan"))
    res = []
    # suspect set: v3 row0 (err 55.03 identical S4/S16 KT16, 0.0 KT64) + a big-error point
    for (K, M, S, KT) in [(4096, 1, 4, 16), (4096, 1, 4, 64), (4096, 1, 16, 16),
                          (2**20, 8, 4, 64), (2**20, 64, 64, 64)]:
        x = X[:K * M].view(K, M).contiguous()
        ref = x.sum(dim=0)
        o = torch.empty(M, dtype=torch.float32, device=dev)
        kper = K // S
        gm = triton.cdiv(M, BM)
        def run():
            split_tiled[(gm, S)](x, P, K, M, S, BM=BM, KT=KT, num_warps=4)
            split_reduce[(triton.cdiv(M, 256),)](P, o, M, S, BM=256, num_warps=4)
        evt_like(run, flush)
        o2 = torch.empty_like(o); run(); torch.cuda.synchronize()
        err_v3 = float((o2 - ref).abs().max())
        ref2 = x.sum(dim=0)
        err_fresh = float((o2 - ref2).abs().max())
        o3 = torch.empty_like(o); 
        split_tiled[(gm, S)](x, P, K, M, S, BM=BM, KT=KT, num_warps=4)
        split_reduce[(triton.cdiv(M, 256),)](P, o3, M, S, BM=256, num_warps=4)
        torch.cuda.synchronize()
        err_rerun = float((o3 - ref2).abs().max())
        # P dump vs emulation NOW
        pk = P[:S * M].view(S, M).clone()
        em = torch.zeros(S, M, device=dev)
        for s in range(S):
            acc = torch.zeros(M, device=dev)
            for i in range(kper // KT):
                acc = acc + x[s * kper + i * KT:(s + 1) * kper if False else s * kper + (i + 1) * KT, :].sum(0)
            em[s] = acc
        dP = float((pk - em).abs().max())
        nanP = int(torch.isnan(P[:S * M]).sum().item())
        # is o2 == o3?
        dO = float((o2 - o3).abs().max())
        res.append(dict(K=K, M=M, S=S, KT=KT, err_v3=err_v3, err_fresh=err_fresh,
                        err_rerun=err_rerun, dP_partial=dP, dO_o2_o3=dO, nanP=nanP))
    json.dump(res, open(OUT, "w"), indent=1)
    for r in res:
        print("K=%8d M=%3d S=%2d KT=%2d | v3 %10.4f fresh %8.4f rerun %8.4f "
              "| dP %8.4f dO %8.4f nanP %d"
              % (r["K"], r["M"], r["S"], r["KT"], r["err_v3"], r["err_fresh"],
                 r["err_rerun"], r["dP_partial"], r["dO_o2_o3"], r["nanP"]))
    print("wrote", OUT)

if __name__ == "__main__":
    main()
