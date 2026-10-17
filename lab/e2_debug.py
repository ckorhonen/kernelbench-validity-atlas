"""e2 debug: localize the O(sqrt(K)) error in e2 v3 tiled-split (found by w4 audit).

Stages, each diffed vs torch on the same seeded data:
  A. kernel partials (split_tiled alone, P read back) vs torch per-(s,chunk,col) sums
  B. kernel final (both stages) vs torch sum
  C. pure-torch emulation of the same accumulation tree vs torch sum
Grid: K in 2^20, M in {1,2,8,64}; S in {4,64}; KT=64 (+KT16 at one point).
If A is clean and B is not -> reduce stage; if A dirty -> tiled stage; column pattern
of the A-diff localizes mask/index bugs.
"""
import torch, triton, triton.language as tl, json, os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "e2_debug.json")
torch.manual_seed(7)

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

def main():
    dev = "cuda:0"
    K = 2 ** 20
    BM = 32
    out = []
    for M in (1, 2, 8, 64):
        x = torch.randn(K, M, device=dev)
        ref = x.sum(dim=0)
        # emulation: per-split chunk, KT-tile tree then sequential across tiles
        for S in (4, 64):
            for KT in (64, 16):
                kper = K // S
                P = torch.full((S * M,), float("nan"), device=dev, dtype=torch.float32)
                split_tiled[(triton.cdiv(M, BM), S)](x, P, K, M, S, BM=BM, KT=KT, num_warps=4)
                torch.cuda.synchronize()
                # A: partials vs faithful emulation (sequential tile adds, torch fp32)
                pt = P.view(S, M)
                em2 = torch.zeros(S, M, device=dev)
                for s in range(S):
                    acc = torch.zeros(M, device=dev)
                    for i in range(kper // KT):
                        acc = acc + x[s * kper + i * KT:s * kper + (i + 1) * KT, :].sum(0)
                    em2[s] = acc
                dA = (pt - em2).abs().max().item()
                # per-column A diff pattern
                colA = (pt - em2).abs().max(0).values
                # B: full kernel
                o = torch.empty(M, device=dev)
                split_reduce[(triton.cdiv(M, 256),)](P, o, M, S, BM=256, num_warps=4)
                torch.cuda.synchronize()
                dB = (o - ref).abs().max().item()
                # C: emulation final = sum of em2 partials (sequential)
                fin = torch.zeros(M, device=dev)
                for s in range(S):
                    fin = fin + em2[s]
                dC = (fin - ref).abs().max().item()
                # nan check
                nanP = int(torch.isnan(P).sum().item())
                nanO = int(torch.isnan(o).sum().item())
                out.append(dict(K=K, M=M, S=S, KT=KT, dA_partial=dA, dB_final=dB,
                                dC_emu=dC,
                                colA_max=float(colA.max()), colA_argmax=int(colA.argmax()),
                                colA_head=[float(v) for v in colA[:4]],
                                nanP=nanP, nanO=nanO,
                                ref_scale=float(ref.abs().mean())))
    json.dump(out, open(OUT, "w"), indent=1)
    for r in out:
        print("K=2^20 M=%3d S=%2d KT=%2d | dA(part) %10.4f dB(fin) %10.4f dC(emu) %10.4f "
              "| colA max@%2d head %s | nanP %d nanO %d | |ref|~%.1f"
              % (r["M"], r["S"], r["KT"], r["dA_partial"], r["dB_final"], r["dC_emu"],
                 r["colA_argmax"], ["%.2f" % v for v in r["colA_head"]],
                 r["nanP"], r["nanO"], r["ref_scale"]))
    print("wrote", OUT)

if __name__ == "__main__":
    main()
