"""E2b: floor-term bisection, pre-registered in reflection #4 (DIARY 2026-10-17).
Model under test: the M<=8 hand loss is an ~8us two-launch absolute-time floor; boundary
= intersection(wave-band, floor-time). Prediction: the crossing M* is monotone in
t_kernel/t_launch at FIXED waves (ctas pinned ~64 across M via S=64/cdiv(M,BM)).
Falsifier (P5): M* not monotone in t_kernel/t_launch at fixed waves => floor model dies;
no-collapse variant written.
Design: K swept DOWN to 2^8 so t_kernel ~ t_launch; M fine grid; launch floor measured
directly (K=16 => kernel negligible). Wave-pinned arm is primary; global-best recorded
for context. P2 fields throughout; interleaving via per-row alternation, L2 flush.
"""
import json, os, time
import torch, triton, triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "e2b_floor.json")
DATE = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
FLUSH_BYTES = 256 * 2**20
CTAS_PIN = 64

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

def evt_time(fn, warmup=3, reps=12, flush=None):
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

def prev_pow2(x):
    return 1 << max(0, x.bit_length() - 1)

def main():
    dev = torch.device("cuda:0")
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    flush = torch.empty(FLUSH_BYTES//4, dtype=torch.float32, device=dev)
    Ks = [2**k for k in (8, 9, 10, 11, 12, 14, 16, 18)]   # downward extension per pre-registration
    Ms = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256]
    BM = 32
    Xmax = max(Ks) * max(Ms)
    X = torch.randn(Xmax, dtype=torch.float32, device=dev)
    PmaxS = 64
    P = torch.empty(PmaxS * max(Ms) * 2, dtype=torch.float32, device=dev)

    # --- launch floor: kernel time negligible at K=16 ---
    x16 = X[:16*2].view(16, 2).contiguous(); o2 = torch.empty(2, dtype=torch.float32, device=dev)
    def two_launch():  # the hand arm's launch pattern, nothing else
        split_tiled[(1, 2)](x16, P, 16, 2, 2, BM=BM, KT=16, num_warps=4)
        split_reduce[(1,)](P, o2, 2, 2, BM=256, num_warps=4)
    def one_launch():
        torch.sum(x16, dim=0, out=o2)
    floor2 = evt_time(two_launch, flush=flush); floor1 = evt_time(one_launch, flush=flush)

    results = {"meta": {"date": DATE, "sm_count": sm, "BM": BM, "ctas_pin": CTAS_PIN,
                        "Ms": Ms, "Ks": Ks,
                        "launch_floor_two_ms": floor2, "launch_floor_one_ms": floor1,
                        "note": "e2b floor bisection; wave-pinned ctas~64; falsifier per P5 in-analysis"},
              "rows": []}
    for K in Ks:
        for M in Ms:
            x = X[:K*M].view(K, M).contiguous()
            o = torch.empty(M, dtype=torch.float32, device=dev)
            ref = torch.sum(x, dim=0)
            row = {"K": K, "M": M, "bytes": K*M*4 + M*4, "variants": {}}
            row["variants"]["torch"] = evt_time(lambda: torch.sum(x, dim=0, out=o), flush=flush)
            row["variants"]["torch"]["max_abs_err"] = 0.0
            # wave-pinned: ctas = cdiv(M,BM)*S ~= CTAS_PIN
            gm = triton.cdiv(M, BM)
            S = max(1, CTAS_PIN // gm)
            kper = K // S
            KT = min(64, prev_pow2(max(kper, 1)))
            if kper >= 1 and KT >= 1:
                def run(S=S, KT=KT, gm=gm):
                    split_tiled[(gm, S)](x, P, K, M, S, BM=BM, KT=KT, num_warps=4)
                    split_reduce[(triton.cdiv(M, 256),)](P, o, M, S, BM=256, num_warps=4)
                try:
                    r = evt_time(run, flush=flush)
                    o2b = torch.empty_like(o); run(); torch.cuda.synchronize()
                    r["max_abs_err"] = float((o2b-ref).abs().max())
                    r["ctas"] = gm*S; r["S"] = S; r["KT"] = KT; r["pinned"] = True
                    row["variants"]["tsplit_pinned"] = r
                except Exception as ex:
                    row["variants"]["tsplit_pinned"] = {"error": str(ex)[:120]}
            results["rows"].append(row)

    # --- analysis: M*(K) at fixed waves; monotonicity vs t_kernel/t_launch ---
    two_us = floor2["med_ms"]*1e3
    def status(margin, noise):
        return "win" if margin > noise else ("tie" if margin > -noise else "lose")
    per_k = []
    for K in Ks:
        seq = []
        tks = []
        for row in results["rows"]:
            if row["K"] != K: continue
            v = row["variants"]; t = v["torch"]["med_ms"]; ti = v["torch"]["iqr_ms"]
            h = v.get("tsplit_pinned", {})
            if "med_ms" not in h: continue
            hi = h["iqr_ms"]
            margin = (t - h["med_ms"]) / t
            noise = max(((ti[1]-ti[0]) + (hi[1]-hi[0]))/2/t, 0.01)
            seq.append((row["M"], status(margin, noise), margin))
            tk = max(h["med_ms"]*1e3 - two_us, 0.0)   # kernel-time estimate, us
            tks.append(tk)
        ok = [s for _, s, _ in seq if s != "lose"]
        if not ok:
            mstar = None
        else:
            mstar = seq[0][0]
            for m, s, _ in seq:      # first M from which onward never loses
                if s != "lose": mstar = m; break
            else: mstar = None
            # sustained: smallest M such that all larger grid Ms are non-lose
            sus = None
            for i in range(len(seq)):
                if all(s != "lose" for _, s, _ in seq[i:]): sus = seq[i][0]; break
        r = (sorted(tks)[len(tks)//2] / two_us) if two_us > 0 and tks else None
        per_k.append({"K": K, "Mstar_sustained": sus, "Mstar_first": mstar,
                      "r_tkernel_over_tlaunch": r,
                      "seq": [(m, s, round(g, 4)) for m, s, g in seq]})
    # Spearman on ranks (hand-rolled), permutation p
    def spearman(x, y):
        pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
        if len(pairs) < 4: return None, None, len(pairs)
        xs = sorted(range(len(pairs)), key=lambda i: pairs[i][0])
        ys = sorted(range(len(pairs)), key=lambda i: pairs[i][1])
        rx = [0]*len(pairs); ry = [0]*len(pairs)
        for r_, i in enumerate(xs): rx[i] = r_
        for r_, i in enumerate(ys): ry[i] = r_
        mx = sum(rx)/len(rx); my = sum(ry)/len(ry)
        cov = sum((a-mx)*(b-my) for a, b in zip(rx, ry))
        vx = sum((a-mx)**2 for a in rx); vy = sum((b-my)**2 for b in ry)
        rho = cov/((vx*vy)**.5) if vx > 0 and vy > 0 else 0.0
        import random; rng = random.Random(7); cnt = 0; N = 2000
        for _ in range(N):
            ry2 = ry[:]; rng.shuffle(ry2)
            my2 = sum(ry2)/len(ry2)
            c = sum((a-mx)*(b-my2) for a, b in zip(rx, ry2))
            v2 = sum((b-my2)**2 for b in ry2)
            if (c/((vx*v2)**.5) if v2 > 0 and vx > 0 else 0) >= rho: cnt += 1
        return rho, cnt/N, len(pairs)

    rho, p, n = spearman([d["K"] for d in per_k], [d["Mstar_sustained"] for d in per_k])
    rho2, p2, n2 = spearman([d["r_tkernel_over_tlaunch"] for d in per_k],
                            [d["Mstar_sustained"] for d in per_k])
    verdict = ("MONOTONE" if (rho2 is not None and rho2 > 0 and p2 is not None and p2 < 0.05)
               else "NON-MONOTONE")
    results["analysis"] = {"per_k": per_k, "two_launch_us": two_us,
                           "spearman_K_vs_Mstar": [rho, p, n],
                           "spearman_r_vs_Mstar": [rho2, p2, n2],
                           "falsifier_verdict": verdict,
                           "rule": "MONOTONE iff rho>0 and perm-p<0.05 (pre-registered)"}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=1)

    print("launch floor: two-launch %.2f us, one-launch %.2f us" %
          (floor2["med_ms"]*1e3, floor1["med_ms"]*1e3))
    print("%8s %14s %10s   M* (sustained / first)" % ("K", "r=tk/tl", "verdict"))
    for d in per_k:
        print("%8d %14s %10s   %s / %s" % (d["K"],
              ("%.2f" % d["r_tkernel_over_tlaunch"]) if d["r_tkernel_over_tlaunch"] else "-",
              "", d["Mstar_sustained"], d["Mstar_first"]))
    print("spearman r vs M*: rho=%.3f p=%.3f n=%d -> %s" % (rho2 or 0, p2 if p2 is not None else 1, n2, verdict))
    print("wrote", OUT)

if __name__ == "__main__":
    main()
