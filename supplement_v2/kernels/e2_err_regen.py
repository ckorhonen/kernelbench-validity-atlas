"""err-regen for e2 v3 (W4 v1 debt #1c): correctness-only pass, fixed harness.

v3's recorded max_abs_err was |uninit garbage - ref| (the o2=empty_like bug; e2_debug2).
Timing tables stand; this pass regenerates EVERY (K,M,variant) error with the
e2_debug2-verified pattern: run() closure writes `o`, cuda synchronize BEFORE read.
3 seeds; per-variant list of max_abs_err and err/sqrt(K). W4 v1 consumes this file
for the correctness audit and the tolerance-latitude column (IDENTITY #4).

Run on spark: python3 e2_err_regen.py -> results/e2_v3_err.json. P1: provenance in meta.
"""
import json, os, time, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch, triton  # noqa: E402
from e2_splitk_v3 import split_tiled, split_reduce  # kernels only; main() guarded

OUT = os.path.join(HERE, "results", "e2_v3_err.json")


def main():
    dev = torch.device("cuda:0")
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    Ks = [2 ** k for k in (12, 14, 16, 18, 20, 22)]
    Ms = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    BM, Ss, KTs = 32, [4, 16, 64], [16, 64]        # identical to v3 grids
    SEEDS = [20261017, 20261018, 20261019]
    Xmax = max(Ks) * max(Ms)
    X = torch.empty(Xmax, dtype=torch.float32, device=dev)
    P = torch.empty(max(Ss) * max(Ms) * 2, dtype=torch.float32, device=dev)

    results = {"meta": {"date": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()),
                        "sm_count": sm, "seeds": SEEDS, "BM": BM, "Ss": Ss, "KTs": KTs,
                        "note": "err-only regen; o closure-written + synced (e2_debug2 pattern); "
                                "supersedes e2_splitk_v3.json err columns, timing there unaffected"},
               "rows": []}
    for seed in SEEDS:
        g = torch.Generator(device="cpu").manual_seed(seed)
        X.copy_(torch.randn(Xmax, generator=g).to(dev))
        for K in Ks:
            for M in Ms:
                x = X[: K * M].view(K, M).contiguous()
                o = torch.empty(M, dtype=torch.float32, device=dev)
                ref = torch.sum(x, dim=0)
                row = None
                for rec in results["rows"]:
                    if rec["K"] == K and rec["M"] == M:
                        row = rec
                if row is None:
                    row = {"K": K, "M": M, "variants": {}}
                    results["rows"].append(row)
                torch.cuda.synchronize()
                (o - ref).abs().max()  # warm the path; torch ref vs itself
                row["variants"].setdefault("torch", {"max_abs_err": []})["max_abs_err"].append(0.0)
                for S in Ss:
                    kper = K // S
                    for KT in KTs:
                        if KT > kper:
                            continue
                        name = "tsplit_S%d_KT%d" % (S, KT)
                        gm = triton.cdiv(M, BM)

                        def run(S=S, KT=KT, gm=gm):
                            split_tiled[(gm, S)](x, P, K, M, S, BM=BM, KT=KT, num_warps=4)
                            split_reduce[(triton.cdiv(M, 256),)](P, o, M, S, BM=256, num_warps=4)

                        run()
                        torch.cuda.synchronize()
                        e = float((o - ref).abs().max())
                        row["variants"].setdefault(name, {"max_abs_err": []})["max_abs_err"].append(e)

    for row in results["rows"]:  # fold in err/sqrt(K) + med
        for nm, v in row["variants"].items():
            es = sorted(v["max_abs_err"])
            v["med_abs_err"] = es[len(es) // 2]
            v["max_abs_err_seedmax"] = es[-1]
            v["rel_sqrtK_med"] = v["med_abs_err"] / row["K"] ** 0.5

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)

    n = sum(len(v["max_abs_err"]) for r in results["rows"] for v in r["variants"].values())
    print("err regen done: %d rows, %d variant-seed measurements -> %s" %
          (len(results["rows"]), n, OUT))
    for nm in sorted({k for r in results["rows"] for k in r["variants"]}):
        vals, rels = [], []
        for r in results["rows"]:
            v = r["variants"].get(nm)
            if v:
                vals.append(v["med_abs_err"]); rels.append(v["rel_sqrtK_med"])
        vals.sort(); rels.sort()
        print(" %-18s med %9.4f max %10.4f  med_rel/sqrtK %.2e" %
              (nm, vals[len(vals) // 2], max(vals), rels[len(rels) // 2]))


if __name__ == "__main__":
    main()
