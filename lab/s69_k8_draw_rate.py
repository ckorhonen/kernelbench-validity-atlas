"""s69 — K=8 repeated-draw admission-rate probe. PREREG FROZEN HERE, before the run.

Mirror of lab/s69_prereg.md (registered 2026-10-31 09:55Z). Purpose: convert s59's paired discordance
(26/228 labels flip between two input draws, median 5.9x budget-ratio swing) into an admission
frequency over draws.

POPULATION (fixed): the 11 reduction shapes that flipped in s59:
  (98304,48),(98304,192),(98304,384),(393216,20),(393216,48),(393216,192),(393216,384),
  (1572864,20),(1572864,48),(1572864,192),(1572864,384)
Arms: the shipped s56 pool — torch.sum, torch.compile, and the six tsplit_{S,KT} arms of RED_SKT
(S in 4,16,64 x KT in 16,64). NOTE the prereg said "12-arm pool ... five tsplit pairs"; the shipped
pool is 8 arms (1+1+6) and the coded run uses it. Recorded as a prereg slip; the check count is
11 x 32 x 8 = 2,816, not the prereg's 4,224.
Gate: KernelBench evaluator tolerance atol=rtol=1e-4 fp32, candidate-scaled, against the default
reference realization (torch.sum(x, dim=0)) of the SAME draw, exactly as budget_ratio in
s56_heldout_contract does. Pass label = budget_ratio <= 1.

DESIGN: N=32 independent draws per shape, seed = (K*1000003 + M*7919 + draw*2654435761) % (2**31-1),
recorded per draw. Pass labels only; no timing. Reference's own rho vs fp64 recorded per draw.

OUTPUT: results/s69_k8_draw_rate.jsonl, one row per (shape, draw) with per-arm labels and ratios,
streamed so a timebox kill keeps partial data. Hard cap 20 min wall inside the script (partial save).
Run on spark: python3 s69_k8_draw_rate.py
"""
import json, os, sys, time
import torch
import triton

HERE = os.path.dirname(os.path.abspath(__file__))


def _kb_res_dir(here):
    for cand in (os.path.join(here, "results"), os.path.join(os.path.dirname(here), "results")):
        if os.path.isdir(cand):
            return cand
    return os.path.join(here, "results")


RES = _kb_res_dir(HERE)
OUT = os.path.join(RES, "s69_k8_draw_rate.jsonl")
ATOL = RTOL = 1e-4
BM = 32
NDRAW = 32
CAP_S = 20 * 60

sys.path.insert(0, HERE)
from e2_splitk_v3 import split_tiled, split_reduce      # noqa: E402
from m3_autotune_curve import RED_SKT                    # noqa: E402

SHAPES = [(98304, 48), (98304, 192), (98304, 384), (393216, 20), (393216, 48), (393216, 192),
          (393216, 384), (1572864, 20), (1572864, 48), (1572864, 192), (1572864, 384)]

dev = torch.device("cuda:0")


def ratio(cand, ref):
    d = (cand.double() - ref.double()).abs()
    denom = ATOL + RTOL * cand.double().abs()
    r = float((d / denom).max())
    return r, bool(r <= 1.0)


def seed_for(K, M, draw):
    return (K * 1000003 + M * 7919 + draw * 2654435761) % (2 ** 31 - 1)


def main():
    t0 = time.time()
    hdr = {"artifact": "s69_k8_draw_rate", "date": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()),
           "prereg": "lab/s69_prereg.md frozen 09:55Z before run; docstring mirrors it",
           "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
           "contract": {"atol": ATOL, "rtol": RTOL, "direction": "evaluator-exact, candidate-scaled",
                        "reference": "torch.sum(x, dim=0) on the same draw"},
           "arms": ["torch", "compile"] + ["tsplit_" + n for n, _, _ in RED_SKT],
           "n_shapes": len(SHAPES), "n_draws": NDRAW,
           "prereg_slip": "prereg said 12 arms / 4,224 checks; shipped pool is 8 arms => 2,816 checks",
           "cap_s": CAP_S}
    rows = []
    cfgs = {}
    for (K, M) in SHAPES:
        gm = triton.cdiv(M, BM)
        cf = torch.compile(lambda x, o: torch.sum(x, dim=0, out=o), dynamic=False)
        cfgs[(K, M)] = (gm, cf)
        print("compiled", (K, M), flush=True)

    killed = False
    for (K, M) in SHAPES:
        gm, cf = cfgs[(K, M)]
        for draw in range(NDRAW):
            if time.time() - t0 > CAP_S:
                killed = True
                break
            seed = seed_for(K, M, draw)
            torch.manual_seed(seed)
            x = torch.randn(K * M, dtype=torch.float32, device=dev).view(K, M).contiguous()
            ref = torch.sum(x, dim=0)
            ref64 = torch.sum(x.double(), dim=0)
            rho_ref, ref_pass = ratio(ref, ref64)
            rec = {"K": K, "M": M, "draw": draw, "seed": seed,
                   "ref_rho_fp64": rho_ref, "ref_pass_vs_fp64": ref_pass, "arms": {}}
            o = torch.empty(M, dtype=torch.float32, device=dev)
            # torch
            torch.sum(x, dim=0, out=o)
            torch.cuda.synchronize()
            r, p = ratio(o, ref)
            rec["arms"]["torch"] = {"ratio": r, "pass": p}
            # compile
            try:
                cf(x, o)
                torch.cuda.synchronize()
                r, p = ratio(o, ref)
                rec["arms"]["compile"] = {"ratio": r, "pass": p}
            except Exception as e:
                rec["arms"]["compile"] = {"error": str(e)[:160]}
            # tsplit arms
            P = torch.empty(64 * M * 2, dtype=torch.float32, device=dev)
            for name, S, KT in RED_SKT:
                if K // S < KT:
                    continue
                try:
                    split_tiled[(gm, S)](x, P, K, M, S, BM=BM, KT=KT, num_warps=4)
                    split_reduce[(triton.cdiv(M, 256),)](P, o, M, S, BM=256, num_warps=4)
                    torch.cuda.synchronize()
                    r, p = ratio(o, ref)
                    rec["arms"]["tsplit_" + name] = {"ratio": r, "pass": p}
                except Exception as e:
                    rec["arms"]["tsplit_" + name] = {"error": str(e)[:160]}
            rows.append(rec)
            del x, ref, ref64, o, P
        if killed:
            print("CAP FIRED — partial results saved", flush=True)
            break

    with open(OUT, "w") as f:
        f.write(json.dumps({"header": hdr}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("wrote", OUT, "rows", len(rows), "killed", killed, "wall_s", round(time.time() - t0, 1))
    # quick console summary
    import collections
    agg = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        for a, v in r["arms"].items():
            if v.get("pass"):
                agg[(r["K"], r["M"])][a] += 1
    for k in SHAPES:
        print(k, dict(agg[k]))


if __name__ == "__main__":
    main()
