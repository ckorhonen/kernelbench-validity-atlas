#!/usr/bin/env python
"""s104b_topology_draws.py — declared post-hoc instrument repair + replication leg for s104 (declared
BEFORE any row of this run; s104's frozen bar and verdict are NOT changed, they are re-reported alongside).

WHY (two defects in s104, both stated plainly):
1. K3 was mis-instrumented: s104 profiled `arms[name].clone()`, so the top CUDA event was a Memcpy DtoD
   and every arm classified OTHER -> s104's frozen verdict came out INSTRUMENT_FAILED. The repair profiles
   the actual arm computation (rebuild inside the profiler), which is what s101b did.
2. s104's topology spread (1.13-1.23x on c) is the same order as the draw-to-draw CV of the amplitude that
   s96 measured (9%), and s104 used one draw per g. A 20% arm difference on one draw is not evidence.
   This leg repeats 8 draws at g=1 and g=64 and reports the PAIRED arm differences.

DESIGN (frozen before any row): tuple T1 as s104; g in {1.0, 64.0}; draws seeds 1085..1092 (declared);
arms C_seq/C_shuf/C_bal/I_seq as s104 (same declared permutation seed 20261114). Per draw: A, X (fp64),
S = sum of the 16 partials in fp64 (the exact chunked decomposition), each arm P, and the fp64-anchor
self-check. Metrics: rho_vs_ref, rho_vs_fp64, rho_vs_S, rho_S_vs_ref, rho_S_vs_fp64, c, and the
partial-magnitude statistics (max |p_i|, mean |p_i|) so the magnitude hypothesis has a measured covariate.
K1 A repeat bit-identical; K3 engine class of every arm recomputed under the profiler (must be
TF32_TENSOROP at both g, else CLASS_NOT_HELD); K4 C_seq rebuild bit-identical; K5 X repeat bit-identical.

Predeclared reading (frozen): MEMBERSHIP_EFFECT iff I_seq's paired c is below C_seq on >= 7 of 8 draws
AND the median paired ratio <= 0.90; MERGE_SHAPE_EFFECT iff the same holds for C_bal or C_shuf vs C_seq;
DRAW_NOISE iff neither holds, in which case s104's 1.2253 spread is not distinguishable from draw
variation and the topology clause of the depth-ratio conjecture is UNRESOLVED (not falsified).
The decomposition rho_vs_S vs rho_S_vs_ref reports how much of c is merge rounding and how much is the
chunk decomposition itself; no threshold on it, descriptive only.
"""
import json, os, re, signal, time, hashlib
import numpy as np
import torch

LAB = os.path.expanduser("~/kb-lab")
OUT = os.path.join(LAB, "results", "s104b_topology_draws.jsonl")
SUMOUT = os.path.join(LAB, "results", "s104b_summary.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

CFG_ALARM_S = 300
GLOBAL_DEADLINE_S = 1200
CHUNK_ELEMS = 1 << 22
STRICT, REF_REL, NOISE_FLOOR = 1e-4, 2.2e-4, 0.05

ARGS = dict(n=2, cin=64, h=151, w=151, cout=64, k=3, s=1, p=1)
GRID = [1.0, 64.0]
SEEDS = list(range(1085, 1093))
DEPTH = 16
PERM_SEED = 20261114

CLASS_PATTERNS = [
    ("WINOGRAD", re.compile(r"winograd", re.I)),
    ("TF32_TENSOROP", re.compile(r"tf32|tensorop|1688|16816|h1688", re.I)),
    ("FP64", re.compile(r"dgrad|dgemm|dconv|fp64|double|_d_", re.I)),
    ("FP32_SIMT", re.compile(r"fprop|implicit_gemm|sgemm|s1688|fp32|scudnn|simt", re.I)),
    ("CUTLASS", re.compile(r"cutlass", re.I)),
]


class Timeout(Exception):
    pass


def _alarm(s, f):
    raise Timeout("config exceeded %ds" % CFG_ALARM_S)


signal.signal(signal.SIGALRM, _alarm)


def classify(names):
    for lab, pat in CLASS_PATTERNS:
        for n in names:
            if pat.search(n):
                return lab, n
    return ("OTHER", names[0] if names else None)


def profile_call(fn):
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    evts = [(e.key, float(e.self_device_time_total)) for e in prof.key_averages()
            if e.self_device_time_total > 0]
    evts.sort(key=lambda t: -t[1])
    return evts[:4]


def conv_fwd(x, w):
    return torch.nn.functional.conv2d(x, w, stride=ARGS["s"], padding=ARGS["p"])


def channel_groups(kind, k):
    C = ARGS["cin"]
    step = C // k
    if kind == "contiguous":
        return [list(range(i * step, (i + 1) * step)) for i in range(k)]
    if kind == "interleaved":
        return [[i + j * k for j in range(step)] for i in range(k)]
    raise ValueError(kind)


def partial(x, w, idx):
    xc = torch.zeros_like(x)
    wc = torch.zeros_like(w)
    sel = torch.tensor(idx, device=x.device)
    xc[:, sel] = x[:, sel]
    wc[:, sel] = w[:, sel]
    p = conv_fwd(xc, wc)
    del xc, wc
    return p


def build_parts(x, w, kind):
    groups = channel_groups(kind, DEPTH)
    return [partial(x, w, g) for g in groups]


def merge_seq(parts, order):
    out = parts[order[0]]
    for i in order[1:]:
        out = out + parts[i]
    return out


def merge_bal(parts):
    level = list(parts)
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level) - 1, 2):
            nxt.append(level[i] + level[i + 1])
        if len(level) % 2 == 1:
            nxt.append(level[-1])
        level = nxt
    return level[0]


def chunked_compare(ref, cand, atol, rtol):
    n_fail, max_ratio, n_tot = 0, 0.0, 0
    rf, cf = ref.detach().reshape(-1), cand.detach().reshape(-1)
    n, i = rf.numel(), 0
    while i < n:
        j = min(n, i + CHUNK_ELEMS)
        d = (rf[i:j] - cf[i:j]).abs()
        b = atol + rtol * cf[i:j].abs()
        r = d / b
        n_fail += int((r > 1).sum().item())
        mr = float(torch.max(r, 0)[0].item())
        max_ratio = max(max_ratio, mr)
        n_tot += (j - i)
        del d, b, r
        i = j
    return n_fail, max_ratio, n_tot


def run_point(g, seed):
    rec = dict(g=g, seed=seed, status="ok")
    t0 = time.time()
    signal.alarm(CFG_ALARM_S)
    try:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(seed)
        x = torch.randn(ARGS["n"], ARGS["cin"], ARGS["h"], ARGS["w"], device="cuda")
        w = torch.randn(ARGS["cout"], ARGS["cin"], ARGS["k"], ARGS["k"], device="cuda")
        fan_in = w.numel() // w.shape[0]
        w.div_(fan_in ** 0.5)
        if g != 1.0:
            w.mul_(g)
        A = conv_fwd(x, w)
        A2 = conv_fwd(x, w)
        rec["a_repeat_bit_identical"] = bool(torch.equal(A, A2))
        del A2
        rec["maxabs_out"] = float(A.abs().max().item())
        rec["u_1e-4"] = REF_REL * rec["maxabs_out"] / STRICT

        torch.backends.cudnn.allow_tf32 = False
        X = conv_fwd(x.double(), w.double())
        X2 = conv_fwd(x.double(), w.double())
        rec["x_repeat_bit_identical"] = bool(torch.equal(X, X2))
        del X2
        torch.backends.cudnn.allow_tf32 = True

        perm = np.random.RandomState(PERM_SEED).permutation(DEPTH).tolist()
        rec["perm"] = perm
        pc = build_parts(x, w, "contiguous")
        pi = build_parts(x, w, "interleaved")
        S = torch.zeros_like(pc[0], dtype=torch.float64)
        for p in pc:
            S += p.double()
        rec["part_stats"] = dict(
            contig_maxabs=max(float(p.abs().max().item()) for p in pc),
            contig_meanabs=float(np.mean([float(p.abs().mean().item()) for p in pc])),
            inter_maxabs=max(float(p.abs().max().item()) for p in pi),
            inter_meanabs=float(np.mean([float(p.abs().mean().item()) for p in pi])),
        )
        arms = dict(C_seq=merge_seq(pc, list(range(DEPTH))),
                    C_shuf=merge_seq(pc, perm),
                    C_bal=merge_bal(pc),
                    I_seq=merge_seq(pi, list(range(DEPTH))))
        rep = merge_seq(pc, list(range(DEPTH)))
        rec["c_seq_repeat_bit_identical"] = bool(torch.equal(arms["C_seq"], rep))
        del rep, pc, pi

        if True:  # K3 at both g values, as s104 required
            evs = {}
            for name, kind, mg in (("C_seq", "contiguous", "seq"), ("C_shuf", "contiguous", "shuf"),
                                   ("C_bal", "contiguous", "bal"), ("I_seq", "interleaved", "seq")):
                def _run(kind=kind, mg=mg):
                    pp = build_parts(x, w, kind)
                    if mg == "bal":
                        return merge_bal(pp)
                    return merge_seq(pp, list(range(DEPTH)) if mg == "seq" else perm)
                ev = profile_call(_run)
                cls, top1 = classify([n for n, _ in ev])
                evs[name] = dict(class_=cls, top1=top1)
            rec["engine_arms"] = evs
            ev2 = profile_call(lambda: conv_fwd(x, w))
            rec["class_A"], rec["top1_A"] = classify([n for n, _ in ev2])

        for name, P in arms.items():
            nf_p, mr_p, nt = chunked_compare(A, P, STRICT, STRICT)
            _, mr_px, _ = chunked_compare(X, P, STRICT, STRICT)
            _, mr_ps, _ = chunked_compare(S, P.double(), STRICT, STRICT)
            nf_sa, mr_sa, nts = chunked_compare(A, S, STRICT, STRICT)
            _, mr_sx, _ = chunked_compare(X, S, STRICT, STRICT)
            rec[name] = dict(rho_vs_ref=mr_p, frac_vs_ref=nf_p / nt, rho_vs_fp64=mr_px,
                             rho_vs_S=mr_ps, argmax_ratio=mr_p)
        nf_a, mr_a, nta = chunked_compare(A, X, STRICT, STRICT)
        rec["A"] = dict(rho=mr_a, frac_over=nf_a / nta)
        rec["S"] = dict(rho_vs_ref=mr_sa, frac_vs_ref=nf_sa / nts, rho_vs_fp64=mr_sx)
        for name in arms:
            rec[name]["c"] = rec[name]["rho_vs_ref"] / mr_a if mr_a > 0 else None
            rec[name]["merge_share"] = rec[name]["rho_vs_S"] / rec[name]["rho_vs_ref"] if rec[name]["rho_vs_ref"] else None
        rec["t_s"] = round(time.time() - t0, 2)
        del A, X, S, arms, x, w
        torch.cuda.empty_cache()
    except Timeout:
        rec["status"] = "timeout"
    except torch.cuda.OutOfMemoryError as e:
        rec["status"] = "oom"; rec["err"] = str(e)[:200]; torch.cuda.empty_cache()
    except Exception as e:
        rec["status"] = "error"; rec["err"] = repr(e)[:300]; torch.cuda.empty_cache()
    finally:
        signal.alarm(0)
        rec.setdefault("t_s", round(time.time() - t0, 2))
    return rec


def main():
    torch.cuda.init()
    open(OUT, "w").close()
    sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    print("script sha", sha[:12], torch.__version__, torch.cuda.get_device_name(0), flush=True)
    rows, t_start = [], time.time()
    for g in GRID:
        for seed in SEEDS:
            if time.time() - t_start > GLOBAL_DEADLINE_S:
                print("GLOBAL DEADLINE", g, seed, flush=True); break
            r = run_point(g, seed)
            rows.append(r)
            with open(OUT, "a") as f:
                f.write(json.dumps(r) + "\n")
            if r["status"] == "ok":
                print("g %-5s seed %d rhoA %.3f | %s" % (g, seed, r["A"]["rho"],
                      " ".join("%s c %.5f" % (n, r[n]["c"]) for n in ("C_seq", "C_shuf", "C_bal", "I_seq"))),
                      flush=True)
            else:
                print("g %-5s seed %d %s %s" % (g, seed, r["status"], r.get("err", "")), flush=True)

    ok = [r for r in rows if r.get("status") == "ok"]
    summary = dict(script_sha=sha, leg="s104b post-hoc declared", parent="s104", n_attempted=len(rows),
                   n_ok=len(ok), grid=GRID, seeds=SEEDS, perm_seed=PERM_SEED, noise_floor=NOISE_FLOOR)
    ctrl = dict(k1=all(r.get("a_repeat_bit_identical") for r in ok),
                k4=all(r.get("c_seq_repeat_bit_identical") for r in ok),
                k5=all(r.get("x_repeat_bit_identical") for r in ok))
    cls = []
    for r in ok:
        for nm, v in r.get("engine_arms", {}).items():
            cls.append([r["g"], r["seed"], nm, v["class_"]])
    ctrl["k3_class_points"] = cls
    ctrl["k3_class_held"] = all(c == "TF32_TENSOROP" for _, _, _, c in cls) if cls else False
    summary["controls"] = ctrl
    summary["instrument_ok"] = bool(ctrl["k1"] and ctrl["k4"] and ctrl["k5"] and ctrl["k3_class_held"])

    # paired analysis
    paired = {}
    for g in GRID:
        grp = [r for r in ok if r["g"] == g and r["A"]["rho"] >= NOISE_FLOOR]
        if len(grp) < 4:
            paired[str(g)] = dict(status="INSUFFICIENT", n=len(grp)); continue
        base = [r["C_seq"]["c"] for r in grp]
        out = dict(n=len(grp), c_seq_median=float(np.median(base)),
                   c_seq_range=[min(base), max(base)])
        for name in ("C_shuf", "C_bal", "I_seq"):
            ratios = [r[name]["c"] / r["C_seq"]["c"] for r in grp]
            below = sum(1 for x in ratios if x < 1.0)
            out[name] = dict(median_ratio=float(np.median(ratios)), min=min(ratios), max=max(ratios),
                             n_below=below, n=len(ratios))
        paired[str(g)] = out
    summary["paired"] = paired

    ms = [paired[str(g)].get("I_seq", {}) for g in GRID]
    mem_ok = all(m.get("n_below", 0) >= 7 and m.get("median_ratio", 1.0) <= 0.90 for m in ms) \
        if all("n_below" in m for m in ms) else False
    shape_arms = ("C_bal", "C_shuf")
    shape_ok = any(all(paired[str(g)].get(a, {}).get("n_below", 0) >= 7 and
                       paired[str(g)].get(a, {}).get("median_ratio", 1.0) <= 0.90 for g in GRID)
                   for a in shape_arms) if all("n_below" in paired[str(g)].get(a, {}) for g in GRID for a in shape_arms) else False
    summary["membership_effect"] = bool(mem_ok)
    summary["merge_shape_effect"] = bool(shape_ok)
    if not summary["instrument_ok"]:
        summary["verdict"] = "INSTRUMENT_FAILED"
    elif mem_ok:
        summary["verdict"] = "MEMBERSHIP_EFFECT"
    elif shape_ok:
        summary["verdict"] = "MERGE_SHAPE_EFFECT"
    else:
        summary["verdict"] = "DRAW_NOISE"
    with open(SUMOUT, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ("verdict", "instrument_ok", "controls", "paired")}, indent=1)[:3000],
          flush=True)


if __name__ == "__main__":
    main()
