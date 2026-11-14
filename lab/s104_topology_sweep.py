#!/usr/bin/env python
"""s104_topology_sweep.py — E3 merge-tree sweep (prereg lab/s104_prereg.md, committed with this runner
BEFORE any row). Closes the five-tick E3 deferral: does the order term depend on tree SHAPE at matched
depth, or only on depth?

Arms (all depth 16, zero-padded chunks so the TF32 tensorop engine is held):
  C_seq  contiguous membership, sequential ascending merge (s101b P16z replica, K2 tie-out)
  C_shuf identical membership, sequential merge in a fixed declared permutation (seed 20261114)
  C_bal  identical membership, fan-in-2 balanced merge (4 levels, declared)
  I_seq  interleaved membership (stride 16), sequential ascending merge
Depth controls: P4z, P64z (contiguous sequential at 4 / 64 chunks).
"""
import json, os, re, signal, time, hashlib
import numpy as np
import torch

LAB = os.path.expanduser("~/kb-lab")
OUT = os.path.join(LAB, "results", "s104_topology_sweep.jsonl")
SUMOUT = os.path.join(LAB, "results", "s104_summary.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

CFG_ALARM_S = 300
GLOBAL_DEADLINE_S = 900
CHUNK_ELEMS = 1 << 22
STRICT, SHIPPED, REF_REL, NOISE_FLOOR = 1e-4, 1e-2, 2.2e-4, 0.05
TOPO_BAR = 1.20

ARGS = dict(n=2, cin=64, h=151, w=151, cout=64, k=3, s=1, p=1)
SEED = 1085
GRID = [1.0, 4.0, 16.0, 64.0]
DEPTH = 16
PERM_SEED = 20261114
CONTROLS_G = [1.0, 64.0]

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


def build():
    x = torch.randn(ARGS["n"], ARGS["cin"], ARGS["h"], ARGS["w"], device="cuda", dtype=torch.float32)
    w = torch.randn(ARGS["cout"], ARGS["cin"], ARGS["k"], ARGS["k"], device="cuda", dtype=torch.float32)
    return x, w


def channel_groups(kind, k):
    """Return a list of channel-index lists (each of length C_in // k)."""
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


def merge_sequential(parts, order):
    out = parts[order[0]]
    for i in order[1:]:
        out = out + parts[i]
    return out


def merge_balanced(parts):
    level = list(parts)
    steps = []
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level) - 1, 2):
            nxt.append(level[i] + level[i + 1])
            steps.append([i, i + 1])
        if len(level) % 2 == 1:
            nxt.append(level[-1])
        level = nxt
    return level[0], steps


def build_arm(x, w, kind, merge, perm=None):
    groups = channel_groups(kind, DEPTH)
    parts = [partial(x, w, g) for g in groups]
    if merge == "sequential":
        order = list(range(DEPTH)) if perm is None else perm
        out = merge_sequential(parts, order)
        steps = [[order[i], order[i + 1]] for i in range(DEPTH - 1)]
    else:
        out, steps = merge_balanced(parts)
        order = None
    del parts
    return out, dict(membership=kind, merge=merge, order=order, merge_steps=steps)


def chunked_compare(ref, cand, atol, rtol):
    n_fail, max_ratio, n_tot = 0, 0.0, 0
    best = dict(ratio=-1.0, ref=0.0, cand=0.0, d=0.0)
    rf, cf = ref.detach().reshape(-1), cand.detach().reshape(-1)
    n, i = rf.numel(), 0
    while i < n:
        j = min(n, i + CHUNK_ELEMS)
        d = (rf[i:j] - cf[i:j]).abs()
        b = atol + rtol * cf[i:j].abs()
        r = d / b
        n_fail += int((r > 1).sum().item())
        mr, am = torch.max(r, 0)
        mr = float(mr.item())
        if mr > max_ratio:
            max_ratio = mr
            kk = int(am.item())
            best = dict(ratio=mr, ref=float(rf[i + kk].item()), cand=float(cf[i + kk].item()),
                        d=float(d[kk].item()))
        n_tot += (j - i)
        del d, b, r
        i = j
    return n_fail, max_ratio, n_tot, best


def run_point(g):
    rec = dict(g=g, status="ok")
    t0 = time.time()
    signal.alarm(CFG_ALARM_S)
    try:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(SEED)
        x, w = build()
        fan_in = w.numel() // w.shape[0]
        w.div_(fan_in ** 0.5)
        if g != 1.0:
            w.mul_(g)
        A = conv_fwd(x, w)
        A2 = conv_fwd(x, w)
        rec["a_repeat_bit_identical"] = bool(torch.equal(A, A2))
        del A2
        rec["maxabs_out"] = float(A.abs().max().item())
        rec["n_elems"] = int(A.numel())

        perm = np.random.RandomState(PERM_SEED).permutation(DEPTH).tolist()
        rec["perm"] = perm
        specs = [("C_seq", "contiguous", "sequential", None),
                 ("C_shuf", "contiguous", "sequential", perm),
                 ("C_bal", "contiguous", "balanced", None),
                 ("I_seq", "interleaved", "sequential", None)]
        arms, times, trees = {}, {}, {}
        for name, kind, merge, pm in specs:
            t = time.time()
            arms[name], trees[name] = build_arm(x, w, kind, merge, pm)
            times[name] = round(time.time() - t, 2)
        for name, k in (("P4z", 4), ("P64z", 64)):
            t = time.time()
            groups = channel_groups("contiguous", k)
            parts = [partial(x, w, gg) for gg in groups]
            arms[name] = merge_sequential(parts, list(range(k)))
            trees[name] = dict(membership="contiguous", merge="sequential", order=list(range(k)))
            times[name] = round(time.time() - t, 2)
            del parts
        rec["arm_times"] = times
        rec["trees"] = trees

        # K4: determinism of the declared tree
        rep, _ = build_arm(x, w, "contiguous", "sequential", None)
        rec["c_seq_repeat_bit_identical"] = bool(torch.equal(arms["C_seq"], rep))
        del rep

        if g in CONTROLS_G:
            evs = {}
            for name, kind, merge, pm in specs:
                ev = profile_call(lambda nm=name: arms[nm].clone())
                cls, top1 = classify([n for n, _ in ev])
                evs[name] = dict(class_=cls, top1=top1)
            rec["engine_arms"] = evs
            ev2 = profile_call(lambda: conv_fwd(x, w))
            rec["class_A"], rec["top1_A"] = classify([n for n, _ in ev2])

        torch.backends.cudnn.allow_tf32 = False
        X = conv_fwd(x.double(), w.double())
        X2 = conv_fwd(x.double(), w.double())
        rec["x_repeat_bit_identical"] = bool(torch.equal(X, X2))
        del X2, x, w

        for tag, tol in (("1e-4", STRICT), ("1e-2", SHIPPED)):
            nf_a, mr_a, nt, diag = chunked_compare(A, X, tol, tol)
            rec["A_" + tag] = dict(n_fail=nf_a, rho=mr_a, frac_over=nf_a / nt, argmax=diag)
            for name, P in arms.items():
                nf_p, mr_p, nt2, dp = chunked_compare(A, P, tol, tol)
                nf_px, mr_px, _, _ = chunked_compare(X, P, tol, tol)
                rec[name + "_" + tag] = dict(rho_vs_ref=mr_p, frac_vs_ref=nf_p / nt2,
                                             rho_vs_fp64=mr_px, frac_vs_fp64=nf_px / nt2, argmax=dp)
        rec["u_1e-4"] = REF_REL * rec["maxabs_out"] / STRICT
        rec["u_1e-2"] = REF_REL * rec["maxabs_out"] / SHIPPED
        rec["t_s"] = round(time.time() - t0, 2)
        del A, X, arms
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
        if time.time() - t_start > GLOBAL_DEADLINE_S:
            print("GLOBAL DEADLINE at g", g, flush=True); break
        r = run_point(g)
        rows.append(r)
        with open(OUT, "a") as f:
            f.write(json.dumps(r) + "\n")
        line = "g %-6s %s" % (g, r["status"])
        if r["status"] == "ok":
            line += (" maxabs %.4f u1e4 %.1f rhoA1e4 %.3f" % (r["maxabs_out"], r["u_1e-4"],
                                                             r["A_1e-4"]["rho"]))
            for name in ("C_seq", "C_shuf", "C_bal", "I_seq", "P4z", "P64z"):
                line += " | %s r1e4 %.4f f %.4f x %.4f" % (name, r[name + "_1e-4"]["rho_vs_ref"],
                                                          r[name + "_1e-4"]["frac_vs_ref"],
                                                          r[name + "_1e-4"]["rho_vs_fp64"])
            line += " | repA %s repC %s repX %s" % (r.get("a_repeat_bit_identical"),
                                                    r.get("c_seq_repeat_bit_identical"),
                                                    r.get("x_repeat_bit_identical"))
        print(line, flush=True)

    ok = [r for r in rows if r.get("status") == "ok"]
    summary = dict(script_sha=sha, prereg="lab/s104_prereg.md", n_attempted=len(rows), n_ok=len(ok),
                   grid=GRID, depth=DEPTH, perm_seed=PERM_SEED, noise_floor=NOISE_FLOOR,
                   topo_bar=TOPO_BAR)

    # controls
    ctrl = dict(
        k1_a_repeat=all(r.get("a_repeat_bit_identical") for r in ok),
        k4_c_repeat=all(r.get("c_seq_repeat_bit_identical") for r in ok),
        k5_x_repeat=all(r.get("x_repeat_bit_identical") for r in ok),
    )
    g1 = [r for r in ok if r["g"] == 1.0]
    tie = dict(status="missing")
    if g1:
        got = g1[0]["C_seq_1e-4"]["rho_vs_ref"]
        rel = abs(got - 0.0376) / 0.0376
        tie = dict(rho_got=got, s101b_p16z=0.0376, rel=rel, pass_=bool(rel <= 0.02))
    ctrl["k2_tieout"] = tie
    cls_pts = []
    for r in ok:
        if "engine_arms" in r:
            for nm, v in r["engine_arms"].items():
                cls_pts.append([r["g"], nm, v["class_"]])
    ctrl["k3_class_points"] = cls_pts
    ctrl["k3_class_held"] = all(c == "TF32_TENSOROP" for _, _, c in cls_pts) if cls_pts else False
    summary["controls"] = ctrl
    instr_ok = bool(tie.get("pass_")) and ctrl["k1_a_repeat"] and ctrl["k4_c_repeat"] \
        and ctrl["k5_x_repeat"] and ctrl["k3_class_held"]
    summary["instrument_ok"] = instr_ok

    TOPO = ["C_seq", "C_shuf", "C_bal", "I_seq"]
    table, verdicts = {}, {}
    for gate in ("1e-4", "1e-2"):
        rows_g = []
        for r in sorted(ok, key=lambda z: z["g"]):
            rho_a = r["A_" + gate]["rho"]
            row = dict(g=r["g"], u=r["u_" + gate], rho_A=rho_a, frac_A=r["A_" + gate]["frac_over"],
                       qualifies=bool(rho_a >= NOISE_FLOOR))
            for name in TOPO + ["P4z", "P64z"]:
                rp = r[name + "_" + gate]["rho_vs_ref"]
                row[name] = dict(rho=rp, frac=r[name + "_" + gate]["frac_vs_ref"],
                                 rho_vs_fp64=r[name + "_" + gate]["rho_vs_fp64"],
                                 acc_ratio=r[name + "_" + gate]["rho_vs_fp64"] / rho_a if rho_a > 0 else None,
                                 c=(rp / rho_a) if rho_a > 0 else None)
            rows_g.append(row)
        table[gate] = rows_g
        spreads = []
        for row in rows_g:
            if not row["qualifies"]:
                continue
            cs = [row[n]["c"] for n in TOPO if row[n]["c"] is not None]
            if len(cs) == 4 and min(cs) > 0:
                spreads.append(dict(g=row["g"], u=row["u"], spread=max(cs) / min(cs),
                                    c={n: row[n]["c"] for n in TOPO}))
        verdicts[gate] = dict(n_qual=sum(1 for x in rows_g if x["qualifies"]), spreads=spreads,
                              max_spread=max([s["spread"] for s in spreads], default=None))
    summary["cost_table"] = table
    summary["topology_spread"] = verdicts

    if not instr_ok:
        summary["verdict"] = "INSTRUMENT_FAILED"
    else:
        gates_ok = [g for g in ("1e-4", "1e-2") if verdicts[g]["n_qual"] >= 2]
        if len(gates_ok) < 2:
            summary["verdict"] = "RANGE_INSUFFICIENT"
        else:
            ms = [verdicts[g]["max_spread"] for g in gates_ok]
            summary["verdict"] = "TOPOLOGY_DEPENDENT" if any(m > TOPO_BAR for m in ms) \
                else "TOPOLOGY_INVARIANT"

    with open(SUMOUT, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ("verdict", "instrument_ok", "controls")}, indent=1)[:2500],
          flush=True)
    print(json.dumps(summary["topology_spread"], indent=1)[:2500], flush=True)


if __name__ == "__main__":
    main()
