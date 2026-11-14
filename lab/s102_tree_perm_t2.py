#!/usr/bin/env python
"""s102_tree_perm_t2.py — second tuple under the s101b declared fixed-class tree permutation.

Prereg: lab/s102_prereg.md, committed BEFORE any row of this run. This script implements that design verbatim:
T2 (1x1 conv, B=2, C=192->160, 211x211, seed 1068), g in {0.0625,0.25,1.0,4.0,16.0,64.0,128.0}, arms A/X/P4z/P16z/P64z
with each chunk zero-padded back to the full C_in=192 so the engine class is held, evaluator-exact predicates at 1e-4
and 1e-2, controls K1-K4, and the frozen comparison against T1's recorded cost curve from results/s101b_summary.json.
"""
import json, os, re, signal, time, hashlib
import torch

LAB = os.path.expanduser("~/kb-lab")
OUT = os.path.join(LAB, "results", "s102_tree_perm_t2.jsonl")
SUMOUT = os.path.join(LAB, "results", "s102_summary.json")
S101B = os.path.join(LAB, "results", "s101b_summary.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

CFG_ALARM_S = 420
GLOBAL_DEADLINE_S = 1500
CHUNK_ELEMS = 1 << 22
STRICT, SHIPPED, REF_REL, NOISE_FLOOR = 1e-4, 1e-2, 2.2e-4, 0.05

ARGS = dict(n=2, cin=192, h=211, w=211, cout=160, k=1, s=1, p=0)
SEED = 1068
GRID = [0.0625, 0.25, 1.0, 4.0, 16.0, 64.0, 128.0]
DEPTHS = [4, 16, 64]
PROBE_CINS = [8, 16, 32, 64, 192]

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


def perm_zeropad_sequential(x, w, k):
    out = None
    step = x.shape[1] // k
    for i in range(k):
        lo, hi = i * step, (i + 1) * step
        xc = torch.zeros_like(x)
        xc[:, lo:hi] = x[:, lo:hi]
        wc = torch.zeros_like(w)
        wc[:, lo:hi] = w[:, lo:hi]
        p = conv_fwd(xc, wc)
        out = p if out is None else out + p
        del xc, wc, p
    return out


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

        arms, times = {}, {}
        for k in DEPTHS:
            t = time.time()
            arms["P%dz" % k] = perm_zeropad_sequential(x, w, k)
            times["P%dz" % k] = round(time.time() - t, 2)
        rec["arm_times"] = times

        if g in (1.0, 64.0):
            ev = profile_call(lambda: perm_zeropad_sequential(x, w, 16))
            rec["engine_P16z"] = [list(v) for v in ev]
            rec["class_P16z"], rec["top1_P16z"] = classify([n for n, _ in ev])
            ev2 = profile_call(lambda: conv_fwd(x, w))
            rec["class_A"], rec["top1_A"] = classify([n for n, _ in ev2])

        torch.backends.cudnn.allow_tf32 = False
        X = conv_fwd(x.double(), w.double())
        del x, w

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


def interp_c(t1_rows, arm, gate, u):
    """log-u linear interpolation of T1's recorded c at matched u; None if outside T1's range."""
    pts = sorted([(r["u"], r[arm]["c"]) for r in t1_rows
                  if r[arm]["c"] is not None], key=lambda t: t[0])
    if not pts or u < pts[0][0] or u > pts[-1][0]:
        return None
    for (u0, c0), (u1, c1) in zip(pts, pts[1:]):
        if u0 <= u <= u1:
            if u1 == u0:
                return c0
            w = (torch.log(torch.tensor(u / u0)) / torch.log(torch.tensor(u1 / u0))).item()
            return c0 * (c1 / c0) ** w
    return pts[-1][1]


def main():
    torch.cuda.init()
    open(OUT, "w").close()
    sha = hashlib.sha256(open(__file__).read().encode()).hexdigest()
    prereg_sha = hashlib.sha256(open(os.path.join(LAB, "s102_prereg.md"), "rb").read()).hexdigest()
    print("script sha", sha[:12], "prereg sha", prereg_sha[:12], torch.__version__,
          torch.cuda.get_device_name(0), flush=True)
    rows, t_start = [], time.time()

    probe = []
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(SEED)
    for cin in PROBE_CINS:
        x = torch.randn(ARGS["n"], cin, ARGS["h"], ARGS["w"], device="cuda")
        w = torch.randn(ARGS["cout"], cin, ARGS["k"], ARGS["k"], device="cuda")
        ev = profile_call(lambda: conv_fwd(x, w))
        cls, top1 = classify([n for n, _ in ev])
        probe.append(dict(cin=cin, class_=cls, top1=top1))
        print("probe cin", cin, cls, top1[:90], flush=True)
        del x, w
    torch.cuda.empty_cache()

    for g in GRID:
        if time.time() - t_start > GLOBAL_DEADLINE_S:
            print("GLOBAL DEADLINE at g", g, flush=True); break
        r = run_point(g)
        rows.append(r)
        with open(OUT, "a") as f:
            f.write(json.dumps(r) + "\n")
        line = "g %-8s %s" % (g, r["status"])
        if r["status"] == "ok":
            line += (" maxabs %.4f u1e4 %.2f rhoA1e4 %.4f fracA1e4 %.5f rhoA1e2 %.4f" %
                     (r["maxabs_out"], r["u_1e-4"], r["A_1e-4"]["rho"], r["A_1e-4"]["frac_over"],
                      r["A_1e-2"]["rho"]))
            for name in ("P4z", "P16z", "P64z"):
                line += " | %s r1e4 %.4f f1e4 %.5f fp64 %.4f" % (name, r[name + "_1e-4"]["rho_vs_ref"],
                                                                r[name + "_1e-4"]["frac_vs_ref"],
                                                                r[name + "_1e-4"]["rho_vs_fp64"])
            line += " | rep %s" % r.get("a_repeat_bit_identical")
            if "class_P16z" in r:
                line += " | clsA %s clsP16z %s" % (r.get("class_A"), r.get("class_P16z"))
        print(line, flush=True)

    ok = [r for r in rows if r.get("status") == "ok"]
    summary = dict(script_sha=sha, prereg_sha=prereg_sha, tid="T2", extension_of="s101b",
                   n_attempted=len(rows), n_ok=len(ok), grid=GRID, depths=DEPTHS,
                   dispatch_probe=probe, noise_floor=NOISE_FLOOR)

    g1 = [r for r in ok if r["g"] == 1.0]
    tie = dict(status="missing")
    if g1:
        got = g1[0]["A_1e-2"]["rho"]
        rel = abs(got - 0.14734) / 0.14734
        tie = dict(rho_got=got, s91_t2_rho=0.14734, rel=rel, pass_=bool(rel <= 0.02))
    summary["tieout"] = tie
    summary["n_repeat_fail"] = sum(1 for r in ok if not r.get("a_repeat_bit_identical"))
    class_pts = [(r["g"], r.get("class_P16z"), r.get("class_A")) for r in ok if "class_P16z" in r]
    summary["class_points"] = class_pts
    summary["class_held"] = all(cp[1] == "TF32_TENSOROP" and cp[2] == "TF32_TENSOROP" for cp in class_pts) \
        if class_pts else False
    instr_ok = bool(tie.get("pass_")) and summary["n_repeat_fail"] == 0 and summary["class_held"]
    summary["instrument_ok"] = instr_ok

    table = {}
    for gate in ("1e-4", "1e-2"):
        table[gate] = []
        for r in sorted(ok, key=lambda z: z["g"]):
            rho_a = r["A_" + gate]["rho"]
            row = dict(g=r["g"], u=r["u_" + gate], rho_A=rho_a, frac_A=r["A_" + gate]["frac_over"])
            for name in ("P4z", "P16z", "P64z"):
                rp = r[name + "_" + gate]["rho_vs_ref"]
                row[name] = dict(rho=rp, frac=r[name + "_" + gate]["frac_vs_ref"],
                                 rho_vs_fp64=r[name + "_" + gate]["rho_vs_fp64"],
                                 c=(rp / rho_a) if rho_a > 0 else None)
            table[gate].append(row)
    summary["cost_table"] = table

    t1 = json.load(open(S101B))["cost_table"]
    comparison, spread_by_gate = [], {}
    for gate in ("1e-4", "1e-2"):
        cs = []
        for row in table[gate]:
            if row["rho_A"] < NOISE_FLOOR:
                continue
            for name in ("P4z", "P16z", "P64z"):
                c2 = row[name]["c"]
                c1 = interp_c(t1[gate], name, gate, row["u"])
                if c2 is None or c1 is None or c1 <= 0:
                    continue
                ratio = c2 / c1
                comparison.append(dict(gate=gate, g=row["g"], u=row["u"], arm=name,
                                       c_t2=c2, c_t1=c1, ratio=ratio, above_u10=bool(row["u"] >= 10)))
                cs.append(c2)
        spread_by_gate[gate] = dict(n=len(cs), c_min=min(cs) if cs else None,
                                    c_max=max(cs) if cs else None,
                                    spread=(max(cs) / min(cs)) if cs and min(cs) > 0 else None)
    summary["t1_comparison"] = comparison
    summary["spread_by_gate"] = spread_by_gate

    low_u = [c for c in comparison if c["u"] < 3]
    summary["low_u_spread"] = dict(n=len(low_u),
                                   spread=(max(x["c_t2"] for x in low_u) / min(x["c_t2"] for x in low_u))
                                   if low_u else None,
                                   t1_reference="2.03-2.12x at u<3 (s101b)")

    ratios = [c["ratio"] for c in comparison]
    viol = [c for c in comparison if c["ratio"] > 1.5 or c["ratio"] < (1 / 1.5)]
    summary["ratio_max"] = max(ratios) if ratios else None
    summary["ratio_min"] = min(ratios) if ratios else None
    summary["violations"] = viol
    summary["above_u10"] = [c for c in comparison if c["above_u10"]]

    if not instr_ok:
        summary["verdict"] = "INSTRUMENT_FAILED"
    elif viol:
        summary["verdict"] = "BRANCH_TUPLE_SPECIFIC"
    else:
        summary["verdict"] = "BRANCH_TUPLE_AGREES"

    # descriptive: permutation's own crossing bracket per gate
    cross = {}
    for gate in ("1e-4", "1e-2"):
        for name in ("P4z", "P16z", "P64z"):
            lo = hi = None
            prev = None
            for row in table[gate]:
                v = row[name]["rho"]
                if prev is not None and prev[1] < 1.0 <= v and lo is None:
                    lo, hi = prev[0], row["u"]
                prev = (row["u"], v)
            cross[gate + "_" + name] = dict(u_lo=lo, u_hi=hi)
    summary["perm_crossing"] = cross
    summary["depth_increment"] = {
        gate: [(row["g"], round(row["P64z"]["rho"] / row["P4z"]["rho"], 3)) for row in table[gate]]
        for gate in ("1e-4", "1e-2")}

    with open(SUMOUT, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ("verdict", "tieout", "instrument_ok", "class_held",
                                             "class_points", "ratio_max", "ratio_min",
                                             "spread_by_gate", "low_u_spread")}, indent=1)[:2500], flush=True)
    print("above u10:", json.dumps(summary["above_u10"])[:1200], flush=True)
    print("violations:", json.dumps(viol)[:800], flush=True)


if __name__ == "__main__":
    main()
