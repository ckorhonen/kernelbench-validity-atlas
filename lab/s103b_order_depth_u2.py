#!/usr/bin/env python
"""s103b_order_depth_u2.py — declared secondary leg of s103 (prereg lab/s103_prereg.md, "Secondary leg").

U2 = 3x3 conv, B=2, C_in=192 -> C_out=64, 151x151 (K=1728, 3x T1's depth) under the identical s101b
fixed-class permutation machinery. Question: does the order-term cost ratio continue to rise with
accumulation depth inside T1's own engine family (cudnn fprop), or does it saturate at K>=576?
Reported descriptively per the prereg; it does not alter s103's primary verdict.
"""
import json, os, re, signal, time, hashlib
import torch

LAB = os.path.expanduser("~/kb-lab")
OUT = os.path.join(LAB, "results", "s103b_order_depth_u2.jsonl")
SUMOUT = os.path.join(LAB, "results", "s103b_summary.json")
S101B = os.path.join(LAB, "results", "s101b_summary.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

CFG_ALARM_S = 600
GLOBAL_DEADLINE_S = 900
CHUNK_ELEMS = 1 << 22
STRICT, SHIPPED, REF_REL, NOISE_FLOOR = 1e-4, 1e-2, 2.2e-4, 0.05

ARGS = dict(n=2, cin=192, h=151, w=151, cout=64, k=3, s=1, p=1)
SEED = 1085  # T1's seed
GRID = [0.0625, 0.25, 1.0, 4.0, 16.0, 64.0]
DEPTHS = [4, 16, 64]

CLASS_PATTERNS = [
    ("WINOGRAD", re.compile(r"winograd", re.I)),
    ("TF32_TENSOROP", re.compile(r"tf32|tensorop|1688|16816|h1688", re.I)),
    ("FP64", re.compile(r"dgrad|dgemm|dconv|fp64|double|_d_", re.I)),
    ("FP32_SIMT", re.compile(r"fprop|implicit_gemm|sgemm|fp32|scudnn|simt", re.I)),
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

        if g in (1.0, 16.0):
            ev = profile_call(lambda: perm_zeropad_sequential(x, w, 16))
            rec["engine_P16z"] = [list(v) for v in ev]
            rec["class_P16z"], rec["top1_P16z"] = classify([n for n, _ in ev])
            ev2 = profile_call(lambda: conv_fwd(x, w))
            rec["class_A"], rec["top1_A"] = classify([n for n, _ in ev2])

        torch.backends.cudnn.allow_tf32 = False
        X = conv_fwd(x.double(), w.double())
        rec["x_maxabs"] = float(X.abs().max().item())
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


def interp_c(rows, arm, u):
    pts = sorted([(r["u"], r[arm]["c"]) for r in rows if r[arm]["c"] is not None], key=lambda t: t[0])
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
    prereg_sha = hashlib.sha256(open(os.path.join(LAB, "s103_prereg.md"), "rb").read()).hexdigest()
    print("script sha", sha[:12], "prereg sha", prereg_sha[:12], torch.__version__,
          torch.cuda.get_device_name(0), flush=True)
    rows, t_start = [], time.time()

    for g in GRID:
        if time.time() - t_start > GLOBAL_DEADLINE_S:
            print("GLOBAL DEADLINE at g", g, flush=True); break
        r = run_point(g)
        rows.append(r)
        with open(OUT, "a") as f:
            f.write(json.dumps(r) + "\n")
        line = "g %-8s %s" % (g, r["status"])
        if r["status"] == "ok":
            line += (" maxabs %.4f u1e4 %.2f rhoA1e4 %.4f fracA1e4 %.5f" %
                     (r["maxabs_out"], r["u_1e-4"], r["A_1e-4"]["rho"], r["A_1e-4"]["frac_over"]))
            for name in ("P4z", "P16z", "P64z"):
                line += " | %s r1e4 %.4f f1e4 %.5f fp64 %.4f" % (name, r[name + "_1e-4"]["rho_vs_ref"],
                                                                r[name + "_1e-4"]["frac_vs_ref"],
                                                                r[name + "_1e-4"]["rho_vs_fp64"])
            line += " | rep %s" % r.get("a_repeat_bit_identical")
            if "class_A" in r:
                line += " | clsA %s clsP16z %s" % (r.get("class_A"), r.get("class_P16z"))
        print(line, flush=True)

    ok = [r for r in rows if r.get("status") == "ok"]
    summary = dict(script_sha=sha, prereg_sha=prereg_sha, tid="U2", extension_of="s103",
                   post_hoc=True, n_attempted=len(rows), n_ok=len(ok), grid=GRID, depths=DEPTHS,
                   noise_floor=NOISE_FLOOR)
    summary["n_repeat_fail"] = sum(1 for r in ok if not r.get("a_repeat_bit_identical"))
    class_pts = [(r["g"], r.get("class_P16z"), r.get("class_A"), r.get("top1_A")) for r in ok if "class_P16z" in r]
    summary["class_points"] = class_pts
    summary["reference_instances"] = sorted(set(cp[3] for cp in class_pts if cp[3]))

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
    comparison = []
    for gate in ("1e-4", "1e-2"):
        for row in table[gate]:
            if row["rho_A"] < NOISE_FLOOR or row["u"] < 10:
                continue
            for name in ("P4z", "P16z", "P64z"):
                c2 = row[name]["c"]
                c1 = interp_c(t1[gate], name, row["u"])
                if c2 is None or c1 is None or c1 <= 0:
                    continue
                comparison.append(dict(gate=gate, g=row["g"], u=row["u"], arm=name,
                                       c_u2=c2, c_t1=c1, ratio=c2 / c1))
    summary["matched_u_vs_t1"] = comparison
    ratios = [c["ratio"] for c in comparison]
    summary["n_evaluable"] = len(ratios)
    summary["ratio_median"] = sorted(ratios)[len(ratios) // 2] if ratios else None
    summary["ratio_min"] = min(ratios) if ratios else None
    summary["ratio_max"] = max(ratios) if ratios else None
    summary["expected_if_k_pow_0.8"] = 3.0 ** 0.8
    summary["expected_if_flat"] = 1.0

    cs = [row[n]["c"] for row in table["1e-4"] if row["u"] >= 10 and row["rho_A"] >= NOISE_FLOOR
          for n in ("P4z", "P16z", "P64z") if row[n]["c"]]
    summary["u2_flatness_above_u10"] = dict(n=len(cs), c_min=min(cs) if cs else None,
                                            c_max=max(cs) if cs else None,
                                            spread=(max(cs) / min(cs)) if cs and min(cs) > 0 else None)

    with open(SUMOUT, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ("n_ok", "n_repeat_fail", "class_points", "reference_instances",
                                             "n_evaluable", "ratio_median", "ratio_min", "ratio_max",
                                             "expected_if_k_pow_0.8", "u2_flatness_above_u10")}, indent=1)[:2500], flush=True)
    print("matched:", json.dumps(comparison)[:1500], flush=True)


if __name__ == "__main__":
    main()
