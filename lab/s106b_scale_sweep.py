#!/usr/bin/env python
"""s106b_scale_sweep.py — scale sweep on the s106 matched-K conv-vs-gemm ladder (prereg lab/s106b_prereg.md)."""
import json, os, re, signal, time, hashlib
import torch

LAB = os.path.expanduser("~/kb-lab")
OUT = os.path.join(LAB, "results", "s106b_scale_sweep.jsonl")
SUM = os.path.join(LAB, "results", "s106b_summary.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
CFG_ALARM_S, CHUNK_ELEMS = 600, 1 << 22
STRICT, SHIPPED, REF_REL = 1e-4, 1e-2, 2.2e-4
GS = [0.015625, 0.0625, 0.25, 1.0, 4.0, 16.0]
DRAWS = 2
SEED0 = 1200
ARMS = {"G": dict(kind="gemm", M=1708, N=1708, K=1728),
        "C1": dict(kind="conv", n=2, cin=1728, cout=64, h=151, w=151, k=1, s=1, p=0),
        "C3": dict(kind="conv", n=2, cin=192, cout=64, h=151, w=151, k=3, s=1, p=1)}
CLASS_PATTERNS = [("WINOGRAD", re.compile(r"winograd", re.I)),
                  ("TF32_TENSOROP", re.compile(r"tf32|tensorop|1688|16816|h1688", re.I)),
                  ("FP64", re.compile(r"dgrad|dgemm|dconv|fp64|double|_d_", re.I)),
                  ("FP32_SIMT", re.compile(r"fprop|implicit_gemm|sgemm|fp32|scudnn|simt", re.I)),
                  ("CUTLASS", re.compile(r"cutlass", re.I))]


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
    return evts[:3]


def build(spec, seed, g):
    torch.manual_seed(seed)
    if spec["kind"] == "gemm":
        a = torch.randn(spec["M"], spec["K"], device="cuda", dtype=torch.float32)
        b = torch.randn(spec["K"], spec["N"], device="cuda", dtype=torch.float32)
        b.div_(spec["K"] ** 0.5).mul_(g)
        return a, b
    x = torch.randn(spec["n"], spec["cin"], spec["h"], spec["w"], device="cuda", dtype=torch.float32)
    w = torch.randn(spec["cout"], spec["cin"], spec["k"], spec["k"], device="cuda", dtype=torch.float32)
    w.div_((spec["cin"] * spec["k"] * spec["k"]) ** 0.5).mul_(g)
    return x, w


def fwd(spec, a, b):
    if spec["kind"] == "gemm":
        return torch.matmul(a, b)
    return torch.nn.functional.conv2d(a, b, stride=spec["s"], padding=spec["p"])


def chunked_compare(ref, cand, atol, rtol):
    n_fail, max_ratio, n_tot = 0, 0.0, 0
    rf, cf = ref.detach().reshape(-1), cand.detach().reshape(-1)
    n, i = rf.numel(), 0
    while i < n:
        j = min(n, i + CHUNK_ELEMS)
        d = (rf[i:j] - cf[i:j]).abs()
        bb = atol + rtol * cf[i:j].abs()
        r = d / bb
        n_fail += int((r > 1).sum().item())
        mr = float(torch.max(r, 0)[0].item())
        max_ratio = max(max_ratio, mr)
        n_tot += (j - i)
        del d, bb, r
        i = j
    return n_fail, max_ratio, n_tot


def run_row(arm, g, d, do_prof):
    spec = ARMS[arm]
    rec = dict(arm=arm, g=g, draw=d, kind=spec["kind"], status="ok")
    t0 = time.time()
    signal.alarm(CFG_ALARM_S)
    try:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        a, b = build(spec, SEED0 + d, g)
        A = fwd(spec, a, b)
        A2 = fwd(spec, a, b)
        rec["a_repeat_bit_identical"] = bool(torch.equal(A, A2))
        del A2
        rec["maxabs_out"] = float(A.abs().max().item())
        rec["n_elems"] = int(A.numel())
        if do_prof:
            ev = profile_call(lambda: fwd(spec, a, b))
            rec["class"], rec["top1"] = classify([n for n, _ in ev])
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        ad, bd = a.double(), b.double()
        del a, b
        X = fwd(spec, ad, bd)
        rec["x_maxabs"] = float(X.abs().max().item())
        del ad, bd
        for tag, tol in (("1e-4", STRICT), ("1e-2", SHIPPED)):
            nf, mr, nt = chunked_compare(A, X, tol, tol)
            rec["rho_" + tag] = mr
            rec["frac_over_" + tag] = nf / nt
        rec["u_1e-4"] = REF_REL * rec["maxabs_out"] / STRICT
        rec["t_s"] = round(time.time() - t0, 2)
        del A, X
        torch.cuda.empty_cache()
    except Timeout:
        rec["status"] = "timeout"
    except Exception as e:
        rec["status"] = "error"; rec["err"] = repr(e)[:200]; torch.cuda.empty_cache()
    finally:
        signal.alarm(0)
        rec.setdefault("t_s", round(time.time() - t0, 2))
    return rec


def med(xs):
    xs = sorted(xs)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def main():
    torch.cuda.init()
    open(OUT, "w").close()
    sha = hashlib.sha256(open(__file__).read().encode()).hexdigest()
    psha = hashlib.sha256(open(os.path.join(LAB, "s106b_prereg.md"), "rb").read()).hexdigest()
    print("script sha", sha[:12], "prereg sha", psha[:12], torch.__version__,
          torch.cuda.get_device_name(0), flush=True)
    rows = []
    with open(OUT, "a") as f:
        for g in GS:
            for d in range(DRAWS):
                for arm in ("G", "C1", "C3"):
                    rec = run_row(arm, g, d, do_prof=(g in (GS[0], GS[-1]) and d == 0))
                    f.write(json.dumps(rec) + "\n"); f.flush()
                    rows.append(rec)
                    print(arm, g, d, rec.get("status"), rec.get("class"), rec.get("rho_1e-4"),
                          rec.get("frac_over_1e-4"), rec.get("maxabs_out"), flush=True)
    ok = [r for r in rows if r.get("status") == "ok"]
    levels = {}
    for g in GS:
        sub = [r for r in ok if r["g"] == g]
        m = {a: med([r["rho_1e-4"] for r in sub if r["arm"] == a]) for a in ("G", "C1", "C3")}
        fm = {a: med([r["frac_over_1e-4"] for r in sub if r["arm"] == a]) for a in ("G", "C1", "C3")}
        levels[str(g)] = dict(rho=m, frac=fm, n=len(sub),
                              R1=(m["C1"] / m["G"]) if m["G"] else None,
                              R3=(m["C3"] / m["G"]) if m["G"] else None,
                              u=med([r["u_1e-4"] for r in sub]),
                              maxabs=med([r["maxabs_out"] for r in sub]))
    worst = max(max(v["R1"] or 0, v["R3"] or 0) for v in levels.values())
    branch = "BRANCH_CLASS_GAP_INVARIANT" if worst < 2.0 else "BRANCH_CLASS_GAP_AT_SCALE"
    # scale law per arm
    import math
    slopes = {}
    for a in ("G", "C1", "C3"):
        pts = [(math.log(v["u"]), math.log(v["rho"][a])) for v in levels.values()
               if v["rho"][a] and v["u"] and v["rho"][a] > 0]
        if len(pts) >= 3:
            n = len(pts); mx = sum(p[0] for p in pts) / n; my = sum(p[1] for p in pts) / n
            num = sum((p[0] - mx) * (p[1] - my) for p in pts)
            den = sum((p[0] - mx) ** 2 for p in pts)
            slopes[a] = num / den if den else None
    summary = dict(date=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), script_sha=sha, prereg_sha=psha,
                   levels=levels, worst_ratio=worst, branch=branch, loglog_slope_vs_u=slopes,
                   controls=dict(k1_all_bit_identical=all(r.get("a_repeat_bit_identical") for r in ok),
                                 classes=[(str(r["g"]), r["arm"], r.get("class")) for r in ok if r.get("class")]),
                   n_ok=len(ok), n_rows=len(rows))
    json.dump(summary, open(SUM, "w"), indent=1)
    print(json.dumps({k: summary[k] for k in ("branch", "worst_ratio", "loglog_slope_vs_u", "n_ok")}))
    print(json.dumps(levels, indent=1))


if __name__ == "__main__":
    main()
