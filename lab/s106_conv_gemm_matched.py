#!/usr/bin/env python
"""s106_conv_gemm_matched.py — matched-K, matched-scale, matched-output conv-vs-gemm attribution ladder.

Prereg: lab/s106_prereg.md, committed BEFORE any row. Implements that design verbatim: arms G (matmul 1708x1728x1708),
C1 (conv 1x1, C_in=1728), C3 (conv 3x3, C_in=192), all implied K=1728 and ~2.917M outputs, unit-normal inputs with
weights scaled 1/sqrt(K), benchmark default flags, 8 paired draws, evaluator-exact rho at 1e-4 and 1e-2 against
element-chunked fp64 anchors, controls K1-K4, frozen branch rules.
"""
import json, os, re, signal, time, hashlib
import torch

LAB = os.path.expanduser("~/kb-lab")
OUT = os.path.join(LAB, "results", "s106_conv_gemm_matched.jsonl")
SUM = os.path.join(LAB, "results", "s106_summary.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

CFG_ALARM_S = 900
GLOBAL_DEADLINE_S = 2400
CHUNK_ELEMS = 1 << 22
STRICT, SHIPPED, REF_REL = 1e-4, 1e-2, 2.2e-4
DRAWS = 8
SEED0 = 1100

ARMS = {
    "G":  dict(kind="gemm", M=1708, N=1708, K=1728),
    "C1": dict(kind="conv", n=2, cin=1728, cout=64, h=151, w=151, k=1, s=1, p=0),
    "C3": dict(kind="conv", n=2, cin=192, cout=64, h=151, w=151, k=3, s=1, p=1),
}

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


def build(spec, seed):
    torch.manual_seed(seed)
    if spec["kind"] == "gemm":
        K = spec["K"]
        a = torch.randn(spec["M"], K, device="cuda", dtype=torch.float32)
        b = torch.randn(K, spec["N"], device="cuda", dtype=torch.float32)
        b.div_(K ** 0.5)
        return a, b
    x = torch.randn(spec["n"], spec["cin"], spec["h"], spec["w"], device="cuda", dtype=torch.float32)
    w = torch.randn(spec["cout"], spec["cin"], spec["k"], spec["k"], device="cuda", dtype=torch.float32)
    w.div_((spec["cin"] * spec["k"] * spec["k"]) ** 0.5)
    return x, w


def fwd(spec, a, b):
    if spec["kind"] == "gemm":
        return torch.matmul(a, b)
    return torch.nn.functional.conv2d(a, b, stride=spec["s"], padding=spec["p"])


def chunked_compare(ref, cand, atol, rtol):
    n_fail, max_ratio, n_tot = 0, 0.0, 0
    best = dict(ratio=-1.0, ref=0.0, cand=0.0, d=0.0)
    rf, cf = ref.detach().reshape(-1), cand.detach().reshape(-1)
    n, i = rf.numel(), 0
    while i < n:
        j = min(n, i + CHUNK_ELEMS)
        d = (rf[i:j] - cf[i:j]).abs()
        bb = atol + rtol * cf[i:j].abs()
        r = d / bb
        n_fail += int((r > 1).sum().item())
        mr, am = torch.max(r, 0)
        mr = float(mr.item())
        if mr > max_ratio:
            max_ratio = mr
            kk = int(am.item())
            best = dict(ratio=mr, ref=float(rf[i + kk].item()), cand=float(cf[i + kk].item()),
                        d=float(d[kk].item()))
        n_tot += (j - i)
        del d, bb, r
        i = j
    return n_fail, max_ratio, n_tot, best


def exact_frac(ref, cand):
    rf, cf = ref.detach().reshape(-1), cand.detach().reshape(-1)
    n, same, i = rf.numel(), 0, 0
    while i < n:
        j = min(n, i + CHUNK_ELEMS)
        same += int((rf[i:j] == cf[i:j]).sum().item())
        i = j
    return same / n


def run_row(arm, d, do_anchor_check=False):
    spec = ARMS[arm]
    rec = dict(arm=arm, draw=d, kind=spec["kind"], status="ok")
    t0 = time.time()
    signal.alarm(CFG_ALARM_S)
    try:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        a, b = build(spec, SEED0 + d)
        rec["implied_K"] = int(spec["K"] if spec["kind"] == "gemm" else spec["cin"] * spec["k"] * spec["k"])
        rec["matched_k"] = bool(rec["implied_K"] == 1728)
        A = fwd(spec, a, b)
        A2 = fwd(spec, a, b)
        rec["a_repeat_bit_identical"] = bool(torch.equal(A, A2))
        del A2
        rec["maxabs_out"] = float(A.abs().max().item())
        rec["n_elems"] = int(A.numel())

        ev = profile_call(lambda: fwd(spec, a, b))
        rec["engine"] = [list(v) for v in ev]
        rec["class"], rec["top1"] = classify([n for n, _ in ev])

        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        ad, bd = a.double(), b.double()
        del a, b
        X = fwd(spec, ad, bd)
        rec["x_maxabs"] = float(X.abs().max().item())
        if do_anchor_check:
            torch.cuda.empty_cache()
            X2 = fwd(spec, ad, bd)
            rec["anchor_selfcheck_rel"] = float((X - X2).abs().max().item() / max(rec["x_maxabs"], 1e-30))
            rec["anchor_selfcheck_bit_identical"] = bool(torch.equal(X, X2))
            del X2
        del ad, bd

        Xf = X.float()
        rec["exact_frac_vs_anchor"] = float(exact_frac(A, Xf))
        del Xf
        for tag, tol in (("1e-4", STRICT), ("1e-2", SHIPPED)):
            nf, mr, nt, diag = chunked_compare(A, X, tol, tol)
            rec["rho_" + tag] = mr
            rec["frac_over_" + tag] = nf / nt
            rec["argmax_" + tag] = diag
        rec["u_1e-4"] = REF_REL * rec["maxabs_out"] / STRICT
        rec["u_1e-2"] = REF_REL * rec["maxabs_out"] / SHIPPED
        rec["t_s"] = round(time.time() - t0, 2)
        del A, X
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


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def main():
    torch.cuda.init()
    open(OUT, "w").close()
    sha = hashlib.sha256(open(__file__).read().encode()).hexdigest()
    prereg_sha = hashlib.sha256(open(os.path.join(LAB, "s106_prereg.md"), "rb").read()).hexdigest()
    print("script sha", sha[:12], "prereg sha", prereg_sha[:12], torch.__version__,
          torch.cuda.get_device_name(0), flush=True)

    rows, t_start = [], time.time()
    with open(OUT, "a") as f:
        for d in range(DRAWS):
            for arm in ("G", "C1", "C3"):
                if time.time() - t_start > GLOBAL_DEADLINE_S:
                    rec = dict(arm=arm, draw=d, status="skipped_deadline")
                else:
                    rec = run_row(arm, d, do_anchor_check=(d == 0 and arm == "G"))
                f.write(json.dumps(rec) + "\n")
                f.flush()
                rows.append(rec)
                print(arm, d, rec.get("status"), rec.get("class"), rec.get("top1"),
                      rec.get("rho_1e-4"), rec.get("t_s"), flush=True)

    ok = [r for r in rows if r.get("status") == "ok" and r.get("matched_k")]
    by = {a: [r for r in ok if r["arm"] == a] for a in ("G", "C1", "C3")}
    med = {a: dict(n=len(v), rho_1e-4=median([r["rho_1e-4"] for r in v]),
                   rho_1e-2=median([r["rho_1e-2"] for r in v]),
                   frac_1e-4=median([r["frac_over_1e-4"] for r in v]),
                   maxabs=median([r["maxabs_out"] for r in v]),
                   exact_frac=median([r["exact_frac_vs_anchor"] for r in v]),
                   classes=sorted(set(r.get("class") for r in v)),
                   instances=sorted(set(r.get("top1") for r in v)))
            for a, v in by.items()}

    summary = dict(date=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), script_sha=sha,
                   prereg_sha=prereg_sha, draws=DRAWS, rows=len(rows), ok_rows=len(ok), medians=med,
                   controls=dict(k1_repeat_bit_identical_all=all(r.get("a_repeat_bit_identical") for r in ok),
                                 k2_anchor_selfcheck=[r.get("anchor_selfcheck_rel") for r in rows
                                                      if "anchor_selfcheck_rel" in r],
                                 k3_classes_per_draw=[[r.get("class") for r in ok if r["draw"] == d]
                                                      for d in range(DRAWS)]),
                   per_draw=[[dict(arm=r["arm"], rho_1e-4=r["rho_1e-4"], rho_1e-2=r["rho_1e-2"],
                                   frac_1e-4=r["frac_over_1e-4"], cls=r.get("class"), top1=r.get("top1"),
                                   maxabs=r["maxabs_out"]) for r in ok if r["draw"] == d]
                             for d in range(DRAWS)])

    # frozen branch rules
    g = med["G"]["rho_1e-4"]; c1 = med["C1"]["rho_1e-4"]; c3 = med["C3"]["rho_1e-4"]
    R1 = R3 = R13 = None
    if g and c1 is not None and c3 is not None and g > 0:
        R1, R3, R13 = c1 / g, c3 / g, c3 / c1 if c1 else None
    classes_ok = all(set(med[a]["classes"]) == {"TF32_TENSOROP"} for a in ("G", "C1", "C3"))
    if not classes_ok:
        branch = "INSTRUMENT_FAILED"
    elif g is not None and g < 0.005 and c1 < 0.05 and c3 < 0.05:
        branch = "NULL_AT_THIS_SCALE"
    elif R1 is None:
        branch = "INCOMPLETE"
    elif max(R1, R3, R13 or 0) < 2.0:
        branch = "BRANCH_SCALE_DEPTH_ONLY"
    elif R1 >= 5.0:
        branch = "BRANCH_OPCLASS_OWN"
    elif R3 >= 5.0:
        branch = "BRANCH_SPATIAL_PATH"
    else:
        branch = "BRANCH_INTERMEDIATE"
    summary["ratios"] = dict(R1=R1, R3=R3, R13=R13)
    summary["branch"] = branch
    summary["classes_held"] = classes_ok
    # per-draw ordering check
    order_consistent = True
    for d in range(DRAWS):
        v = {r["arm"]: r["rho_1e-4"] for r in ok if r["draw"] == d}
        if len(v) == 3:
            if not (v["C3"] >= v["C1"] and v["C1"] >= v["G"]):
                order_consistent = False
    summary["order_consistent_all_draws"] = order_consistent
    summary["wall_s"] = round(time.time() - t_start, 1)
    with open(SUM, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ("branch", "ratios", "classes_held", "ok_rows",
                                              "order_consistent_all_draws", "wall_s")}), flush=True)
    print(json.dumps(med, indent=1), flush=True)


if __name__ == "__main__":
    main()
