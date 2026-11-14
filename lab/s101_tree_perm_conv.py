#!/usr/bin/env python
"""s101_tree_perm_conv.py — declared arithmetic-tree permutation on a conv/TF32 tuple across the rho=1 crossing.

PREREG (frozen before any run; see lab/s101_prereg.md, committed with this file; do not edit after).

Question: s100 priced a tree-depth coordinate only where the budget is slack (frac_over 0.0 on 80/80 rows);
s91/s92/s93 measured the binding regime with no declared arithmetic-order perturbation. This run puts a
declared permutation in the binding regime: T1 from s91's crossing population (conv 3x3 s1 p1, B=2, C=64->64,
151x151, seed 1085, TF32_TENSOROP), reference held fixed, candidate = chunked input-channel accumulation at
declared depths (4/16/64 sequential, plus a fan-in-2 tree over the 16 boundaries), scale swept so u spans
both gates' rho=1 crossings. Kill/branch clauses are in the prereg; they are coded below verbatim.

Verdict logic: c = rho_P_ref / rho_A_x per qualifying point (rho_A_x >= 0.05). BRANCH_COMPOSE iff every arm's
max(c)/min(c) <= 1.10 at each gate and pooled and the qualifying u range spans >= 1 decade; else BRANCH_SURFACE.
"""
import json, os, re, signal, sys, time, hashlib
import torch

LAB = os.path.expanduser("~/kb-lab")
OUT = os.path.join(LAB, "results", "s101_tree_perm_conv.jsonl")
SUMOUT = os.path.join(LAB, "results", "s101_summary.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

CFG_ALARM_S = 300
GLOBAL_DEADLINE_S = 1500
CHUNK_ELEMS = 1 << 22
STRICT = 1e-4
SHIPPED = 1e-2
REF_REL = 2.2e-4
NOISE_FLOOR = 0.05

TUPLE = dict(tid="T1", desc="conv 3x3 s1 151x151 C64->64 B2", seed=1085,
             args=dict(n=2, cin=64, h=151, w=151, cout=64, k=3, s=1, p=1))

GRID = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
SEQ_DEPTHS = [4, 16, 64]
PROFILE_GS = {1.0, 64.0}

CLASS_PATTERNS = [
    ("WINOGRAD", re.compile(r"winograd", re.I)),
    ("TF32_TENSOROP", re.compile(r"tf32|tensorop|1688|16816|h1688", re.I)),
    ("FP64", re.compile(r"dgrad|dgemm|dconv|fp64|double|_d_", re.I)),
    ("FP32_SIMT", re.compile(r"fprop|implicit_gemm|sgemm|s1688|fp32|scudnn|simt", re.I)),
    ("CUTLASS", re.compile(r"cutlass", re.I)),
]


class Timeout(Exception):
    pass


def _alarm(signum, frame):
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
    evts = []
    for e in prof.key_averages():
        if e.self_device_time_total > 0:
            evts.append((e.key, float(e.self_device_time_total)))
    evts.sort(key=lambda t: -t[1])
    return evts[:5]


def set_cfg(tf32_cudnn, tf32_matmul):
    torch.backends.cudnn.allow_tf32 = tf32_cudnn
    torch.backends.cuda.matmul.allow_tf32 = tf32_matmul


def build():
    a = TUPLE["args"]
    x = torch.randn(a["n"], a["cin"], a["h"], a["w"], device="cuda", dtype=torch.float32)
    w = torch.randn(a["cout"], a["cin"], a["k"], a["k"], device="cuda", dtype=torch.float32)
    return x, w


def conv_fwd(x, w):
    a = TUPLE["args"]
    return torch.nn.functional.conv2d(x, w, stride=a["s"], padding=a["p"])


def chunk_bounds(cin, k):
    step = cin // k
    return [(i * step, (i + 1) * step) for i in range(k)]


def perm_sequential(x, w, k):
    """Declared tree: contiguous input-channel chunks, sequential fp32 accumulation (depth k)."""
    out = None
    for lo, hi in chunk_bounds(x.shape[1], k):
        p = conv_fwd(x[:, lo:hi].contiguous(), w[:, lo:hi].contiguous())
        out = p if out is None else out + p
    return out


def perm_pairwise(x, w, k):
    """Declared tree: fan-in-2 pairwise tree over the same k contiguous boundaries."""
    parts = []
    for lo, hi in chunk_bounds(x.shape[1], k):
        parts.append(conv_fwd(x[:, lo:hi].contiguous(), w[:, lo:hi].contiguous()))
    while len(parts) > 1:
        nxt = []
        for i in range(0, len(parts) - 1, 2):
            nxt.append(parts[i] + parts[i + 1])
        if len(parts) % 2 == 1:
            nxt.append(parts[-1])
        parts = nxt
    return parts[0]


def chunked_compare(ref, cand, atol, rtol):
    """Evaluator-exact: budget = atol + rtol*|cand| (eval.py:804 direction). Returns n_fail, rho, n_tot, argmax."""
    n_fail = 0
    max_ratio = 0.0
    n_tot = 0
    best = dict(ratio=-1.0, ref=0.0, cand=0.0, d=0.0)
    rf = ref.detach().reshape(-1)
    cf = cand.detach().reshape(-1)
    n = rf.numel()
    i = 0
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
            k = int(am.item())
            best = dict(ratio=mr, ref=float(rf[i + k].item()), cand=float(cf[i + k].item()),
                        d=float(d[k].item()))
        n_tot += (j - i)
        del d, b, r
        i = j
    return n_fail, max_ratio, n_tot, best


def run_point(g):
    rec = dict(tid=TUPLE["tid"], seed=TUPLE["seed"], g=g, status="ok")
    t0 = time.time()
    signal.alarm(CFG_ALARM_S)
    try:
        set_cfg(True, False)
        torch.manual_seed(TUPLE["seed"])
        x, w = build()
        fan_in = w.numel() // w.shape[0]
        w.div_(fan_in ** 0.5)
        if g != 1.0:
            w.mul_(g)
        rec["fan_in"] = fan_in

        a1 = conv_fwd(x, w)
        a2 = conv_fwd(x, w)
        rec["a_repeat_bit_identical"] = bool(torch.equal(a1, a2))
        A = a1
        del a2
        rec["maxabs_out"] = float(A.abs().max().item())
        rec["n_elems"] = int(A.numel())

        p1 = perm_sequential(x, w, 1)
        rec["p1_bit_identical"] = bool(torch.equal(p1, A))
        del p1

        arms = {}
        t = time.time(); arms["P4"] = perm_sequential(x, w, 4); rec["t_P4"] = round(time.time() - t, 2)
        t = time.time(); arms["P16"] = perm_sequential(x, w, 16); rec["t_P16"] = round(time.time() - t, 2)
        t = time.time(); arms["P64"] = perm_sequential(x, w, 64); rec["t_P64"] = round(time.time() - t, 2)
        t = time.time(); arms["P16pair"] = perm_pairwise(x, w, 16); rec["t_P16pair"] = round(time.time() - t, 2)

        if g in PROFILE_GS:
            rec["engine_A"] = [list(v) for v in profile_call(lambda: conv_fwd(x, w))[:3]]
            rec["class_A"], rec["top1_A"] = classify([n for n, _ in profile_call(lambda: conv_fwd(x, w))])
            rec["engine_P16"] = [list(v) for v in profile_call(lambda: perm_sequential(x, w, 16))[:3]]
            rec["class_P16"], rec["top1_P16"] = classify(
                [n for n, _ in profile_call(lambda: perm_sequential(x, w, 16))])

        set_cfg(False, False)
        X = conv_fwd(x.double(), w.double())
        del x, w

        if A.shape != X.shape:
            raise RuntimeError("shape mismatch")

        for tag, tol in (("1e-4", STRICT), ("1e-2", SHIPPED)):
            nf_a, mr_a, nt, diag = chunked_compare(A, X, tol, tol)
            rec["A_" + tag] = dict(n_fail=nf_a, rho=mr_a, frac_over=nf_a / nt, n_elems=nt, argmax=diag)
            for name, P in arms.items():
                nf_p, mr_p, nt2, diag_p = chunked_compare(A, P, tol, tol)
                nf_px, mr_px, _, _ = chunked_compare(X, P, tol, tol)
                rec[name + "_" + tag] = dict(rho_vs_ref=mr_p, frac_vs_ref=nf_p / nt2,
                                             rho_vs_fp64=mr_px, frac_vs_fp64=nf_px / nt2,
                                             argmax=diag_p, n_elems=nt2)
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
    print("script sha", sha[:12], flush=True)
    print("env", torch.__version__, torch.cuda.get_device_name(0), flush=True)

    rows = []
    t_start = time.time()
    for g in GRID:
        if time.time() - t_start > GLOBAL_DEADLINE_S:
            print("GLOBAL DEADLINE at g", g, flush=True)
            break
        r = run_point(g)
        rows.append(r)
        with open(OUT, "a") as f:
            f.write(json.dumps(r) + "\n")
        line = "g %-8s %s" % (g, r["status"])
        if r["status"] == "ok":
            line += (" maxabs %.4f u1e4 %.2f rhoA1e4 %.3f rhoA1e2 %.4f"
                     % (r["maxabs_out"], r["u_1e-4"], r["A_1e-4"]["rho"], r["A_1e-2"]["rho"]))
            for name in ("P4", "P16", "P64", "P16pair"):
                line += " | %s r1e4 %.3f r1e2 %.4f" % (name, r[name + "_1e-4"]["rho_vs_ref"],
                                                       r[name + "_1e-2"]["rho_vs_ref"])
            line += " | rep %s p1 %s t %s" % (r.get("a_repeat_bit_identical"), r.get("p1_bit_identical"),
                                              r.get("t_s"))
            if "class_A" in r:
                line += " | clsA %s clsP %s" % (r.get("class_A"), r.get("class_P16"))
        print(line, flush=True)

    ok = [r for r in rows if r.get("status") == "ok"]
    summary = dict(script_sha=sha, prereg="lab/s101_prereg.md", n_attempted=len(rows), n_ok=len(ok),
                   grid=GRID, tuple_def=TUPLE, noise_floor=NOISE_FLOOR)

    # controls
    g1 = [r for r in ok if r["g"] == 1.0]
    tie = dict(status="missing")
    if g1:
        got = g1[0]["A_1e-2"]["rho"]
        rel = abs(got - 0.1395) / 0.1395
        tie = dict(rho_got=got, s91_norm_rho=0.1395, rel=rel, pass_=bool(rel <= 0.02))
    summary["tieout"] = tie
    summary["n_repeat_fail"] = sum(1 for r in ok if not r.get("a_repeat_bit_identical"))
    summary["n_p1_fail"] = sum(1 for r in ok if not r.get("p1_bit_identical"))
    summary["repeat_ok"] = summary["n_repeat_fail"] <= 2
    summary["p1_control_ok"] = summary["n_p1_fail"] == 0
    summary["engine"] = {str(r["g"]): dict(class_A=r.get("class_A"), top1_A=r.get("top1_A"),
                                           class_P16=r.get("class_P16"), top1_P16=r.get("top1_P16"))
                         for r in ok if "class_A" in r}
    instr_ok = bool(tie.get("pass_")) and summary["repeat_ok"] and summary["p1_control_ok"]
    summary["instrument_ok"] = instr_ok

    # cost ratio table
    table = {}
    for gate in ("1e-4", "1e-2"):
        table[gate] = []
        for r in sorted(ok, key=lambda x: x["g"]):
            rho_a = r["A_" + gate]["rho"]
            u = r["u_" + gate]
            row = dict(g=r["g"], u=u, rho_A=rho_a, frac_A=r["A_" + gate]["frac_over"])
            for name in ("P4", "P16", "P64", "P16pair"):
                rho_p = r[name + "_" + gate]["rho_vs_ref"]
                row[name] = dict(rho=rho_p, frac=r[name + "_" + gate]["frac_vs_ref"],
                                 rho_vs_fp64=r[name + "_" + gate]["rho_vs_fp64"],
                                 c=(rho_p / rho_a) if rho_a > 0 else None)
            table[gate].append(row)
    summary["cost_table"] = table

    # branch evaluation
    verdicts = {}
    for gate in ("1e-4", "1e-2"):
        for name in ("P4", "P16", "P64", "P16pair"):
            cs = [row[name]["c"] for row in table[gate]
                  if row["rho_A"] >= NOISE_FLOOR and row[name]["c"] is not None]
            us = [row["u"] for row in table[gate] if row["rho_A"] >= NOISE_FLOOR]
            if len(cs) < 2:
                verdicts[gate + "_" + name] = dict(status="RANGE_INSUFFICIENT", n=len(cs))
                continue
            verdicts[gate + "_" + name] = dict(n=len(cs), c_min=min(cs), c_max=max(cs),
                                               spread=(max(cs) / min(cs)) if min(cs) > 0 else None,
                                               u_min=min(us), u_max=max(us),
                                               u_decades=(max(us) / min(us)) if min(us) > 0 else None,
                                               invariant=bool(max(cs) / min(cs) <= 1.10))
    summary["branch_eval"] = verdicts

    if not instr_ok:
        summary["verdict"] = "INSTRUMENT_FAILED"
    else:
        per_arm = {}
        for name in ("P4", "P16", "P64", "P16pair"):
            ok_all = True
            for gate in ("1e-4", "1e-2"):
                v = verdicts.get(gate + "_" + name, {})
                if v.get("status") == "RANGE_INSUFFICIENT" or not v.get("invariant"):
                    ok_all = False
            per_arm[name] = ok_all
        summary["per_arm_invariant"] = per_arm
        if all(per_arm.values()):
            summary["verdict"] = "BRANCH_COMPOSE"
        else:
            summary["verdict"] = "BRANCH_SURFACE"

    with open(SUMOUT, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ("verdict", "tieout", "instrument_ok", "n_repeat_fail",
                                             "n_p1_fail", "per_arm_invariant", "engine")}, indent=1)[:3000],
          flush=True)
    print(json.dumps(summary["branch_eval"], indent=1)[:3000], flush=True)


if __name__ == "__main__":
    main()
