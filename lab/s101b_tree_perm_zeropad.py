#!/usr/bin/env python
"""s101b_tree_perm_conv_zeropad.py — post-hoc declared extension of s101 (declared AFTER s101's numbers,
before any row of this run; no s101 threshold changed).

WHY. s101 measured the declared chunked permutation on T1 and returned BRANCH_COMPOSE with c = 1.0000 across
u 0.33..1360 (1e-4) and 0.0033..13.6 (1e-2), but its own K_ENGINE control fired: the chunked arms dispatch
implicit_convolve_sgemm (FP32_SIMT) instead of cutlass_tensorop_s1688fprop TF32 (class_A TF32_TENSOROP on
13/13 points, class_P FP32_SIMT at g=1 and g=64). The chunked arms are therefore ~100-1000x closer to the
fp64 anchor than the reference, so |P-A| ~ |A-X| and c ~ 1 is the signature of a CLASS CHANGE, not of a
tree permutation. This run holds the engine class fixed while permuting the tree: each input-channel chunk is
zero-padded back to the full C_in=64 so the same TF32 tensorop engine is dispatched (zeros contribute exact
zero), and the partial outputs are accumulated in a declared sequential tree. The cost ratio is then
attributable to the arithmetic tree at fixed engine class.

DESIGN (frozen before any row of this run):
  Tuple: T1 from s101/s91 (conv 3x3 s1 p1, B=2, C=64->64, 151x151, seed 1085).
  Scale: g in {0.0625, 0.25, 1.0, 4.0, 16.0, 64.0} => u_1e-4 0.66..680, u_1e-2 0.0066..6.8, bracketing
  rho=1 at both gates.
  Arms: A reference (cudnn allow_tf32 True, run twice for repeat control); X fp64 anchor; P4z/P16z/P64z =
  4/16/64 contiguous input-channel chunks, each zero-padded to C_in=64, sequential fp32 accumulation.
  Controls: K1 A repeat bit-identical; K2 g=1 A rho_1e-2 must reproduce s91's 0.1395 within 2%; K3 engine
  class of P16z must be TF32_TENSOROP at g=1 and g=64 (else CLASS_NOT_HELD and no tree claim); K4 a
  dispatch-boundary diagnostic profiles single convs at C_in in {8,16,32,64} to document where cudnn leaves
  the tensorop engine (the s101 confound's mechanism).
  Branch: cost ratio c = rho_P_ref / rho_A_x at qualifying points (rho_A_x >= 0.05). BRANCH_COMPOSE_FIXED_CLASS
  iff c spread <= 1.10 at each gate and pooled with >= 1 decade of u; else BRANCH_SURFACE_FIXED_CLASS.
  Descriptive: c_z vs s101's class-changing c (the confound's size), frac_vs_ref vs frac_A (does a
  fixed-class tree permutation inherit the reference's exceedance set?), timing.
"""
import json, os, re, signal, time, hashlib
import torch

LAB = os.path.expanduser("~/kb-lab")
OUT = os.path.join(LAB, "results", "s101b_tree_perm_zeropad.jsonl")
SUMOUT = os.path.join(LAB, "results", "s101b_summary.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

CFG_ALARM_S = 300
GLOBAL_DEADLINE_S = 900
CHUNK_ELEMS = 1 << 22
STRICT, SHIPPED, REF_REL, NOISE_FLOOR = 1e-4, 1e-2, 2.2e-4, 0.05

ARGS = dict(n=2, cin=64, h=151, w=151, cout=64, k=3, s=1, p=1)
SEED = 1085
GRID = [0.0625, 0.25, 1.0, 4.0, 16.0, 64.0]
DEPTHS = [4, 16, 64]
PROBE_CINS = [8, 16, 32, 64]

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
    """Declared tree: k contiguous input-channel chunks, each zero-padded to full C_in (engine class held),
    partials accumulated sequentially in fp32."""
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


def main():
    torch.cuda.init()
    open(OUT, "w").close()
    sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    print("script sha", sha[:12], torch.__version__, torch.cuda.get_device_name(0), flush=True)
    rows, t_start = [], time.time()

    # K4 dispatch-boundary diagnostic (single convs at reduced C_in, same spatial shape)
    probe = []
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(SEED)
    for cin in PROBE_CINS:
        x = torch.randn(ARGS["n"], cin, ARGS["h"], ARGS["w"], device="cuda")
        w = torch.randn(ARGS["cout"], cin, ARGS["k"], ARGS["k"], device="cuda")
        ev = profile_call(lambda: conv_fwd(x, w))
        cls, top1 = classify([n for n, _ in ev])
        probe.append(dict(cin=cin, class_=cls, top1=top1))
        print("probe cin", cin, cls, top1[:80], flush=True)
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
    summary = dict(script_sha=sha, extension_of="s101", n_attempted=len(rows), n_ok=len(ok),
                   grid=GRID, depths=DEPTHS, dispatch_probe=probe, noise_floor=NOISE_FLOOR)

    g1 = [r for r in ok if r["g"] == 1.0]
    tie = dict(status="missing")
    if g1:
        got = g1[0]["A_1e-2"]["rho"]
        rel = abs(got - 0.1395) / 0.1395
        tie = dict(rho_got=got, s91_norm_rho=0.1395, rel=rel, pass_=bool(rel <= 0.02))
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

    verdicts = {}
    for gate in ("1e-4", "1e-2"):
        for name in ("P4z", "P16z", "P64z"):
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
                                               invariant=bool(min(cs) > 0 and max(cs) / min(cs) <= 1.10))
    summary["branch_eval"] = verdicts

    if not instr_ok:
        summary["verdict"] = "INSTRUMENT_FAILED"
    else:
        per_arm = {}
        for name in ("P4z", "P16z", "P64z"):
            per_arm[name] = all(verdicts.get(gate + "_" + name, {}).get("invariant") for gate in ("1e-4", "1e-2"))
        summary["per_arm_invariant"] = per_arm
        summary["verdict"] = "BRANCH_COMPOSE_FIXED_CLASS" if all(per_arm.values()) else "BRANCH_SURFACE_FIXED_CLASS"

    with open(SUMOUT, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ("verdict", "tieout", "instrument_ok", "class_held",
                                             "class_points", "per_arm_invariant")}, indent=1)[:2000], flush=True)
    print(json.dumps(summary["branch_eval"], indent=1)[:2500], flush=True)


if __name__ == "__main__":
    main()
