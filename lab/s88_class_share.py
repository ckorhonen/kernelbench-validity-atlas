#!/usr/bin/env python
"""s88_class_share.py — class-conditional attribution on profiled-WINOGRAD rows.

PREREG (frozen before the completed run; do not edit after).

Population rule (a measured conjunction, not an outcome): the 19 rows of results/s63_engine_predicate.jsonl
whose default-flags reference call was labelled WINOGRAD by s63's classifier (profiler top-5 kernel names).
Controls: the first 3 TF32_TENSOROP rows in s63 file order (declared, no outcome selection).

This is a CONSTRUCTED replication at the same (N,H,W,C,CO) tuples: a direct nn.Conv2d(C,CO,3,stride=1,
padding=1) with seeded weights and inputs, not a re-run of the task models. The instrument gate is per row: the
constructed conv's A call must label WINOGRAD; rows labelling otherwise are excluded and counted. K3 narrows
the population and says so if fewer than 10 of 19 survive.

Question: is the reference-induced share of failing elements class-scoped? s84/s87 measured 0.9985 inside
TF32_TENSOROP (153.4M failing elements, 17/17 configurations TF32-labelled). The 19 winograd rows are the only
population where a NON-TF32 reference both exceeds the gate and has failing elements to split (s87c's
constructed extent sweep produced only reference-clean winograd cells).

Arms per row, one model instance, identical inputs:
  A   default fp32 reference (cudnn tf32 on, matmul tf32 off)
  X   fp64 anchor from the same fp32 weights
  C1  torch.compile(dynamic=False) fp32 default flags   (class-sharing candidate)
  C2  fp32, cudnn tf32 off                             (class-changing candidate)

Predicate evaluator-exact, candidate-scaled: b_e = 1e-4 + 1e-4*|C| (eval.py:804 direction).
  m1 frac_over_ref = frac(|A-X| > 1e-4 + 1e-4*|A|); rho_max = max(|A-X|/(1e-4+1e-4*|X|)).
  m2 failing elements per candidate: |C-A| > b_e; each failing element is REFERENCE-INDUCED if
     |A-X| > b_e, else CANDIDATE-INDUCED. Per-row and pooled shares.
  m3 Wilson 95% CI on the pooled share; per-row engine labels for A, X, C1, C2; bit-identity counts (C==A).

Kill clauses (predeclared):
  K1  pooled share CI lower bound < 0.90 => the class claim becomes per-class shares; the 0.9985 constant is
      class-scoped and the column must carry per-class entries.
  K2  power: pooled failing elements < 500 => UNDERPOWERED; report the CI, make no class claim.
  K3  instrument: fewer than 10 of the 19 rows survive the WINOGRAD label gate => population narrowed, stated.
  K4  control: the 3 TF32 controls' pooled share < 0.99 => harness disagrees with s84; no winograd number is
      quoted (control gate fails).

Output: results/s88_class_share.jsonl + results/s88_class_share_summary.json
Run on spark: python3 s88_class_share.py   (CPU-side analysis only after the GPU pass; no timing claims)
"""
import json, os, re, sys, time
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUTL = os.path.join(HERE, "results", "s88_class_share.jsonl")
SUML = os.path.join(HERE, "results", "s88_class_share_summary.json")
S63 = os.path.join(HERE, "results", "s63_engine_predicate.jsonl")
DATE = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
ATOL = RTOL = 1e-4
MIN_FAIL = 500
MIN_ROWS = 10
CONTROL_BAR = 0.99
SHARE_BAR = 0.90

CLASS_PATTERNS = [
    ("WINOGRAD", re.compile(r"winograd", re.I)),
    ("TF32_TENSOROP", re.compile(r"tf32|tensorop|1688|16816|h1688", re.I)),
    ("FP64", re.compile(r"dgrad|dgemm|dconv|fp64|double|_d_", re.I)),
    ("FP32_SIMT", re.compile(r"fprop|implicit_gemm|sgemm|s1688|fp32|scudnn|simt", re.I)),
    ("CUTLASS", re.compile(r"cutlass", re.I)),
]


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


def set_flags(tf32_cudnn):
    torch.backends.cudnn.allow_tf32 = tf32_cudnn
    torch.backends.cuda.matmul.allow_tf32 = False


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return (p, max(0.0, c - hw), min(1.0, c + hw))


def load_rows():
    rows = [json.loads(l) for l in open(S63)]
    wino = [r for r in rows if r["refs"]["def"]["engine_class"] == "WINOGRAD"]
    tf32 = [r for r in rows if r["refs"]["def"]["engine_class"] == "TF32_TENSOROP"][:3]
    return wino, tf32


def run_row(r, idx, arm):
    N, H, W, C, CO = r["N"], r["H"], r["W"], r["C"], r["CO"]
    torch.manual_seed(8800 + idx)
    conv = torch.nn.Conv2d(C, CO, 3, stride=1, padding=1, bias=True).cuda()
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float32)
    w64 = conv.weight.detach().double()
    b64 = conv.bias.detach().double()
    x64 = x.double()

    set_flags(True)
    with torch.no_grad():
        A = F.conv2d(x, conv.weight, conv.bias, stride=1, padding=1)
    engA = classify([n for n, _ in profile_call(
        lambda: F.conv2d(x, conv.weight, conv.bias, stride=1, padding=1))])

    set_flags(True)
    with torch.no_grad():
        X = F.conv2d(x64, w64, b64, stride=1, padding=1)
    engX = classify([n for n, _ in profile_call(
        lambda: F.conv2d(x64, w64, b64, stride=1, padding=1))])

    set_flags(False)
    with torch.no_grad():
        C2 = F.conv2d(x, conv.weight, conv.bias, stride=1, padding=1)
    engC2 = classify([n for n, _ in profile_call(
        lambda: F.conv2d(x, conv.weight, conv.bias, stride=1, padding=1))])

    set_flags(True)
    try:
        cm = torch.compile(conv, dynamic=False)
        with torch.no_grad():
            C1 = cm(x)
        engC1 = classify([n for n, _ in profile_call(lambda: cm(x))])
        c1_err = None
    except Exception as e:  # declared: a compile failure removes C1 only
        C1 = None
        engC1 = ("COMPILE_FAIL", str(e)[:120])
        c1_err = str(e)[:200]

    Ad = A.double()
    dref = (Ad - X).abs()
    bA = ATOL + RTOL * Ad.abs()
    frac_over = float((dref > bA).sum().item()) / dref.numel()
    rho = float((dref / (ATOL + RTOL * X.abs())).max().item())

    out = {"idx": idx, "arm": arm, "tag": r["tag"], "N": N, "H": H, "W": W, "C": C, "CO": CO,
           "n_elem": int(dref.numel()), "date": DATE,
           "engine_A": engA[0], "engine_A_name": engA[1], "engine_X": engX[0], "engine_X_name": engX[1],
           "engine_C2": engC2[0], "engine_C2_name": engC2[1], "engine_C1": engC1[0],
           "frac_over_ref": frac_over, "rho_max": rho,
           "cands": {}}
    for name, cand in (("C1", C1), ("C2", C2)):
        if cand is None:
            out["cands"][name] = {"ok": False, "error": c1_err}
            continue
        Cd = cand.double()
        be = ATOL + RTOL * Cd.abs()
        fail = (Cd - Ad).abs() > be
        nf = int(fail.sum().item())
        ref_ind = int((fail & (dref > be)).sum().item())
        bit = bool(torch.equal(cand, A))
        out["cands"][name] = {"ok": True, "n_fail": nf, "n_ref_ind": ref_ind,
                              "n_cand_ind": nf - ref_ind,
                              "share": (ref_ind / nf) if nf else None,
                              "bit_identical": bit,
                              "engine": engC2[0] if name == "C2" else engC1[0],
                              "max_ratio_ref": float((dref / be).max().item())}
    return out


def main():
    wino, tf32 = load_rows()
    rows = [(r, "wino") for r in wino] + [(r, "tf32_control") for r in tf32]
    os.makedirs(os.path.dirname(OUTL), exist_ok=True)
    out = []
    for i, (r, arm) in enumerate(rows):
        t = time.time()
        rec = run_row(r, i, arm)
        rec["wall_s"] = round(time.time() - t, 2)
        out.append(rec)
        print(i, arm, r["tag"], "%dx%d C%d" % (r["H"], r["W"], r["C"]), rec["engine_A"],
              "frac_over %.4f" % rec["frac_over_ref"], "rho %.3f" % rec["rho_max"],
              {k: (v.get("n_fail"), v.get("share")) for k, v in rec["cands"].items()}, flush=True)
        with open(OUTL, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def pool(arm, cand):
        k = n = 0
        for rec in out:
            if rec["arm"] != arm:
                continue
            c = rec["cands"].get(cand, {})
            if c.get("ok"):
                k += c["n_ref_ind"]
                n += c["n_fail"]
        return k, n

    summ = {"date": DATE, "n_rows": len(out), "population": {"wino": len(wino), "tf32_control": len(tf32)},
            "label_gate": {"wino_labelled_rows": sum(1 for r in out if r["arm"] == "wino" and r["engine_A"] == "WINOGRAD"),
                           "tf32_control_labelled": sum(1 for r in out if r["arm"] == "tf32_control" and r["engine_A"] == "TF32_TENSOROP")},
            "pooled": {}, "per_row": []}
    for arm in ("wino", "tf32_control"):
        for cand in ("C1", "C2"):
            k, n = pool(arm, cand)
            p, lo, hi = wilson(k, n)
            summ["pooled"]["%s/%s" % (arm, cand)] = {"n_ref_ind": k, "n_fail": n, "share": p if n else None,
                                                     "ci_low": lo, "ci_high": hi}
    for rec in out:
        summ["per_row"].append({"idx": rec["idx"], "arm": rec["arm"], "tag": rec["tag"],
                                "shape": [rec["N"], rec["H"], rec["W"], rec["C"], rec["CO"]],
                                "engine_A": rec["engine_A"], "engine_X": rec["engine_X"],
                                "engine_C2": rec["engine_C2"], "frac_over_ref": rec["frac_over_ref"],
                                "rho_max": rec["rho_max"],
                                "cands": {k: {kk: vv for kk, vv in v.items() if kk != "error"}
                                          for k, v in rec["cands"].items()}})

    wf = summ["pooled"].get("wino/C2", {})
    cf = summ["pooled"].get("tf32_control/C2", {})
    wf1 = summ["pooled"].get("wino/C1", {})
    k1 = (wf.get("ci_low") is not None and wf["ci_low"] < SHARE_BAR)
    k2 = (wf.get("n_fail") or 0) < MIN_FAIL
    k3 = summ["label_gate"]["wino_labelled_rows"] < MIN_ROWS
    k4 = (cf.get("share") is not None and cf["share"] < CONTROL_BAR)
    summ["verdict"] = {"K1_class_scoped": bool(k1), "K2_underpowered": bool(k2),
                       "K3_population_narrowed": bool(k3), "K4_control_gate_failed": bool(k4),
                       "wino_share_C2": wf.get("share"), "wino_share_C1": wf1.get("share"),
                       "control_share_C2": cf.get("share"),
                       "reading": ("CONTROL_GATE_FAILED" if k4 else
                                   "UNDERPOWERED" if k2 else
                                   "POPULATION_NARROWED" if k3 else
                                   "CLASS_SCOPED" if k1 else "SHARE_HOLDS_IN_WINOGRAD_CLASS")}
    json.dump(summ, open(SUML, "w"), indent=1)
    print(json.dumps(summ["pooled"], indent=1))
    print(json.dumps(summ["verdict"], indent=1))


if __name__ == "__main__":
    main()
