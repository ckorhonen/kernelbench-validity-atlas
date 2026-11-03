#!/usr/bin/env python
"""s89_tf32off_exceed_attrib.py — attribution where the tf32-off REFERENCE itself exceeds the gate.

PREREG (frozen before the completed run; do not edit after the first result).

Population (measured conjunction from results/s71_full_audit.jsonl, declared, no outcome selection):
  TARGETS = the 2 rows of the 192-row s71 population with exceed_strict_tf32off == true
            (L2/11 convT composite: B_vs_anchor 4,991,815/18,939,904 at 1e-4, rho 58.66;
             L2/14 gemm composite: 4/1024 at 1e-4, rho 0.01386).
  CONTROLS = the first 2 rows in s71 file order with exceed_strict_default == true and
             exceed_strict_tf32off == false (mechanical file-order rule, no outcome selection).

Question. s84/s87 measured a reference-induced share of 0.9985 (TF32_TENSOROP class, reference itself
inside budget). s88 falsified the winograd population because a non-TF32 reference there was reference-CLEAN
(no failing elements to split). This population is the only one in my records where a NON-TF32 reference
(tf32-off) both exceeds and has failing elements. Does the share constant transfer, or is it bounded by the
reference's own exceedance fraction?

Model instance per row: s71's exact seed sequence (manual_seed(0); load_task; get_inputs(); manual_seed(0);
Model(*init_args)) so that A and B reproduce s71 bit-for-bit and the tie-out gate is exact.

Arms, identical inputs, one model instance:
  A   default fp32 reference      (cudnn tf32 on, matmul tf32 off)
  B   tf32-off fp32 reference     (cudnn off, matmul off) — the exceeding non-TF32 realization
  X   fp64 anchor                 (same weights, fp64)
  C1  torch.compile(dynamic=False) under default flags — benchmark-style candidate
  C2  B used as a candidate against reference A — pure reference-swap arm, zero arithmetic change

Predicate evaluator-exact, candidate-scaled: b_e = 1e-4 + 1e-4*|C| (eval.py:804 direction).
  frac_over_ref(R) = frac(|R-X| > 1e-4 + 1e-4*|R|); rho_max = max(|R-X|/(1e-4+1e-4*|X|)).
  For (C,R): n_fail = frac(|C-R| > b_e); n_ref_ind = frac(fail & (|R-X| > b_e)); share = n_ref_ind/n_fail.
  ceiling(C,R) = frac(|R-X| > b_e) — the exact upper bound on share for that pair.
  Engine labels (profiler top-5) for A, B, X, C1; bit-identity counts C==R.

Primary pair: C1 vs B (a default-class candidate against the exceeding non-TF32 reference).
Secondary: C2 vs A (reference swap, no candidate arithmetic change). Descriptive: C1 vs A.

Kill / gate clauses (predeclared):
  K3 tie-out: B_vs_X element count must equal s71's B_vs_anchor ew_1e-4 count EXACTLY on both targets
     (4,991,815 and 4). Any mismatch => INSTRUMENT_FAILED, no attribution claim quoted.
  K4 premise: if B's engine label on either target is TF32_TENSOROP, the tf32-off reference did not change
     class => PREMISE_FAILED, no class claim; report the labels.
  K2 power: pooled n_fail < 500 on the primary pair => UNDERPOWERED.
  K1 transfer: pooled share CI low >= 0.90 => SHARE_HOLDS_IN_NON_TF32_CLASS.
     share < 0.90 and share <= ceiling_pooled + 0.02 => SHARE_BOUNDED_BY_REFERENCE_EXCEEDANCE
     (structural boundary: the constant's domain is reference-clean populations, not a class property).
     share < 0.90 and share < ceiling_pooled - 0.10 => CLASS_SCOPED (per-class shares required).

Output: results/s89_tf32off_attrib.jsonl + results/s89_tf32off_attrib_summary.json
Run on spark in ~/kb-lab (rsync from lab/). CPU-side analysis only after the GPU pass; no timing claims.
"""
import importlib.util, json, os, re, signal, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUTL = os.path.join(HERE, "results", "s89_tf32off_attrib.jsonl")
SUML = os.path.join(HERE, "results", "s89_tf32off_attrib_summary.json")
S71 = os.path.join(HERE, "results", "s71_full_audit.jsonl")
KB = os.path.expanduser("~/kernelbench/KernelBench")
DATE = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
ATOL = RTOL = 1e-4
CFG_ALARM_S = 180

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


def load_task(path):
    name = "kb_" + os.path.basename(path).replace(".py", "").replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cast_inputs(x, dtype, device):
    if isinstance(x, torch.Tensor):
        if x.is_floating_point():
            return x.to(device=device, dtype=dtype)
        return x.to(device=device)
    if isinstance(x, (list, tuple)):
        return type(x)(cast_inputs(v, dtype, device) for v in x)
    if isinstance(x, dict):
        return {k: cast_inputs(v, dtype, device) for k, v in x.items()}
    return x


def first_tensor(o):
    if isinstance(o, torch.Tensor):
        return o
    if isinstance(o, (list, tuple)):
        for v in o:
            t = first_tensor(v)
            if t is not None:
                return t
    return None


def run_model(model, inputs, dtype, device):
    model = model.to(device=device, dtype=dtype)
    model.eval()
    with torch.no_grad():
        out = model(*[cast_inputs(x, dtype, device) for x in inputs])
    return first_tensor(out)


def decompose(cand, ref, anchor):
    """Chunked elementwise attribution. cand/ref fp32, anchor fp64, same shape."""
    cd = cand.double()
    rd = ref.double()
    n_fail = n_ref = n_ceil = 0
    mx = 0.0
    fa = cd.reshape(-1, cd.shape[-1]) if cd.ndim else cd.reshape(1, 1)
    fr = rd.reshape(-1, rd.shape[-1]) if rd.ndim else rd.reshape(1, 1)
    fx = anchor.reshape(-1, anchor.shape[-1]) if anchor.ndim else anchor.reshape(1, 1)
    step = max(1, (1 << 22) // max(1, fa.shape[1]))
    for i in range(0, fa.shape[0], step):
        c = fa[i:i + step]
        r = fr[i:i + step]
        x = fx[i:i + step]
        be = ATOL + RTOL * c.abs()
        d = (c - r).abs()
        fail = d > be
        dref = (r - x).abs()
        n_fail += int(fail.sum())
        n_ref += int((fail & (dref > be)).sum())
        n_ceil += int((dref > be).sum())
        m = float(d.max())
        if m > mx:
            mx = m
    total = int(fa.shape[0] * fa.shape[1])
    return {"n_fail": n_fail, "n_ref_ind": n_ref, "n_cand_ind": n_fail - n_ref,
            "share": (n_ref / n_fail) if n_fail else None,
            "ceiling": n_ceil / total, "maxabs": mx, "total": total}


def ref_stats(ref, anchor):
    rd = ref.double()
    fr = rd.reshape(-1, rd.shape[-1]) if rd.ndim else rd.reshape(1, 1)
    fx = anchor.reshape(-1, anchor.shape[-1]) if anchor.ndim else anchor.reshape(1, 1)
    cnt = 0
    mx = 0.0
    step = max(1, (1 << 22) // max(1, fr.shape[1]))
    for i in range(0, fr.shape[0], step):
        r = fr[i:i + step]
        x = fx[i:i + step]
        d = (r - x).abs()
        be = ATOL + RTOL * r.abs()
        cnt += int((d > be).sum())
        m = float(d.max())
        if m > mx:
            mx = m
    total = int(fr.shape[0] * fr.shape[1])
    return {"count": cnt, "total": total, "frac_over": cnt / total, "maxabs": mx}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return (p, max(0.0, c - hw), min(1.0, c + hw))


def select_rows():
    rows = [json.loads(l) for l in open(S71)]
    targets = [r for r in rows if r.get("exceed_strict_tf32off")]
    ctrl = []
    for r in rows:
        if r.get("exceed_strict_default") and not r.get("exceed_strict_tf32off"):
            ctrl.append(r)
        if len(ctrl) == 2:
            break
    return targets, ctrl


def run_row(r, idx, arm):
    path = os.path.join(KB, "KernelBench", "level%d" % r["level"], r["name"])
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    mod = load_task(path)
    init_args = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
    inputs = mod.get_inputs()
    torch.manual_seed(0)
    model = mod.Model(*init_args)

    signal.alarm(CFG_ALARM_S)
    set_cfg(True, False)
    A = run_model(model, inputs, torch.float32, "cuda")
    signal.alarm(0)
    engA = classify([n for n, _ in profile_call(lambda: run_model(model, inputs, torch.float32, "cuda"))])

    signal.alarm(CFG_ALARM_S)
    set_cfg(False, False)
    B = run_model(model, inputs, torch.float32, "cuda")
    signal.alarm(0)
    engB = classify([n for n, _ in profile_call(lambda: run_model(model, inputs, torch.float32, "cuda"))])

    signal.alarm(CFG_ALARM_S)
    set_cfg(False, False)
    X = run_model(model, inputs, torch.float64, "cuda")
    signal.alarm(0)

    signal.alarm(CFG_ALARM_S)
    set_cfg(True, False)
    try:
        cm = torch.compile(model, dynamic=False)
        C1 = cm(*[cast_inputs(x, torch.float32, "cuda") for x in inputs])
        C1 = first_tensor(C1)
        engC1 = classify([n for n, _ in profile_call(
            lambda: cm(*[cast_inputs(x, torch.float32, "cuda") for x in inputs]))])
        c1_err = None
    except Exception as e:
        C1 = None
        engC1 = ("COMPILE_FAIL", str(e)[:120])
        c1_err = str(e)[:200]
    signal.alarm(0)

    if A is None or B is None or X is None or A.shape != B.shape or A.shape != X.shape:
        raise RuntimeError("missing or shape-mismatched outputs")

    rec = {"idx": idx, "arm": arm, "level": r["level"], "problem_id": r["problem_id"], "name": r["name"],
           "op_class": r["op_class"], "shape": list(A.shape), "n_elem": int(A.numel()), "date": DATE,
           "engine_A": engA[0], "engine_A_name": engA[1], "engine_B": engB[0], "engine_B_name": engB[1],
           "engine_C1": engC1[0], "ref_stats_A": ref_stats(A, X), "ref_stats_B": ref_stats(B, X),
           "bit_AB": bool(torch.equal(A, B)), "cands": {}}

    pairs = [("C1_vs_B", C1, B), ("C1_vs_A", C1, A), ("C2_vs_A", B, A)]
    for name, cand, ref in pairs:
        if cand is None:
            rec["cands"][name] = {"ok": False, "error": c1_err}
            continue
        d = decompose(cand, ref, X)
        d["ok"] = True
        d["bit_identical"] = bool(torch.equal(cand, ref))
        rec["cands"][name] = d
    return rec


def main():
    targets, ctrl = select_rows()
    print("targets", [(r["problem_id"], r["name"]) for r in targets], flush=True)
    print("controls", [(r["problem_id"], r["name"]) for r in ctrl], flush=True)
    os.makedirs(os.path.dirname(OUTL), exist_ok=True)
    out = []
    for i, (r, arm) in enumerate([(x, "target") for x in targets] + [(x, "control") for x in ctrl]):
        t = time.time()
        rec = run_row(r, i, arm)
        rec["wall_s"] = round(time.time() - t, 2)
        out.append(rec)
        print(i, arm, r["name"], rec["engine_A"], rec["engine_B"], "bitAB", rec["bit_AB"],
              "B_over %.5f" % rec["ref_stats_B"]["frac_over"],
              {k: (v.get("n_fail"), v.get("share"), v.get("ceiling")) for k, v in rec["cands"].items()},
              flush=True)
        with open(OUTL, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def pool(arm, pair):
        k = n = ceil = tot = 0
        for rec in out:
            if rec["arm"] != arm:
                continue
            c = rec["cands"].get(pair, {})
            if not c.get("ok"):
                continue
            k += c["n_ref_ind"]
            n += c["n_fail"]
            ceil += int(round(c["ceiling"] * c["total"]))
            tot += c["total"]
        return k, n, (ceil / tot if tot else None)

    summ = {"date": DATE, "n_rows": len(out),
            "population": {"targets": [[r["problem_id"], r["name"]] for r in targets],
                           "controls": [[r["problem_id"], r["name"]] for r in ctrl]},
            "pooled": {}, "per_row": [], "tie_out": {}}
    for arm in ("target", "control"):
        for pair in ("C1_vs_B", "C1_vs_A", "C2_vs_A"):
            k, n, ceil = pool(arm, pair)
            p, lo, hi = wilson(k, n)
            summ["pooled"]["%s/%s" % (arm, pair)] = {"n_ref_ind": k, "n_fail": n,
                                                     "share": p if n else None, "ci_low": lo,
                                                     "ci_high": hi, "ceiling": ceil}
    # exact tie-out against s71
    for rec in out:
        s71 = next(json.loads(l) for l in open(S71)
                   if json.loads(l)["problem_id"] == rec["problem_id"]
                   and json.loads(l)["level"] == rec["level"])
        summ["tie_out"]["%d" % rec["problem_id"]] = {
            "s71_B_count": s71["B_vs_anchor"]["ew_1e-4"]["count"],
            "s89_B_count": rec["ref_stats_B"]["count"],
            "match": int(s71["B_vs_anchor"]["ew_1e-4"]["count"]) == int(rec["ref_stats_B"]["count"]),
            "s71_A_count": s71["A_vs_anchor"]["ew_1e-4"]["count"],
            "s89_A_count": rec["ref_stats_A"]["count"],
            "a_match": int(s71["A_vs_anchor"]["ew_1e-4"]["count"]) == int(rec["ref_stats_A"]["count"])}
    for rec in out:
        summ["per_row"].append({k: v for k, v in rec.items() if k != "cands"})

    prim = summ["pooled"].get("target/C1_vs_B", {})
    k3 = not all(v["match"] and v["a_match"] for v in summ["tie_out"].values())
    k4 = any(rec["engine_B"] == "TF32_TENSOROP" for rec in out if rec["arm"] == "target")
    k2 = (prim.get("n_fail") or 0) < 500
    share = prim.get("share")
    ceil = prim.get("ceiling")
    k1 = (prim.get("ci_low") is not None and prim["ci_low"] >= 0.90)
    if k3:
        reading = "INSTRUMENT_FAILED"
    elif k4:
        reading = "PREMISE_FAILED"
    elif k2:
        reading = "UNDERPOWERED"
    elif k1:
        reading = "SHARE_HOLDS_IN_NON_TF32_CLASS"
    elif share is not None and ceil is not None and share <= ceil + 0.02:
        reading = "SHARE_BOUNDED_BY_REFERENCE_EXCEEDANCE"
    elif share is not None and ceil is not None and share < ceil - 0.10:
        reading = "CLASS_SCOPED"
    else:
        reading = "AMBIGUOUS"
    summ["verdict"] = {"K3_tie_out_failed": bool(k3), "K4_premise_failed": bool(k4),
                       "K2_underpowered": bool(k2), "K1_transfer": bool(k1),
                       "primary_share": share, "primary_ci_low": prim.get("ci_low"),
                       "primary_ceiling": ceil, "reading": reading}
    json.dump(summ, open(SUML, "w"), indent=1)
    print(json.dumps(summ["pooled"], indent=1))
    print(json.dumps(summ["verdict"], indent=1))
    print(json.dumps(summ["tie_out"], indent=1))


if __name__ == "__main__":
    main()
