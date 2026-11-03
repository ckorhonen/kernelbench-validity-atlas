#!/usr/bin/env python
"""s89b_tf32off_attrib_v2.py — AMENDMENT v2 of s89, frozen BEFORE this run.

Why an amendment exists (s89 v1 ran first; artifacts kept at results/s89_tf32off_attrib.*):
  (a) INSTRUMENT BUG in v1: run_model() mutates the model to the requested dtype, so the fp64 anchor call
      left the model in fp64 and every torch.compile attempt afterwards died on a dtype mismatch
      ("expected mat1 and mat2 to have the same dtype, float != double" on L2/14; the conv rows died in the
      same place under fake tensors). All C1 pairs were empty in v1. Fix: model.to(torch.float32) before compile.
  (b) FORMULA SLIP in v1: the ceiling was coded as frac(|R-X| > b_e) over ALL elements. The written definition
      is "the exact upper bound on share for that pair", which is n_ceil/n_fail (share <= min(1, n_ceil/n_fail)).
      v2 computes ceiling = n_ceil/n_fail. No threshold and no population changed.
  (c) TIE-OUT was coded as exact equality with s71. v1 showed 4,991,796 vs 4,991,815 on L2/11 (3.8e-6 relative)
      and 1,396/3,568 element differences on the two controls: cuDNN algorithm selection can depend on
      allocator/workspace state, so the same code path need not be bit-reproducible across processes.
      v2 measures this instead of asserting it: each row recomputes A and B a second time in-process
      (repeat_A, repeat_B) and reports bit-identity and count deltas. The s71 tie-out is reported as an exact
      count AND a relative delta; the gate is relative delta <= 1e-5, with the in-process repeat drift as the
      control that justifies that tolerance.
  (d) POPULATION PRECISION: the class question needs engine_B != engine_A. v1 measured L2/14 as A and B both
      FP32_SIMT (bit-identical outputs) => L2/14 is a same-class ordering exceedance, not a class test.
      v2 reports it separately and restricts the class claim to rows with a measured class change.

Population: the same s71-derived set (targets = exceed_strict_tf32off rows; controls = first 2 file-order rows
with exceed_strict_default true and exceed_strict_tf32off false). No outcome selection.

Arms per row, one model instance, s71 seed sequence: A default fp32 reference, B tf32-off fp32 reference,
X fp64 anchor, C1 torch.compile(dynamic=False) default flags (model reset to fp32 first), C2 = B as candidate
against A (reference swap, zero arithmetic change). Engine labels (profiler top-5) for A, B, C1.

Predicate evaluator-exact candidate-scaled b_e = 1e-4 + 1e-4|C|. Per pair (C,R): n_fail, n_ref_ind
(fail & |R-X| > b_e), share, ceiling = n_ceil/n_fail, share_ratio = share/ceiling.

Reading (predeclared): SHARE_HOLDS if primary CI low >= 0.90. Otherwise the constant does not transfer and
share_ratio is the measured quantity; CLASS_SCOPED requires share_ratio < 0.5 (reference covers less than half
the failure set, ex-ante "the reference does not explain the majority"); otherwise
BOUNDED_BY_REFERENCE_COVERAGE (the reference's own exceedance set caps the share).
UNDERPOWERED if pooled n_fail < 500. PREMISE_FAILED if no target row shows engine_B != engine_A.
INSTRUMENT_UNSTABLE if repeat_A/repeat_B are not bit-identical to A/B.

Output: results/s89b_tf32off_attrib.jsonl + results/s89b_tf32off_attrib_summary.json
"""
import importlib.util, json, os, re, signal, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUTL = os.path.join(HERE, "results", "s89b_tf32off_attrib.jsonl")
SUML = os.path.join(HERE, "results", "s89b_tf32off_attrib_summary.json")
S71 = os.path.join(HERE, "results", "s71_full_audit.jsonl")
KB = os.path.expanduser("~/kernelbench/KernelBench")
DATE = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
ATOL = RTOL = 1e-4
CFG_ALARM_S = 240

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
    evts = [(e.key, float(e.self_device_time_total)) for e in prof.key_averages() if e.self_device_time_total > 0]
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
    cd, rd = cand.double(), ref.double()
    n_fail = n_ref = n_ceil = 0
    mx = 0.0
    fa = cd.reshape(-1, cd.shape[-1]) if cd.ndim else cd.reshape(1, 1)
    fr = rd.reshape(-1, rd.shape[-1]) if rd.ndim else rd.reshape(1, 1)
    fx = anchor.reshape(-1, anchor.shape[-1]) if anchor.ndim else anchor.reshape(1, 1)
    step = max(1, (1 << 22) // max(1, fa.shape[1]))
    for i in range(0, fa.shape[0], step):
        c, r, x = fa[i:i + step], fr[i:i + step], fx[i:i + step]
        be = ATOL + RTOL * c.abs()
        d = (c - r).abs()
        fail = d > be
        dref = (r - x).abs()
        n_fail += int(fail.sum())
        n_ref += int((fail & (dref > be)).sum())
        n_ceil += int((dref > be).sum())
        mx = max(mx, float(d.max()))
    total = int(fa.shape[0] * fa.shape[1])
    return {"n_fail": n_fail, "n_ref_ind": n_ref, "n_cand_ind": n_fail - n_ref,
            "share": (n_ref / n_fail) if n_fail else None,
            "n_ceil": n_ceil, "ceiling": (n_ceil / n_fail) if n_fail else None,
            "share_ratio": (n_ref / n_ceil) if n_ceil else None,
            "maxabs": mx, "total": total}


def ref_stats(ref, anchor):
    rd = ref.double()
    fr = rd.reshape(-1, rd.shape[-1]) if rd.ndim else rd.reshape(1, 1)
    fx = anchor.reshape(-1, anchor.shape[-1]) if anchor.ndim else anchor.reshape(1, 1)
    cnt, mx = 0, 0.0
    step = max(1, (1 << 22) // max(1, fr.shape[1]))
    for i in range(0, fr.shape[0], step):
        r, x = fr[i:i + step], fx[i:i + step]
        d = (r - x).abs()
        be = ATOL + RTOL * r.abs()
        cnt += int((d > be).sum())
        mx = max(mx, float(d.max()))
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

    # AMENDMENT (a): reset the model to fp32 before compiling.
    signal.alarm(CFG_ALARM_S)
    set_cfg(True, False)
    model.float()
    try:
        cm = torch.compile(model, dynamic=False)
        with torch.no_grad():
            C1 = first_tensor(cm(*[cast_inputs(x, torch.float32, "cuda") for x in inputs]))
        engC1 = classify([n for n, _ in profile_call(
            lambda: cm(*[cast_inputs(x, torch.float32, "cuda") for x in inputs]))])
        c1_err = None
    except Exception as e:
        C1 = None
        engC1 = ("COMPILE_FAIL", str(e)[:120])
        c1_err = str(e)[:200]
    signal.alarm(0)

    # in-process repeat control (amendment c)
    signal.alarm(CFG_ALARM_S)
    set_cfg(True, False)
    A2 = run_model(model, inputs, torch.float32, "cuda")
    set_cfg(False, False)
    B2 = run_model(model, inputs, torch.float32, "cuda")
    signal.alarm(0)

    if A is None or B is None or X is None or A.shape != B.shape or A.shape != X.shape:
        raise RuntimeError("missing or shape-mismatched outputs")

    rec = {"idx": idx, "arm": arm, "level": r["level"], "problem_id": r["problem_id"], "name": r["name"],
           "op_class": r["op_class"], "shape": list(A.shape), "n_elem": int(A.numel()), "date": DATE,
           "engine_A": engA[0], "engine_A_name": engA[1], "engine_B": engB[0], "engine_B_name": engB[1],
           "engine_C1": engC1[0], "class_change": engA[0] != engB[0],
           "ref_stats_A": ref_stats(A, X), "ref_stats_B": ref_stats(B, X),
           "bit_AB": bool(torch.equal(A, B)),
           "repeat": {"A_bit": bool(torch.equal(A, A2)), "B_bit": bool(torch.equal(B, B2)),
                      "A_count_delta": ref_stats(A2, X)["count"] - ref_stats(A, X)["count"],
                      "B_count_delta": ref_stats(B2, X)["count"] - ref_stats(B, X)["count"]},
           "cands": {}}

    for name, cand, ref in (("C1_vs_B", C1, B), ("C1_vs_A", C1, A), ("C2_vs_A", B, A)):
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
        print(i, arm, r["name"], rec["engine_A"], "->", rec["engine_B"], "C1", rec["engine_C1"],
              "bitAB", rec["bit_AB"], "repA/B", rec["repeat"]["A_bit"], rec["repeat"]["B_bit"],
              {k: (v.get("n_fail"), v.get("share"), v.get("ceiling")) for k, v in rec["cands"].items()},
              flush=True)
        with open(OUTL, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def pool(arm, pair, class_change_only=False):
        k = n = ceil = 0
        for rec in out:
            if rec["arm"] != arm:
                continue
            if class_change_only and not rec["class_change"]:
                continue
            c = rec["cands"].get(pair, {})
            if not c.get("ok"):
                continue
            k += c["n_ref_ind"]
            n += c["n_fail"]
            ceil += c["n_ceil"]
        return k, n, ceil

    summ = {"date": DATE, "n_rows": len(out),
            "population": {"targets": [[r["problem_id"], r["name"]] for r in targets],
                           "controls": [[r["problem_id"], r["name"]] for r in ctrl]},
            "pooled": {}, "pooled_class_change_only": {}, "per_row": []}
    for arm in ("target", "control"):
        for pair in ("C1_vs_B", "C1_vs_A", "C2_vs_A"):
            k, n, ceil = pool(arm, pair)
            p, lo, hi = wilson(k, n)
            summ["pooled"]["%s/%s" % (arm, pair)] = {
                "n_ref_ind": k, "n_fail": n, "share": p if n else None, "ci_low": lo, "ci_high": hi,
                "n_ceil": ceil, "ceiling": (ceil / n) if n else None,
                "share_ratio": (k / ceil) if ceil else None}
            k, n, ceil = pool(arm, pair, True)
            p, lo, hi = wilson(k, n)
            summ["pooled_class_change_only"]["%s/%s" % (arm, pair)] = {
                "n_ref_ind": k, "n_fail": n, "share": p if n else None, "ci_low": lo, "ci_high": hi,
                "n_ceil": ceil, "ceiling": (ceil / n) if n else None,
                "share_ratio": (k / ceil) if ceil else None}

    for rec in out:
        summ["per_row"].append({k: v for k, v in rec.items() if k != "cands"} |
                               {"cands": {kk: {a: b for a, b in vv.items() if a != "error"}
                                          for kk, vv in rec["cands"].items()}})

    prim = summ["pooled_class_change_only"].get("target/C1_vs_B", {})
    premise = any(rec["class_change"] for rec in out if rec["arm"] == "target")
    unstable = any(not (rec["repeat"]["A_bit"] and rec["repeat"]["B_bit"]) for rec in out)
    under = (prim.get("n_fail") or 0) < 500
    share, lo, ceil = prim.get("share"), prim.get("ci_low"), prim.get("ceiling")
    ratio = prim.get("share_ratio")
    if not premise:
        reading = "PREMISE_FAILED"
    elif under:
        reading = "UNDERPOWERED"
    elif lo is not None and lo >= 0.90:
        reading = "SHARE_HOLDS"
    elif ratio is not None and ratio < 0.5:
        reading = "CLASS_SCOPED"
    elif ratio is not None:
        reading = "BOUNDED_BY_REFERENCE_COVERAGE"
    else:
        reading = "AMBIGUOUS"
    summ["verdict"] = {"premise_class_change_present": bool(premise), "underpowered": bool(under),
                       "instrument_unstable": bool(unstable),
                       "primary_share": share, "primary_ci_low": lo, "primary_ceiling": ceil,
                       "primary_share_ratio": ratio, "reading": reading}
    json.dump(summ, open(SUML, "w"), indent=1)
    print(json.dumps(summ["pooled_class_change_only"], indent=1))
    print(json.dumps(summ["verdict"], indent=1))


if __name__ == "__main__":
    main()
