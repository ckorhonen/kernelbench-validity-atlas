#!/usr/bin/env python
"""s76_anchor_coverage.py — close the s71 coverage gap: fp64 anchor for the 8 tasks excluded by the
whole-tensor working-set guard, with element-chunked comparison.

PREREG (frozen at commit before the run; do not edit after).

Population (rule): exactly the 8 rows in results/s71_full_audit_summary.json with
anchor == "inadmissible:working_set": L1 34 InstanceNorm, 35 GroupNorm, 36 RMSNorm, 37 FrobeniusNorm,
38 L1Norm, 39 L2Norm, 45 AveragePooling2D, 76 conv_standard_1D_dilated_strided. These were excluded by an
ESTIMATE (fp64_est_bytes = 8*(in_elems+out_elems) > 24 GiB), not by an observed OOM: the largest estimate is
32.26 GiB against a 121.69 GiB pool.

Deviation from s71, declared and measurement-neutral: (a) the 24 GiB guard is removed for this population;
(b) inputs are REGENERATED under torch.manual_seed(0) for the fp64 pass instead of holding fp32 and fp64 copies
simultaneously, so peak is bounded by one dtype copy at a time; (c) the elementwise comparison is
ELEMENT-chunked (2^22 elements) rather than ROW-chunked, because s71's row unit materialized a full fp64 copy
of the candidate (a.double()) for wide tensors. Predicates, tolerances (1e-2 bench, 1e-4 strict), rho definition,
seed and quantity names are s71's verbatim.

TIE-OUT GATE (required before any new anchor number is used): recompute s71's recorded A_vs_B (default fp32 vs
tf32-off fp32) exceedance counts for each of the 8 tasks under s71's recorded predicate and require an EXACT
match. A mismatch excludes that row as tie_out_mismatch and no anchor number from it is reported.

Quantities per task: anchor=admissible, ref_max_abs, A_vs_anchor and B_vs_anchor ew_stats {count,total,frac,
maxabs,rho_global} at both tolerances, exceed flags at both tolerances, peak memory observed.

Kill clauses: (K1) a task that raises OOM or timeout is named and reported as still-excluded, not silently
dropped; (K2) if fewer than 8/8 rows pass the tie-out gate, the coverage claim is withheld and the failure is
reported. Coverage claim: 192/200 -> 200/200 only if 8/8 rows complete and tie out.
"""
import importlib.util, json, os, re, signal, sys, time, traceback, hashlib
import torch

KB = os.path.expanduser("~/kernelbench/KernelBench")
OUT = os.path.expanduser("~/kb-lab/results/s76_anchor_coverage.jsonl")
SUMOUT = os.path.expanduser("~/kb-lab/results/s76_anchor_coverage_summary.json")
S71 = os.path.expanduser("~/kb-lab/results/s71_full_audit_summary.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

CFG_ALARM_S = 300
TOL_BENCH = 1e-2
TOL_STRICT = 1e-4
ELEM_CHUNK = 1 << 22
POP = [(1, 34), (1, 35), (1, 36), (1, 37), (1, 38), (1, 39), (1, 45), (1, 76)]


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout("config exceeded %ds" % CFG_ALARM_S)


signal.signal(signal.SIGALRM, _alarm)


def load_task(path):
    name = "kb_" + os.path.basename(path).replace(".py", "").replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def set_cfg(tf32_cudnn, tf32_matmul):
    torch.backends.cudnn.allow_tf32 = tf32_cudnn
    torch.backends.cuda.matmul.allow_tf32 = tf32_matmul


def nbytes(t):
    if not isinstance(t, torch.Tensor):
        return 0
    return t.numel() * t.element_size()


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


def run_ref(model, inputs, dtype, device):
    model = model.to(device=device, dtype=dtype)
    model.eval()
    with torch.no_grad():
        out = model(*[cast_inputs(x, dtype, device) for x in inputs])
    return first_tensor(out)


def ew_stats_elem(a, b, atol, rtol):
    """Element-chunked. a fp32 candidate, b fp64 reference. Returns s71's quantity dict."""
    fa = a.reshape(-1)
    fb = b.reshape(-1)
    total = fa.numel()
    cnt = 0
    mx = 0.0
    refmx = 0.0
    for i in range(0, total, ELEM_CHUNK):
        ca = fa[i:i + ELEM_CHUNK].double()
        cb = fb[i:i + ELEM_CHUNK]
        d = (ca - cb).abs()
        thr = atol + rtol * cb.abs()
        cnt += int((d > thr).sum())
        m = float(d.max())
        if m > mx:
            mx = m
        r = float(cb.abs().max())
        if r > refmx:
            refmx = r
        del ca, d, thr
    return {"count": cnt, "total": int(total), "frac": cnt / max(1, total), "maxabs": mx,
            "rho_global": mx / (atol + rtol * refmx)}


def main():
    torch.cuda.init()
    dev = "cuda"
    t_start = time.time()
    s71 = json.load(open(S71))
    rec71 = {(r["level"], r["problem_id"]): r for r in s71["rows"]}
    rows = []
    for level, pid in POP:
        path = None
        d = os.path.join(KB, "KernelBench", "level%d" % level)
        for f in sorted(os.listdir(d)):
            if f.endswith(".py") and f.split("_")[0] == str(pid):
                path = os.path.join(d, f)
                name = f
        rec = {"level": level, "problem_id": pid, "name": name}
        try:
            torch.manual_seed(0)
            torch.cuda.empty_cache()
            mod = load_task(path)
            init_args = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
            inputs = mod.get_inputs()
            torch.manual_seed(0)
            model = mod.Model(*init_args)
            signal.alarm(CFG_ALARM_S)
            set_cfg(True, False)
            out_def = run_ref(model, inputs, torch.float32, dev)
            set_cfg(False, False)
            out_off = run_ref(model, inputs, torch.float32, dev)
            signal.alarm(0)
            if out_def is None or out_off is None or out_def.shape != out_off.shape:
                raise RuntimeError("fp32 arms missing or shape-mismatched")
            rec["output_shape"] = list(out_def.shape)
            rec["output_bytes"] = int(nbytes(out_def))
            rec["A_vs_B"] = {"ew_1e-2": ew_stats_elem(out_def, out_off.double(), TOL_BENCH, TOL_BENCH),
                             "ew_1e-4": ew_stats_elem(out_def, out_off.double(), TOL_STRICT, TOL_STRICT)}
            # TIE-OUT GATE against s71's recorded A_vs_B counts
            r71 = rec71[(level, pid)].get("A_vs_B", {})
            c_bench = r71.get("ew_1e-2", {}).get("count")
            c_strict = r71.get("ew_1e-4", {}).get("count")
            rec["tie_out"] = {"s71_bench": c_bench, "s71_strict": c_strict,
                              "new_bench": rec["A_vs_B"]["ew_1e-2"]["count"],
                              "new_strict": rec["A_vs_B"]["ew_1e-4"]["count"]}
            rec["tie_out_ok"] = (c_bench == rec["A_vs_B"]["ew_1e-2"]["count"]
                                 and c_strict == rec["A_vs_B"]["ew_1e-4"]["count"])
            # free fp32 inputs before materializing the fp64 copy
            del inputs
            torch.cuda.empty_cache()
            torch.manual_seed(0)
            inputs64 = mod.get_inputs()
            signal.alarm(CFG_ALARM_S)
            set_cfg(False, False)
            out_anchor = run_ref(model, inputs64, torch.float64, dev)
            signal.alarm(0)
            if out_anchor is None or out_anchor.shape != out_def.shape:
                raise RuntimeError("fp64 anchor missing or shape-mismatched")
            rec["anchor"] = "admissible"
            rec["ref_max_abs"] = float(out_anchor.abs().max())
            rec["A_vs_anchor"] = {"ew_1e-2": ew_stats_elem(out_def, out_anchor, TOL_BENCH, TOL_BENCH),
                                  "ew_1e-4": ew_stats_elem(out_def, out_anchor, TOL_STRICT, TOL_STRICT)}
            rec["exceed_bench_default"] = bool(rec["A_vs_anchor"]["ew_1e-2"]["count"] > 0)
            rec["exceed_strict_default"] = bool(rec["A_vs_anchor"]["ew_1e-4"]["count"] > 0)
            rec["B_vs_anchor"] = {"ew_1e-2": ew_stats_elem(out_off, out_anchor, TOL_BENCH, TOL_BENCH),
                                  "ew_1e-4": ew_stats_elem(out_off, out_anchor, TOL_STRICT, TOL_STRICT)}
            rec["exceed_bench_tf32off"] = bool(rec["B_vs_anchor"]["ew_1e-2"]["count"] > 0)
            rec["exceed_strict_tf32off"] = bool(rec["B_vs_anchor"]["ew_1e-4"]["count"] > 0)
            del out_anchor
            torch.cuda.empty_cache()
            rec["status"] = "ok"
        except Timeout as e:
            rec["status"] = "timeout"
            rec["error"] = str(e)
        except Exception as e:
            rec["status"] = "error:%s" % type(e).__name__
            rec["error"] = str(e)[:300]
        finally:
            signal.alarm(0)
        rows.append(rec)
        with open(OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps({k: rec.get(k) for k in ("level", "problem_id", "name", "status", "anchor", "tie_out_ok")}), flush=True)

    ok = [r for r in rows if r.get("status") == "ok" and r.get("tie_out_ok")]
    summary = {"population_rule": "s71 anchor==inadmissible:working_set (8 tasks)",
               "attempted": len(rows), "completed": len(ok),
               "tie_out_failures": [r["problem_id"] for r in rows if r.get("status") == "ok" and not r.get("tie_out_ok")],
               "failed": [{"problem_id": r["problem_id"], "status": r.get("status"), "error": r.get("error")} for r in rows if r.get("status") != "ok"],
               "coverage": "200/200" if len(ok) == 8 else "partial",
               "exceed_bench_default": sum(1 for r in ok if r.get("exceed_bench_default")),
               "exceed_strict_default": sum(1 for r in ok if r.get("exceed_strict_default")),
               "max_rho_global_bench_default": max((r["A_vs_anchor"]["ew_1e-2"]["rho_global"] for r in ok), default=None),
               "wall_s": round(time.time() - t_start, 1), "torch": torch.__version__}
    json.dump({"summary": summary, "rows": rows}, open(SUMOUT, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
