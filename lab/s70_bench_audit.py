#!/usr/bin/env python
"""s70_bench_audit.py — benchmark-wide reference-error audit, pilot.

Prereg: lab/s70_prereg.md (frozen before run). Population rule: level1/level2, problem_id % 5 == 1.
Configs: A default, B tf32-off, anchor fp64. rho at benchmark tolerance (1e-2) and strict (1e-4).
No thresholds beyond the prereg's K1-K3; all labels computed from recorded numbers.
"""
import importlib.util, json, os, signal, sys, time, traceback, hashlib
import torch

KB = os.path.expanduser("~/kernelbench/KernelBench")
OUT = os.path.expanduser("~/kb-lab/results/s70b_bench_audit.json")
OUTDIR = os.path.dirname(OUT)
os.makedirs(OUTDIR, exist_ok=True)

TOL_BENCH = 1e-2
TOL_STRICT = 1e-4
MEM_CAP_BYTES = 24 * 1024**3
ALARM_S = 120


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout("task exceeded %ds" % ALARM_S)


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


def run_ref(model, inputs, dtype, device):
    model = model.to(device=device, dtype=dtype)
    model.eval()
    with torch.no_grad():
        out = model(*[x.to(device=device, dtype=dtype) if isinstance(x, torch.Tensor) else x for x in inputs])
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


def rho_terms(diff_max, ref_max, tol):
    return float(diff_max / (tol + tol * ref_max))


def maxdiff_chunked(a, b, chunk_rows=1 << 20):
    """max|a-b| computed in row chunks to bound temporaries."""
    a64 = a.double()
    if a64.ndim == 0:
        return float((a64 - b).abs().max())
    flat_a = a64.reshape(-1, a64.shape[-1])
    flat_b = b.reshape(-1, b.shape[-1])
    m = 0.0
    for i in range(0, flat_a.shape[0], chunk_rows):
        d = (flat_a[i:i + chunk_rows] - flat_b[i:i + chunk_rows]).abs().max().item()
        if d > m:
            m = d
    return float(m)


def main():
    torch.cuda.init()
    dev = "cuda"
    rows = []
    excluded = []
    t_start = time.time()
    for level in (1, 2):
        d = os.path.join(KB, "KernelBench", "level%d" % level)
        files = sorted(f for f in os.listdir(d) if f.endswith(".py"))
        for f in files:
            try:
                pid = int(f.split("_")[0])
            except ValueError:
                continue
            if pid % 5 != 1:
                continue
            path = os.path.join(d, f)
            rec = {"level": level, "problem_id": pid, "name": f}
            signal.alarm(ALARM_S)
            try:
                torch.manual_seed(0)
                torch.cuda.empty_cache()
                mod = load_task(path)
                init_args = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
                inputs = mod.get_inputs()
                # size accounting
                in_bytes = sum(nbytes(x) for x in inputs if isinstance(x, torch.Tensor))
                # ONE model instance per task: weights must be identical across configs
                # (defect found in the first run: re-instantiating per config compared different
                #  random weights on parameterized tasks, which faked conv exceedance).
                torch.manual_seed(0)
                model = mod.Model(*init_args)
                set_cfg(True, False)
                t0 = time.time()
                out_def = run_ref(model, inputs, torch.float32, dev)
                t_def = time.time() - t0
                out_bytes = nbytes(out_def)
                rec.update({"input_bytes": in_bytes, "output_bytes": out_bytes,
                            "output_shape": list(out_def.shape),
                            "input_shapes": [list(x.shape) for x in inputs if isinstance(x, torch.Tensor)],
                            "input_dtypes": [str(x.dtype) for x in inputs if isinstance(x, torch.Tensor)],
                            "t_default_s": round(t_def, 3)})
                set_cfg(False, False)
                t0 = time.time()
                out_off = run_ref(model, inputs, torch.float32, dev)
                t_off = time.time() - t0
                rec["t_tf32off_s"] = round(t_off, 3)
                # A vs B (no anchor needed)
                dab = maxdiff_chunked(out_def, out_off)
                rec["A_vs_B_maxabs"] = dab
                rec["A_vs_B_rho_bench"] = rho_terms(dab, float(out_off.abs().max()), TOL_BENCH)
                rec["A_vs_B_bit_identical"] = bool(dab == 0.0)
                # anchor admissibility by estimated fp64 working set
                est = 8 * (in_bytes // 4 + out_bytes // 4)
                rec["fp64_est_bytes"] = int(est)
                if est > MEM_CAP_BYTES:
                    rec["anchor"] = "inadmissible:working_set"
                    excluded.append({"level": level, "problem_id": pid, "name": f,
                                     "reason": "fp64 estimated working set %.1f GB > 24 GB" % (est / 1024**3)})
                else:
                    t0 = time.time()
                    set_cfg(False, False)
                    out_anchor = run_ref(model, inputs, torch.float64, dev)
                    t_anchor = time.time() - t0
                    rec["t_fp64_s"] = round(t_anchor, 3)
                    ref_max = float(out_anchor.abs().max())
                    d_def = maxdiff_chunked(out_def, out_anchor)
                    d_off = maxdiff_chunked(out_off, out_anchor)
                    rec["anchor"] = "admissible"
                    rec["ref_max_abs"] = ref_max
                    rec["rho_bench_default"] = rho_terms(d_def, ref_max, TOL_BENCH)
                    rec["rho_bench_tf32off"] = rho_terms(d_off, ref_max, TOL_BENCH)
                    rec["rho_strict_default"] = rho_terms(d_def, ref_max, TOL_STRICT)
                    rec["rho_strict_tf32off"] = rho_terms(d_off, ref_max, TOL_STRICT)
                    rec["exceed_bench_default"] = bool(rec["rho_bench_default"] > 1)
                    rec["exceed_bench_tf32off"] = bool(rec["rho_bench_tf32off"] > 1)
                    rec["exceed_strict_default"] = bool(rec["rho_strict_default"] > 1)
                    rec["exceed_strict_tf32off"] = bool(rec["rho_strict_tf32off"] > 1)
                    model.float()  # fp32->fp64->fp32 round trip is exact; restore for any later use
                    del out_anchor
                del out_def, out_off
                torch.cuda.empty_cache()
            except Timeout as e:
                rec["status"] = "timeout"
                excluded.append({"level": level, "problem_id": pid, "name": f, "reason": str(e)})
            except Exception as e:
                rec["status"] = "error:%s: %s" % (type(e).__name__, str(e)[:160])
                excluded.append({"level": level, "problem_id": pid, "name": f,
                                 "reason": rec["status"]})
            finally:
                signal.alarm(0)
            rows.append(rec)
            print(json.dumps({k: rec.get(k) for k in ("level", "problem_id", "name", "anchor",
                                                      "rho_bench_default", "rho_bench_tf32off",
                                                      "status")}), flush=True)

    adm = [r for r in rows if r.get("anchor") == "admissible"]
    def rate(key, pop):
        n = sum(1 for r in pop if r.get(key))
        return {"count": n, "denominator": len(pop)}
    summary = {
        "population_rule": "level1+level2, problem_id % 5 == 1",
        "tasks_attempted": len(rows),
        "tasks_anchor_admissible": len(adm),
        "excluded": excluded,
        "exceedance": {
            "bench_default": rate("exceed_bench_default", adm),
            "bench_tf32off": rate("exceed_bench_tf32off", adm),
            "strict_default": rate("exceed_strict_default", adm),
            "strict_tf32off": rate("exceed_strict_tf32off", adm),
            "A_vs_B_any_delta": {"count": sum(1 for r in rows if r.get("A_vs_B_maxabs", 0) > 0),
                                 "denominator": len([r for r in rows if "A_vs_B_maxabs" in r])},
        },
        "max_rho_bench_default": max((r.get("rho_bench_default", 0) for r in adm), default=None),
        "max_rho_bench_tf32off": max((r.get("rho_bench_tf32off", 0) for r in adm), default=None),
        "wall_s": round(time.time() - t_start, 1),
        "torch": torch.__version__,
        "prereg_sha256": hashlib.sha256(open(os.path.expanduser("~/kb-lab/s70_prereg.md"), "rb").read()).hexdigest(),
    }
    json.dump({"summary": summary, "rows": rows}, open(OUT, "w"), indent=1)
    print(json.dumps(summary, indent=1)[:2500])


if __name__ == "__main__":
    main()
