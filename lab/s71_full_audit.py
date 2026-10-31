#!/usr/bin/env python
"""s71_full_audit.py — benchmark-wide reference-error audit, FULL population (level1+level2, 200 tasks).

Prereg: lab/s71_prereg.md (frozen before run, sha recorded in the output). Primary label is the elementwise
torch.allclose predicate at atol=rtol=1e-2 against an fp64 anchor; secondary 1e-4, A-vs-B, and global-max rho.
One model instance per task (s70's re-instantiation defect fixed). Integer/bool inputs are NOT cast (only float
tensors are widened), so index tensors survive the fp64 anchor.
"""
import importlib.util, json, os, re, signal, sys, time, traceback, hashlib
import torch

KB = os.path.expanduser("~/kernelbench/KernelBench")
OUT = os.path.expanduser("~/kb-lab/results/s71_full_audit.jsonl")
SUMOUT = os.path.expanduser("~/kb-lab/results/s71_full_audit_summary.json")
OUTDIR = os.path.dirname(OUT)
os.makedirs(OUTDIR, exist_ok=True)

MEM_CAP_BYTES = 24 * 1024**3
CFG_ALARM_S = 90
GLOBAL_DEADLINE_S = 1500
TOL_BENCH = 1e-2
TOL_STRICT = 1e-4


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
    """Widen only floating-point tensors; leave integer/bool tensors (indices) untouched."""
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


def ew_stats(a, b, atol, rtol, chunk_rows=1 << 20):
    """Elementwise |a-b| > atol + rtol*|b| count against fp64 b. a fp32 tensor, b fp64 tensor, same shape."""
    a64 = a.double()
    if a64.ndim == 0:
        d = (a64 - b).abs()
        thr = atol + rtol * b.abs()
        return {"count": int((d > thr).sum()), "total": 1, "frac": float((d > thr).float()),
                "maxabs": float(d), "rho_global": float(d / (atol + rtol * float(b.abs())))}
    fa = a64.reshape(-1, a64.shape[-1])
    fb = b.reshape(-1, b.shape[-1])
    cnt = 0
    mx = 0.0
    refmx = 0.0
    for i in range(0, fa.shape[0], chunk_rows):
        d = (fa[i:i + chunk_rows] - fb[i:i + chunk_rows]).abs()
        thr = atol + rtol * fb[i:i + chunk_rows].abs()
        cnt += int((d > thr).sum())
        m = float(d.max())
        if m > mx:
            mx = m
        r = float(fb[i:i + chunk_rows].abs().max())
        if r > refmx:
            refmx = r
    total = fa.shape[0] * fa.shape[1]
    return {"count": cnt, "total": int(total), "frac": cnt / total, "maxabs": mx,
            "rho_global": mx / (atol + rtol * refmx)}


def op_class(name):
    n = name.lower()
    if "convtranspose" in n or "conv_transpose" in n:
        return "conv_transpose"
    if "conv" in n:
        return "conv"
    if "matmul" in n or "gemm" in n or "bmm" in n:
        return "gemm"
    if "batchnorm" in n or "layernorm" in n or "groupnorm" in n or "instancenorm" in n or "rmsnorm" in n:
        return "norm"
    if "softmax" in n:
        return "softmax"
    if "maxpool" in n or "avgpool" in n or "pool" in n:
        return "pool"
    if "lstm" in n or "gru" in n or "rnn" in n:
        return "rnn"
    if "attention" in n:
        return "attention"
    if "sum" in n or "mean" in n or "max" in n or "min" in n or "prod" in n or "cumsum" in n:
        return "reduction"
    if any(k in n for k in ("relu", "gelu", "sigmoid", "tanh", "add", "mul", "sub", "div", "scale", "clamp")):
        return "elementwise"
    return "other"


def main():
    torch.cuda.init()
    dev = "cuda"
    prereg_path = os.path.expanduser("~/kb-lab/s71_prereg.md")
    prereg_sha = hashlib.sha256(open(prereg_path, "rb").read()).hexdigest()
    t_start = time.time()
    rows = []
    excluded = []
    truncated = False
    tasks = []
    for level in (1, 2):
        d = os.path.join(KB, "KernelBench", "level%d" % level)
        for f in sorted(os.listdir(d)):
            if not f.endswith(".py"):
                continue
            try:
                pid = int(f.split("_")[0])
            except ValueError:
                continue
            tasks.append((level, pid, f, os.path.join(d, f)))
    print("population size", len(tasks), flush=True)

    for level, pid, f, path in tasks:
        if time.time() - t_start > GLOBAL_DEADLINE_S:
            truncated = True
            excluded.append({"level": level, "problem_id": pid, "name": f,
                             "reason": "global deadline %ds reached" % GLOBAL_DEADLINE_S})
            continue
        rec = {"level": level, "problem_id": pid, "name": f, "op_class": op_class(f)}
        try:
            torch.manual_seed(0)
            torch.cuda.empty_cache()
            mod = load_task(path)
            init_args = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
            inputs = mod.get_inputs()
            in_bytes = sum(nbytes(x) for x in inputs if isinstance(x, torch.Tensor))
            torch.manual_seed(0)
            model = mod.Model(*init_args)
            nparam = sum(p.numel() for p in model.parameters())
            rec.update({"input_bytes": int(in_bytes), "n_params": int(nparam),
                        "input_shapes": [list(x.shape) for x in inputs if isinstance(x, torch.Tensor)],
                        "input_dtypes": [str(x.dtype) for x in inputs if isinstance(x, torch.Tensor)]})
            signal.alarm(CFG_ALARM_S)
            t0 = time.time()
            set_cfg(True, False)
            out_def = run_ref(model, inputs, torch.float32, dev)
            rec["t_default_s"] = round(time.time() - t0, 3)
            signal.alarm(0)
            if out_def is None:
                raise RuntimeError("no tensor output")
            out_bytes = nbytes(out_def)
            rec.update({"output_bytes": int(out_bytes), "output_shape": list(out_def.shape),
                        "output_dtype": str(out_def.dtype)})
            signal.alarm(CFG_ALARM_S)
            t0 = time.time()
            set_cfg(False, False)
            out_off = run_ref(model, inputs, torch.float32, dev)
            rec["t_tf32off_s"] = round(time.time() - t0, 3)
            signal.alarm(0)
            if out_off is None or out_off.shape != out_def.shape:
                raise RuntimeError("tf32-off output missing or shape-mismatched")
            rec["A_vs_B"] = {"ew_1e-2": ew_stats(out_def, out_off.double(), TOL_BENCH, TOL_BENCH),
                             "ew_1e-4": ew_stats(out_def, out_off.double(), TOL_STRICT, TOL_STRICT)}
            rec["A_vs_B_bit_identical"] = bool(rec["A_vs_B"]["ew_1e-2"]["maxabs"] == 0.0)

            est = 8 * (in_bytes // 4 + out_bytes // 4)
            rec["fp64_est_bytes"] = int(est)
            if est > MEM_CAP_BYTES:
                rec["anchor"] = "inadmissible:working_set"
                excluded.append({"level": level, "problem_id": pid, "name": f,
                                 "reason": "fp64 estimated working set %.2f GiB > 24 GiB" % (est / 1024**3)})
                del out_def, out_off
            else:
                signal.alarm(CFG_ALARM_S)
                t0 = time.time()
                set_cfg(False, False)
                out_anchor = run_ref(model, inputs, torch.float64, dev)
                rec["t_fp64_s"] = round(time.time() - t0, 3)
                signal.alarm(0)
                if out_anchor is None or out_anchor.shape != out_def.shape:
                    raise RuntimeError("fp64 anchor missing or shape-mismatched")
                rec["anchor"] = "admissible"
                rec["ref_max_abs"] = float(out_anchor.abs().max())
                rec["A_vs_anchor"] = {"ew_1e-2": ew_stats(out_def, out_anchor, TOL_BENCH, TOL_BENCH),
                                      "ew_1e-4": ew_stats(out_def, out_anchor, TOL_STRICT, TOL_STRICT)}
                rec["exceed_bench_default"] = bool(rec["A_vs_anchor"]["ew_1e-2"]["count"] > 0)
                rec["exceed_strict_default"] = bool(rec["A_vs_anchor"]["ew_1e-4"]["count"] > 0)
                del out_def
                rec["B_vs_anchor"] = {"ew_1e-2": ew_stats(out_off, out_anchor, TOL_BENCH, TOL_BENCH),
                                      "ew_1e-4": ew_stats(out_off, out_anchor, TOL_STRICT, TOL_STRICT)}
                rec["exceed_bench_tf32off"] = bool(rec["B_vs_anchor"]["ew_1e-2"]["count"] > 0)
                rec["exceed_strict_tf32off"] = bool(rec["B_vs_anchor"]["ew_1e-4"]["count"] > 0)
                del out_anchor, out_off
                model.float()
            torch.cuda.empty_cache()
        except Timeout as e:
            rec["status"] = "timeout"
            excluded.append({"level": level, "problem_id": pid, "name": f, "reason": str(e)})
        except Exception as e:
            rec["status"] = "error:%s: %s" % (type(e).__name__, str(e)[:200])
            excluded.append({"level": level, "problem_id": pid, "name": f, "reason": rec["status"]})
        finally:
            signal.alarm(0)
        rows.append(rec)
        with open(OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps({k: rec.get(k) for k in ("level", "problem_id", "name", "op_class", "anchor", "status",
                                                  "exceed_bench_default", "exceed_strict_default")}), flush=True)

    adm = [r for r in rows if r.get("anchor") == "admissible"]
    def rate(key, pop):
        n = sum(1 for r in pop if r.get(key))
        return {"count": n, "denominator": len(pop)}
    summary = {
        "population_rule": "all level1+level2 tasks, rev 423217d",
        "tasks_attempted": len(rows),
        "tasks_anchor_admissible": len(adm),
        "anchor_admissibility_rate": len(adm) / max(1, len(rows)),
        "excluded": excluded,
        "truncated_by_deadline": truncated,
        "exceedance_anchor": {
            "bench_default": rate("exceed_bench_default", adm),
            "bench_tf32off": rate("exceed_bench_tf32off", adm),
            "strict_default": rate("exceed_strict_default", adm),
            "strict_tf32off": rate("exceed_strict_tf32off", adm),
        },
        "exceedance_ab": {
            "bench": {"count": sum(1 for r in rows if r.get("A_vs_B", {}).get("ew_1e-2", {}).get("count", 0) > 0),
                      "denominator": len([r for r in rows if "A_vs_B" in r])},
            "strict": {"count": sum(1 for r in rows if r.get("A_vs_B", {}).get("ew_1e-4", {}).get("count", 0) > 0),
                       "denominator": len([r for r in rows if "A_vs_B" in r])},
        },
        "max_rho_global_bench_default": max((r.get("A_vs_anchor", {}).get("ew_1e-2", {}).get("rho_global", 0) for r in adm), default=None),
        "max_ew_frac_bench_default": max((r.get("A_vs_anchor", {}).get("ew_1e-2", {}).get("frac", 0) for r in adm), default=None),
        "max_rho_global_bench_tf32off": max((r.get("B_vs_anchor", {}).get("ew_1e-2", {}).get("rho_global", 0) for r in adm), default=None),
        "op_class_counts": {c: sum(1 for r in rows if r["op_class"] == c) for c in sorted({r["op_class"] for r in rows})},
        "op_class_adm": {c: sum(1 for r in adm if r["op_class"] == c) for c in sorted({r["op_class"] for r in adm})},
        "wall_s": round(time.time() - t_start, 1),
        "torch": torch.__version__,
        "prereg_sha256": prereg_sha,
        "prereg_path": "lab/s71_prereg.md",
    }
    json.dump({"summary": summary, "rows": rows}, open(SUMOUT, "w"), indent=1)
    print(json.dumps(summary, indent=1)[:3000])


if __name__ == "__main__":
    main()
