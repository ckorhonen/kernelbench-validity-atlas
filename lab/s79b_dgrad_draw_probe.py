#!/usr/bin/env python
"""s79b_dgrad_draw_probe.py — draw-attributable share of the reference margin at two fixed dgrad tuples.

PREREG (frozen before the run; do not edit after).

Question left open by s79a: inside the single indexed-dgrad engine instance the reference budget ratio is monotone
in accumulation depth (Spearman 0.78, p 0.011, n=10, span 10,654x). That is a shape effect across 12 tuples.
What is NOT measured anywhere in s73-s78 is how much the same margin moves when only the INPUT DRAW changes at a
fixed tuple. Without that number the 10,654x cannot be attributed to shape.

Population (rule, frozen): the two s75 rows in that instance at the ends of the span.
  T1 = level1/58 (s73_bias_1e4 0.4835, K=3360)
  T2 = level1/79 (s73_bias_1e4 4.5378e-05, K=96)
Both are conv_transpose tasks whose reference call dispatches the s75 instance
"sm80_xmma_dgrad_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_...".

Design per tuple: model weights fixed once (init seed 0, as s73). Then N_DRAWS=12 input draws, seeds 100..111;
for each draw, torch.manual_seed(seed) before mod.get_inputs(). Arms per draw, exactly s73's:
  A = default fp32 reference (cudnn tf32 on, matmul tf32 off)
  X = fp64 anchor (same model, model.to(float64) from the same fp32 weights)
Predicate: evaluator-exact thr = atol + rtol*|X| (anchor-scaled, s71 convention) at 1e-4 and 1e-2.
Quantities per draw: rho_max = max(|A-X| / thr_1e-4), frac over budget at 1e-4, max abs |A-X|, out volume.
Element-chunked comparison (ELEM_CHUNK 2^22), sequential release; no full-size fp64 copy materialized.

Engine check: on draw 0 only, one torch.profiler pass on the A call records kernel top-1 names; the tuple is
declared in-instance only if any kernel name contains the instance needle. If not, the tuple is reported as
out-of-instance and its numbers are kept but flagged.

FROZEN VERDICT RULE (single branch, no post-hoc edits):
  DRAW_DEPENDENT  if, at either tuple, max(rho)/min(rho) across draws > 10;
  DRAW_STABLE     otherwise (<= 10).
Rationale for the bar: the across-shape span in this instance is 10,654x, so a draw spread of <=10x accounts for
at most 0.1% of it, which makes the span shape-attributable at the resolution that matters here.

Kill/limits declared: two tuples, one seed family (100..111), one draw generator per task (mod.get_inputs), one
arch (GB10 sm_120), torch 2.14.1+cu130. A DRAW_STABLE result does not prove shape attribution for the other ten
rows; it removes input-value dependence as the explanation for the two ends of the span.
"""
import importlib.util, json, os, signal, time, hashlib, glob
import torch

KB = os.path.expanduser("~/kernelbench/KernelBench")
LAB = os.path.expanduser("~/kb-lab")
OUT = os.path.join(LAB, "results", "s79b_dgrad_draw_probe.jsonl")
SUMOUT = os.path.join(LAB, "results", "s79b_dgrad_draw_probe_summary.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

NEEDLE = "sm80_xmma_dgrad_implicit_gemm_indexed_tf32"
ELEM_CHUNK = 1 << 22
ATOL_RTOL = 1e-4
ATOL_RTOL_BENCH = 1e-2
N_DRAWS = 12
SEED_BASE = 100
CFG_ALARM_S = 300
TARGETS = [(1, 58), (1, 79)]


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


def _rows_cols(t):
    if t.ndim == 0:
        return 1, 1
    return t.reshape(-1, t.shape[-1]).shape


def measure(a, x):
    """Chunked: rho_max and over-budget counts at 1e-4 and 1e-2, anchor-scaled predicate."""
    rows, cols = _rows_cols(a)
    af = a.reshape(rows, cols) if a.ndim else a.reshape(1, 1)
    xf = x.reshape(rows, cols) if x.ndim else x.reshape(1, 1)
    res = {"total": int(rows * cols)}
    for tag, tol in (("1e-4", ATOL_RTOL), ("1e-2", ATOL_RTOL_BENCH)):
        res[tag] = {"over": 0, "rho_max": 0.0, "maxabs": 0.0}
    step = max(1, ELEM_CHUNK // max(1, cols))
    for i in range(0, rows, step):
        ac = af[i:i + step].double()
        xc = xf[i:i + step]
        d = (ac - xc).abs()
        for tag, tol in (("1e-4", ATOL_RTOL), ("1e-2", ATOL_RTOL_BENCH)):
            s = res[tag]
            thr = tol + tol * xc.abs()
            s["over"] += int((d > thr).sum())
            s["maxabs"] = max(s["maxabs"], float(d.max()))
            s["rho_max"] = max(s["rho_max"], float((d / thr).max()))
        del ac, d
    for tag in ("1e-4", "1e-2"):
        s = res[tag]
        s["frac_over"] = s["over"] / float(res["total"]) if res["total"] else None
    return res


def profile_kernels(model, inputs):
    model = model.to(device="cuda", dtype=torch.float32)
    model.eval()
    ci = [cast_inputs(x, torch.float32, "cuda") for x in inputs]
    with torch.no_grad():
        model(*ci)  # warmup
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            model(*ci)
            torch.cuda.synchronize()
    names = []
    for evt in prof.key_averages():
        if evt.device_type == torch.autograd.DeviceType.CUDA or evt.self_device_time_total > 0:
            names.append((evt.key, evt.self_device_time_total))
    names.sort(key=lambda t: -t[1])
    return [n for n, _ in names[:8]]


def resolve(level, pid):
    pat = os.path.join(KB, "KernelBench", "level%d" % level, "%d_*.py" % pid)
    hits = sorted(glob.glob(pat))
    if len(hits) != 1:
        raise RuntimeError("resolve %d/%d -> %d hits" % (level, pid, len(hits)))
    return hits[0]


def main():
    torch.cuda.init()
    dev = "cuda"
    t_start = time.time()
    prereg_sha = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    rows = []
    for level, pid in TARGETS:
        path = resolve(level, pid)
        torch.manual_seed(0)
        torch.cuda.empty_cache()
        mod = load_task(path)
        init_args = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
        torch.manual_seed(0)
        model = mod.Model(*init_args)
        print("tuple L%d/%d %s" % (level, pid, os.path.basename(path)), flush=True)

        # engine identity check on the first draw
        torch.manual_seed(SEED_BASE)
        probe_inputs = mod.get_inputs()
        kernels = profile_kernels(model, probe_inputs)
        in_instance = any(NEEDLE in k for k in kernels)
        print("  engine_in_instance=%s top=%s" % (in_instance, kernels[0][:90] if kernels else None), flush=True)
        del probe_inputs
        torch.cuda.empty_cache()

        for i in range(N_DRAWS):
            seed = SEED_BASE + i
            torch.manual_seed(seed)
            inputs = mod.get_inputs()
            rec = {"level": level, "problem_id": pid, "name": os.path.basename(path),
                   "draw_seed": seed, "engine_in_instance": in_instance,
                   "kernels_top8": kernels[:4], "producer": "s79b_dgrad_draw_probe",
                   "prereg_sha": prereg_sha}
            try:
                signal.alarm(CFG_ALARM_S)
                t0 = time.time()
                set_cfg(True, False)
                A = run_model(model, inputs, torch.float32, dev)
                rec["t_ref_s"] = round(time.time() - t0, 3)
                if A is None:
                    raise RuntimeError("no tensor output")
                t0 = time.time()
                set_cfg(False, False)
                X = run_model(model, inputs, torch.float64, dev)
                rec["t_anchor_s"] = round(time.time() - t0, 3)
                signal.alarm(0)
                rec["out_shape"] = list(A.shape)
                rec.update(measure(A, X))
                rec["status"] = "ok"
                del A, X
            except Exception as e:
                signal.alarm(0)
                rec["status"] = "error:%s" % type(e).__name__
                rec["error"] = str(e)[:300]
                print("  draw %d ERROR %s" % (seed, e), flush=True)
            torch.cuda.empty_cache()
            rows.append(rec)
            print("  draw %d rho=%s frac1e-4=%s wall=%.1fs" %
                  (seed, rec.get("1e-4", {}).get("rho_max"), rec.get("1e-4", {}).get("frac_over"),
                   time.time() - t_start), flush=True)

    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    summary = {"prereg_sha": prereg_sha, "n_draws": N_DRAWS, "targets": TARGETS,
               "wall_s": round(time.time() - t_start, 1), "per_tuple": {}}
    for level, pid in TARGETS:
        rs = [r for r in rows if r["level"] == level and r["problem_id"] == pid and r["status"] == "ok"]
        rhos = [r["1e-4"]["rho_max"] for r in rs]
        fr = [r["1e-4"]["frac_over"] for r in rs]
        if not rhos:
            summary["per_tuple"]["L%d/%d" % (level, pid)] = {"n_ok": 0}
            continue
        lo, hi = min(rhos), max(rhos)
        spread = (hi / lo) if lo > 0 else float("inf")
        mean = sum(rhos) / len(rhos)
        var = sum((v - mean) ** 2 for v in rhos) / len(rhos)
        cv = (var ** 0.5) / mean if mean else None
        summary["per_tuple"]["L%d/%d" % (level, pid)] = {
            "n_ok": len(rs), "engine_in_instance": rs[0]["engine_in_instance"],
            "rho_min": lo, "rho_max": hi, "rho_median": sorted(rhos)[len(rhos) // 2],
            "draw_spread_max_over_min": spread, "rho_cv": cv,
            "frac_over_1e-4_min": min(fr), "frac_over_1e-4_max": max(fr),
            "frac_over_1e-2_max": max(r["1e-2"]["frac_over"] for r in rs),
            "maxabs_min": min(r["1e-4"]["maxabs"] for r in rs),
            "maxabs_max": max(r["1e-4"]["maxabs"] for r in rs),
        }
    spreads = [v.get("draw_spread_max_over_min") for v in summary["per_tuple"].values()
               if v.get("draw_spread_max_over_min") is not None]
    verdict = "DRAW_DEPENDENT" if (spreads and max(spreads) > 10) else "DRAW_STABLE"
    summary["verdict"] = verdict
    summary["across_shape_span_s79a"] = 10654.43425652576
    summary["limits"] = ["two tuples", "seeds 100..111 only", "GB10 sm_120 torch 2.14.1+cu130",
                         "DRAW_STABLE does not prove shape attribution for the other ten rows"]
    json.dump(summary, open(SUMOUT, "w"), indent=1)
    print("VERDICT", verdict, json.dumps(summary["per_tuple"], indent=1), flush=True)
    print("wrote", OUT, SUMOUT, flush=True)


if __name__ == "__main__":
    main()
