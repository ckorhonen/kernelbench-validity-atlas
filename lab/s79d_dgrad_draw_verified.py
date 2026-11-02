#!/usr/bin/env python
"""s79d_dgrad_draw_probe_verified.py — draw invariance on the five DISPATCH-VERIFIED top-1 dgrad tuples.

PREREG (frozen before the run; do not edit after).

s79c corrected the instance grouping: under top-1 dispatch identity the indexed-dgrad instance is five rows
(level1/58, level1/68, level1/73, level2/16, level2/77) with a 3.66x budget span, and s79b measured a 1.12x
draw-to-draw spread at two of them (one of which, level1/79, turned out not to be in the instance at all).
This script repeats the draw probe on the CORRECTED five-tuple population so the invariance statement is made
only on dispatch-verified tuples.

Population (rule, frozen): the five (level, problem_id) pairs in results/s79c_instance_regroup.json
["dgrad_top1_rows"]. Each tuple is engine-verified live before its draws: one torch.profiler pass on the
reference call must return a top-1 kernel containing "sm80_xmma_dgrad_implicit_gemm_indexed_tf32"; if it does
not, the tuple is flagged and its numbers are kept but excluded from the verdict.

Design: model weights fixed once per task (init seed 0). N_DRAWS=8 input draws, seeds 200..207.
Arms per draw, exactly s73's: A = default fp32 reference (cudnn tf32 on, matmul tf32 off); X = fp64 anchor from
the same fp32 weights. Predicate: thr = atol + rtol*|X| at 1e-4 and 1e-2. Element-chunked comparison.
Quantities per draw: rho_max at 1e-4, frac over budget at 1e-4 and 1e-2, max abs |A-X|.

FROZEN VERDICT RULE (three-way, declared before the run):
  DRAW_INVARIANT if every engine-verified tuple has max(rho)/min(rho) <= 1.5;
  DRAW_DEPENDENT if any engine-verified tuple exceeds 2.0;
  DRAW_PARTIAL otherwise.
Rationale: the corrected across-shape span is 3.66x, so a draw spread <=1.5x bounds the draw-attributable share
well below the shape-attributable one.

Limits declared: eight seeds per tuple, one draw generator per task, GB10 sm_120, torch 2.14.1+cu130, one run.
"""
import importlib.util, json, os, signal, time, hashlib, glob
import torch

KB = os.path.expanduser("~/kernelbench/KernelBench")
LAB = os.path.expanduser("~/kb-lab")
OUT = os.path.join(LAB, "results", "s79d_dgrad_draw_verified.jsonl")
SUMOUT = os.path.join(LAB, "results", "s79d_dgrad_draw_verified_summary.json")
S79C = os.path.join(LAB, "results", "s79c_instance_regroup.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

NEEDLE = "sm80_xmma_dgrad_implicit_gemm_indexed_tf32"
ELEM_CHUNK = 1 << 22
ATOL_RTOL = 1e-4
ATOL_RTOL_BENCH = 1e-2
N_DRAWS = 8
SEED_BASE = 200
CFG_ALARM_S = 300


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
        model(*ci)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            model(*ci)
            torch.cuda.synchronize()
    names = []
    for evt in prof.key_averages():
        if evt.self_device_time_total > 0:
            names.append((evt.key, evt.self_device_time_total))
    names.sort(key=lambda t: -t[1])
    return [n for n, _ in names[:6]]


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
    s79c = json.load(open(S79C))
    targets = [(int(r["level"]), int(r["problem_id"])) for r in s79c["dgrad_top1_rows"]]
    print("targets", targets, flush=True)
    rows = []
    for level, pid in targets:
        path = resolve(level, pid)
        torch.manual_seed(0)
        torch.cuda.empty_cache()
        mod = load_task(path)
        init_args = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
        torch.manual_seed(0)
        model = mod.Model(*init_args)
        torch.manual_seed(SEED_BASE)
        probe_inputs = mod.get_inputs()
        kernels = profile_kernels(model, probe_inputs)
        verified = any(NEEDLE in k for k in kernels)
        print("tuple L%d/%d verified=%s top=%s" % (level, pid, verified, kernels[0][:80] if kernels else None), flush=True)
        del probe_inputs
        torch.cuda.empty_cache()
        for i in range(N_DRAWS):
            seed = SEED_BASE + i
            torch.manual_seed(seed)
            inputs = mod.get_inputs()
            rec = {"level": level, "problem_id": pid, "name": os.path.basename(path),
                   "draw_seed": seed, "engine_verified": verified,
                   "kernel_top1_live": kernels[0] if kernels else None,
                   "prereg_sha": prereg_sha, "producer": "s79d_dgrad_draw_verified"}
            try:
                signal.alarm(CFG_ALARM_S)
                set_cfg(True, False)
                A = run_model(model, inputs, torch.float32, dev)
                if A is None:
                    raise RuntimeError("no tensor output")
                set_cfg(False, False)
                X = run_model(model, inputs, torch.float64, dev)
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
            print("  L%d/%d draw %d rho=%s frac=%s" % (level, pid, seed,
                  rec.get("1e-4", {}).get("rho_max"), rec.get("1e-4", {}).get("frac_over")), flush=True)

    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    per = {}
    for level, pid in targets:
        rs = [r for r in rows if r["level"] == level and r["problem_id"] == pid and r["status"] == "ok"]
        if not rs:
            per["L%d/%d" % (level, pid)] = {"n_ok": 0}
            continue
        rhos = [r["1e-4"]["rho_max"] for r in rs]
        fr = [r["1e-4"]["frac_over"] for r in rs]
        lo, hi = min(rhos), max(rhos)
        per["L%d/%d" % (level, pid)] = {
            "n_ok": len(rs), "engine_verified": rs[0]["engine_verified"],
            "rho_min": lo, "rho_max": hi, "rho_median": sorted(rhos)[len(rhos) // 2],
            "draw_spread_max_over_min": (hi / lo) if lo > 0 else None,
            "frac_over_1e-4_min": min(fr), "frac_over_1e-4_max": max(fr),
            "frac_over_1e-2_max": max(r["1e-2"]["frac_over"] for r in rs),
        }
    ver = [v for v in per.values() if v.get("engine_verified")]
    spreads = [v["draw_spread_max_over_min"] for v in ver if v.get("draw_spread_max_over_min")]
    if not spreads:
        verdict = "UNDERPOWERED"
    elif all(s <= 1.5 for s in spreads):
        verdict = "DRAW_INVARIANT"
    elif any(s > 2.0 for s in spreads):
        verdict = "DRAW_DEPENDENT"
    else:
        verdict = "DRAW_PARTIAL"
    summary = {"prereg_sha": prereg_sha, "n_draws": N_DRAWS, "targets": targets,
               "per_tuple": per, "verdict": verdict, "wall_s": round(time.time() - t_start, 1),
               "across_shape_span_s79c": s79c["dgrad_top1_span"],
               "limits": ["8 seeds per tuple", "one draw generator per task", "GB10 sm_120", "torch 2.14.1+cu130"]}
    json.dump(summary, open(SUMOUT, "w"), indent=1)
    print("VERDICT", verdict, json.dumps(per, indent=1), flush=True)
    print("wrote", OUT, SUMOUT, flush=True)


if __name__ == "__main__":
    main()
