#!/usr/bin/env python
"""s73_cand_ref_decomp.py — candidate-error decomposition at the benchmark's own gate.

PREREG (frozen at commit time, BEFORE first run; do not edit after the run).

Question the audit cannot answer alone: s71 measured the reference's own deviation from an fp64 anchor
(54/192 tasks exceed 1e-4 on the reference side). That is a reference-side property. The community's
missing measurement is the join: does that reference deviation change a CANDIDATE's verdict, and when a
candidate fails, how much of its margin is the reference's error and how much is its own?

Population (rule, not outcome): the 54 rows of results/s71_full_audit.jsonl with exceed_strict_default
true (anchor admissible), plus 8 negative controls = admissible rows with exceed_strict_default false,
selected as every 13th such row in file order. Total 62 tasks.

Arms per task, all on the same model instance and the same inputs (s70 lesson: one instantiation):
  A  default fp32 reference realization (cudnn tf32 on, matmul tf32 off) — the benchmark's shipped default
  X  fp64 anchor: model.to(float64) from the same fp32 weights (exact widening), flags off
  C  torch.compile(model, dynamic=False) candidate, fp32, default flags

Predicates: evaluator-exact direction, i.e. allclose(reference, candidate) scales rtol by the CANDIDATE,
so threshold b_e = atol + rtol*|candidate| (eval.py:804 verified in earlier ticks). Both 1e-4 (the axis
r19/A6-A9 used) and 1e-2 (the benchmark's shipped gate for fp32).

TIE-OUT GATE (must pass before any new number is reported): recompute s71's recorded A_vs_anchor count at
1e-4 under s71's own reference-scaled threshold (thr = atol + rtol*|anchor|) and require exact equality with
the recorded count for every task. Any mismatch is recorded as a harness defect and the task is excluded
from the new numbers (declared, not silently repaired).

Measurements per task:
  m1 ref_budget_share: distribution of d_ref/b_e where d_ref = |A-X|, b_e = atol+rtol*|C| — the fraction of
     the candidate's tolerance budget consumed by the reference's own error. Median, q90, frac(d_ref>b_e).
  m2 verdict_default: elements and task-level pass/fail of |C-A| vs b_e at 1e-4 and 1e-2.
  m3 verdict_anchor:  same with X as reference (counterfactual exact reference) — the reference-swap verdict.
  m4 failing-element split: among elements where |C-A| > b_e (candidate fails under the default reference),
     count those where d_ref > b_e (would fail even with an exact candidate = REFERENCE-INDUCED) vs those
     where d_ref <= b_e (CANDIDATE-INDUCED).
  m5 compile fidelity: bit-identity of C vs A, compile wall time, dynamo/inductor failures.

Kill clauses (predeclared):
  K1  if C fails 0 of the 54 tasks under the default reference at 1e-4 => the decomposition is vacuous on
      this population; report the null: the reference's 1e-4 deviation does not propagate to a verdict for a
      compile-class candidate, and the 54/192 figure stays a reference-side property only.
  K2  if the pooled reference-induced share of failing elements is < 5% => the reference's error is not a
      material contributor to candidate failure margins at 1e-4 (same conclusion, element-level form).
  K3  instrument: if >25% of tasks fail to compile, the population is narrowed to compiled tasks and stated
      as such; no verdict claim beyond them.

Controls: the 8 non-exceeding tasks must show d_ref/b_e median < 1 by construction; if any control shows
frac(d_ref>b_e) > 0.10 the harness is suspect and the run is recorded as instrument-defective.
"""
import importlib.util, json, os, re, signal, sys, time, hashlib
import torch

KB = os.path.expanduser("~/kernelbench/KernelBench")
LAB = os.path.expanduser("~/kb-lab")
OUT = os.path.join(LAB, "results", "s73_cand_ref_decomp.jsonl")
SUMOUT = os.path.join(LAB, "results", "s73_cand_ref_decomp_summary.json")
S71 = os.path.join(LAB, "results", "s71_full_audit.jsonl")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

ATOL_RTOL_STRICT = 1e-4
ATOL_RTOL_BENCH = 1e-2
CFG_ALARM_S = 180
GLOBAL_DEADLINE_S = 1500
N_CONTROLS = 8
CHUNK = 1 << 22


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


def ew_count_ref_scaled(a, b, atol, rtol):
    """s71's predicate: thr = atol + rtol*|b| (b is the anchor). Used ONLY for the tie-out gate."""
    a64 = a.double()
    b = b.double()
    if a64.ndim == 0:
        return int((a64 - b).abs() > atol + rtol * b.abs())
    fa = a64.reshape(-1, a64.shape[-1])
    fb = b.reshape(-1, b.shape[-1])
    cnt = 0
    for i in range(0, fa.shape[0], CHUNK):
        d = (fa[i:i + CHUNK] - fb[i:i + CHUNK]).abs()
        thr = atol + rtol * fb[i:i + CHUNK].abs()
        cnt += int((d > thr).sum())
    return cnt


def decomp(a, x, c, atol, rtol, ratio_stride=997, ratio_cap=20000):
    """Elementwise decomposition. a = default fp32 reference, x = fp64 anchor, c = candidate (fp32).
    Threshold b_e = atol + rtol*|c| (candidate-scaled, evaluator-exact).
    Returns counts for |c-a|>b_e (fail default), |c-x|>b_e (fail anchor), and the reference-induced split."""
    a64 = a.double()
    x = x.double()
    c64 = c.double()
    shape = a64.shape
    if a64.ndim == 0:
        a64 = a64.reshape(1); x = x.reshape(1); c64 = c64.reshape(1)
        rows, cols = 1, 1
    else:
        rows, cols = a64.reshape(-1, a64.shape[-1]).shape
        a64 = a64.reshape(rows, cols); x = x.reshape(rows, cols); c64 = c64.reshape(rows, cols)

    stats = {
        "fail_default": 0, "fail_anchor": 0, "ref_induced": 0, "cand_induced": 0,
        "ref_over_budget": 0, "total": int(rows * cols),
        "maxabs_ca": 0.0, "maxabs_cx": 0.0, "maxabs_ax": 0.0,
        "max_ratio_ref": 0.0, "max_ratio_cand": 0.0,
        "ratio_samples": [], "bit_identical_ca": True,
    }
    k = 0
    for i in range(0, rows, CHUNK):
        ac = a64[i:i + CHUNK]; xc = x[i:i + CHUNK]; cc = c64[i:i + CHUNK]
        d_ca = (cc - ac).abs()
        d_cx = (cc - xc).abs()
        d_ax = (ac - xc).abs()
        thr = atol + rtol * cc.abs()
        fail_d = d_ca > thr
        fail_a = d_cx > thr
        refover = d_ax > thr
        stats["fail_default"] += int(fail_d.sum())
        stats["fail_anchor"] += int(fail_a.sum())
        stats["ref_over_budget"] += int(refover.sum())
        stats["ref_induced"] += int((fail_d & refover).sum())
        stats["cand_induced"] += int((fail_d & ~refover).sum())
        stats["maxabs_ca"] = max(stats["maxabs_ca"], float(d_ca.max()))
        stats["maxabs_cx"] = max(stats["maxabs_cx"], float(d_cx.max()))
        stats["maxabs_ax"] = max(stats["maxabs_ax"], float(d_ax.max()))
        r_ref = d_ax / thr
        r_cand = d_ca / thr
        stats["max_ratio_ref"] = max(stats["max_ratio_ref"], float(r_ref.max()))
        stats["max_ratio_cand"] = max(stats["max_ratio_cand"], float(r_cand.max()))
        if d_ca.max() > 0:
            stats["bit_identical_ca"] = False
        # bounded reservoir for percentiles of the reference's budget consumption
        if len(stats["ratio_samples"]) < ratio_cap:
            flat = r_ref.reshape(-1)
            take = flat[::ratio_stride]
            room = ratio_cap - len(stats["ratio_samples"])
            stats["ratio_samples"].extend(take[:room].tolist())
    return stats


def pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def main():
    torch.cuda.init()
    dev = "cuda"
    t_start = time.time()
    prereg_sha = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()

    rows71 = [json.loads(l) for l in open(S71)]
    strict = [r for r in rows71 if r.get("exceed_strict_default")]
    admissible = [r for r in rows71 if r.get("anchor") == "admissible"]
    controls_pool = [r for r in admissible if not r.get("exceed_strict_default")]
    controls = controls_pool[::13][:N_CONTROLS]
    pop = [("exceed", r) for r in strict] + [("control", r) for r in controls]
    print("population", len(pop), "exceed", len(strict), "controls", len(controls), flush=True)

    out_rows = []
    excluded = []
    truncated = False
    for kind, r71 in pop:
        if time.time() - t_start > GLOBAL_DEADLINE_S:
            truncated = True
            excluded.append({"problem_id": r71["problem_id"], "name": r71["name"], "reason": "global deadline"})
            continue
        level, pid, fname = r71["level"], r71["problem_id"], r71["name"]
        path = os.path.join(KB, "KernelBench", "level%d" % level, fname)
        rec = {"kind": kind, "level": level, "problem_id": pid, "name": fname,
               "op_class": r71.get("op_class"), "s71_strict_count": r71["A_vs_anchor"]["ew_1e-4"]["count"],
               "s71_output_shape": r71.get("output_shape")}
        try:
            torch.manual_seed(0)
            torch.cuda.empty_cache()
            mod = load_task(path)
            init_args = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
            inputs = mod.get_inputs()
            torch.manual_seed(0)
            model = mod.Model(*init_args)

            # A: default fp32 reference (benchmark shipped default)
            signal.alarm(CFG_ALARM_S)
            t0 = time.time()
            set_cfg(True, False)
            A = run_model(model, inputs, torch.float32, dev)
            rec["t_ref_s"] = round(time.time() - t0, 3)
            signal.alarm(0)
            if A is None:
                raise RuntimeError("no tensor output from reference")

            # X: fp64 anchor from the same weights (exact widening)
            signal.alarm(CFG_ALARM_S)
            t0 = time.time()
            set_cfg(False, False)
            X = run_model(model, inputs, torch.float64, dev)
            rec["t_anchor_s"] = round(time.time() - t0, 3)
            signal.alarm(0)
            if X is None or X.shape != A.shape:
                raise RuntimeError("anchor missing or shape mismatch")
            model.float()

            # TIE-OUT: reproduce s71's count under s71's reference-scaled predicate
            tie = ew_count_ref_scaled(A, X, ATOL_RTOL_STRICT, ATOL_RTOL_STRICT)
            rec["tie_out_count"] = tie
            rec["tie_out_ok"] = bool(tie == rec["s71_strict_count"])
            if not rec["tie_out_ok"]:
                rec["status"] = "tie_out_mismatch"
                excluded.append({"problem_id": pid, "name": fname,
                                 "reason": "tie-out %d != s71 %d" % (tie, rec["s71_strict_count"])})
                out_rows.append(rec)
                with open(OUT, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                continue

            # C: torch.compile candidate
            signal.alarm(CFG_ALARM_S)
            t0 = time.time()
            set_cfg(True, False)
            try:
                cmod = torch.compile(model, dynamic=False)
                C = run_model(cmod, inputs, torch.float32, dev)
                rec["compile_ok"] = True
            except Exception as e:
                rec["compile_ok"] = False
                rec["compile_error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
                C = None
            rec["t_compile_s"] = round(time.time() - t0, 3)
            signal.alarm(0)
            if C is None:
                rec["status"] = "compile_failed"
                excluded.append({"problem_id": pid, "name": fname, "reason": rec.get("compile_error")})
                out_rows.append(rec)
                with open(OUT, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                continue
            if C.shape != A.shape:
                raise RuntimeError("candidate shape mismatch")

            # New measurements: strict (1e-4) and benchmark (1e-2) decompositions
            for tag, tol in (("1e-4", ATOL_RTOL_STRICT), ("1e-2", ATOL_RTOL_BENCH)):
                signal.alarm(CFG_ALARM_S)
                d = decomp(A, X, C, tol, tol)
                signal.alarm(0)
                samples = d.pop("ratio_samples")
                d["ratio_ref_median"] = pct(samples, 0.5)
                d["ratio_ref_q90"] = pct(samples, 0.9)
                d["ratio_ref_q99"] = pct(samples, 0.99)
                d["n_ratio_samples"] = len(samples)
                d["verdict_default"] = "fail" if d["fail_default"] > 0 else "pass"
                d["verdict_anchor"] = "fail" if d["fail_anchor"] > 0 else "pass"
                d["ref_share_of_fails"] = (d["ref_induced"] / d["fail_default"]) if d["fail_default"] else None
                rec["decomp_" + tag] = d
            rec["status"] = "ok"
        except Timeout as e:
            rec["status"] = "timeout"
            excluded.append({"problem_id": pid, "name": fname, "reason": str(e)})
        except Exception as e:
            rec["status"] = "error:%s: %s" % (type(e).__name__, str(e)[:200])
            excluded.append({"problem_id": pid, "name": fname, "reason": rec["status"]})
        finally:
            signal.alarm(0)
        out_rows.append(rec)
        with open(OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps({"pid": pid, "kind": kind, "op": rec.get("op_class"), "status": rec.get("status"),
                          "c1e4": rec.get("decomp_1e-4", {}).get("verdict_default"),
                          "a1e4": rec.get("decomp_1e-4", {}).get("verdict_anchor"),
                          "refshare": rec.get("decomp_1e-4", {}).get("ref_share_of_fails")}), flush=True)

    ok = [r for r in out_rows if r.get("status") == "ok"]
    ex = [r for r in ok if r["kind"] == "exceed"]
    ct = [r for r in ok if r["kind"] == "control"]

    def agg(tag):
        rows = [r["decomp_" + tag] for r in ex]
        fail_def = sum(1 for r in rows if r["verdict_default"] == "fail")
        fail_anc = sum(1 for r in rows if r["verdict_anchor"] == "fail")
        tot_fail = sum(r["fail_default"] for r in rows)
        tot_refind = sum(r["ref_induced"] for r in rows)
        return {
            "tasks": len(rows),
            "tasks_fail_default": fail_def,
            "tasks_fail_anchor": fail_anc,
            "verdict_flips_ref_swap": sum(1 for r in rows if r["verdict_default"] != r["verdict_anchor"]),
            "failing_elements_default": tot_fail,
            "ref_induced_elements": tot_refind,
            "pooled_ref_share_of_fails": (tot_refind / tot_fail) if tot_fail else None,
            "median_ratio_ref": pct([r["ratio_ref_median"] for r in rows if r["ratio_ref_median"] is not None], 0.5),
            "median_q90_ratio_ref": pct([r["ratio_ref_q90"] for r in rows if r["ratio_ref_q90"] is not None], 0.5),
            "tasks_with_ref_over_budget": sum(1 for r in rows if r["ref_over_budget"] > 0),
        }

    ctrl_ref_frac = [r["decomp_1e-4"]["ref_over_budget"] / max(1, r["decomp_1e-4"]["total"]) for r in ct]
    summary = {
        "prereg": "docstring of this file, frozen pre-run",
        "prereg_sha256": prereg_sha,
        "population": {"exceed_rows": len(strict), "controls": len(controls), "attempted": len(out_rows),
                       "ok": len(ok), "excluded": excluded, "truncated": truncated},
        "strict_1e-4": agg("1e-4"),
        "bench_1e-2": agg("1e-2"),
        "controls": {"n": len(ct), "max_ref_over_budget_frac": max(ctrl_ref_frac) if ctrl_ref_frac else None,
                     "instrument_ok": bool(ctrl_ref_frac and max(ctrl_ref_frac) <= 0.10)},
        "compile": {"attempted": len([r for r in out_rows if "compile_ok" in r]),
                    "ok": sum(1 for r in out_rows if r.get("compile_ok")),
                    "bit_identical_to_ref": sum(1 for r in ok if r.get("decomp_1e-4", {}).get("bit_identical_ca"))},
        "wall_s": round(time.time() - t_start, 1),
        "torch": torch.__version__,
        "kills": {},
    }
    s = summary["strict_1e-4"]
    summary["kills"]["K1_vacuous"] = bool(s["tasks_fail_default"] == 0)
    summary["kills"]["K2_ref_not_material"] = bool((s["pooled_ref_share_of_fails"] is not None
                                                   and s["pooled_ref_share_of_fails"] < 0.05))
    summary["kills"]["K3_compile_coverage"] = bool(len(ok) > 0 and
                                                  summary["compile"]["ok"] < 0.75 * summary["compile"]["attempted"])
    json.dump(summary, open(SUMOUT, "w"), indent=1)
    print(json.dumps(summary, indent=1)[:4000])


if __name__ == "__main__":
    main()
