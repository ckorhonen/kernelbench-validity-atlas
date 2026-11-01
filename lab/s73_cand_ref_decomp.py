#!/usr/bin/env python
"""s73_cand_ref_decomp.py — candidate-error decomposition at the benchmark's own gate.

PREREG (frozen before the completed run; do not edit after).

AMENDMENT v2 — declared before any completed run. The first attempt (v1, commit ca3fcf9) was killed by an
ssh disconnect after 8 rows and wrote no summary. Those 8 rows showed the compile candidate passing the
default-reference gate at 1e-4 on 8/8 tasks (fail_default = 0 everywhere), which makes the failing-element
split vacuous. Two changes, both declared here and both measurement-neutral with respect to the quantities
already registered: (a) a second candidate arm C2 = the same model run with cudnn/matmul tf32 off (s71's B
arm) is added, because it is a genuinely different arithmetic realization and gives the split something to
split; (b) the two tolerance legs are computed in one shared pass and ratio percentiles are sampled from a
contiguous prefix of the first chunk instead of a strided reservoir over the whole tensor. No threshold,
population or kill clause changed. The v1 rows are discarded and regenerated uniformly.

Question the audit cannot answer alone: s71 measured the reference's own deviation from an fp64 anchor
(54/192 tasks exceed 1e-4 on the reference side). That is a reference-side property. The missing
measurement is the join: does that deviation change a CANDIDATE's verdict, and when a candidate fails, how
much of its margin is the reference's error and how much is its own?

Population (rule, not outcome): the 54 rows of results/s71_full_audit.jsonl with exceed_strict_default
true (anchor admissible), plus 8 negative controls = admissible rows with exceed_strict_default false,
selected as every 13th such row in file order. Total 62 tasks.

Arms per task, all on the same model instance and the same inputs (s70 lesson: one instantiation):
  A   default fp32 reference realization (cudnn tf32 on, matmul tf32 off) — the benchmark's shipped default
  X   fp64 anchor: model.to(float64) from the same fp32 weights (exact widening), flags off
  C1  torch.compile(model, dynamic=False) candidate, fp32, default flags
  C2  same model, fp32, tf32 off (different arithmetic realization)

Predicates: evaluator-exact direction, i.e. allclose(reference, candidate) scales rtol by the CANDIDATE,
so threshold b_e = atol + rtol*|candidate| (eval.py:804 verified in earlier ticks). Both 1e-4 (the axis
r19/A6-A9 used) and 1e-2 (the benchmark's shipped gate for fp32).

TIE-OUT GATE (must pass before any new number is reported): recompute s71's recorded A_vs_anchor count at
1e-4 under s71's own reference-scaled threshold (thr = atol + rtol*|anchor|) and require exact equality with
the recorded count for every task. Any mismatch is recorded as a harness defect and the task is excluded
from the new numbers (declared, not silently repaired).

Measurements per task and per candidate arm:
  m1 ref_budget_share: distribution of d_ref/b_e where d_ref = |A-X|, b_e = atol+rtol*|C| — the fraction of
     the candidate's tolerance budget consumed by the reference's own error. Median, q90, max, frac(d_ref>b_e).
  m2 verdict_default: elements and task-level pass/fail of |C-A| vs b_e at 1e-4 and 1e-2.
  m3 verdict_anchor:  same with X as reference (counterfactual exact reference) — the reference-swap verdict.
  m4 failing-element split: among elements where |C-A| > b_e, count those where d_ref > b_e (would fail even
     with an exact candidate = REFERENCE-INDUCED) vs those where d_ref <= b_e (CANDIDATE-INDUCED).
  m5 fidelity: bit-identity of C vs A, compile wall time, dynamo/inductor failures.

Kill clauses (predeclared):
  K1  if no candidate arm fails any of the 54 tasks under the default reference at 1e-4 => the decomposition
      is vacuous on this population; report the null: the reference's 1e-4 deviation does not propagate to a
      verdict for these candidates, and the 54/192 figure stays a reference-side property only.
  K2  if the pooled reference-induced share of failing elements is < 5% => the reference's error is not a
      material contributor to candidate failure margins at 1e-4 (element-level form of the same conclusion).
  K3  instrument: if >25% of tasks fail to produce both candidates, the population is narrowed to completed
      tasks and stated as such; no verdict claim beyond them.

Controls: the 8 non-exceeding tasks must show the reference's over-budget element fraction <= 0.10 under the
s71 reference-scaled predicate; otherwise the harness is recorded as instrument-defective.
"""
import importlib.util, json, os, re, signal, time, hashlib
import torch

KB = os.path.expanduser("~/kernelbench/KernelBench")
LAB = os.path.expanduser("~/kb-lab")
OUT = os.path.join(LAB, "results", "s73_cand_ref_decomp.jsonl")
SUMOUT = os.path.join(LAB, "results", "s73_cand_ref_decomp_summary.json")
S71 = os.path.join(LAB, "results", "s71_full_audit.jsonl")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

ATOL_RTOL_STRICT = 1e-4
ATOL_RTOL_BENCH = 1e-2
CFG_ALARM_S = 240
GLOBAL_DEADLINE_S = 2400
N_CONTROLS = 8
CHUNK = 1 << 22
RATIO_CAP = 20000


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


def decomp(a64, x, c, atol_s, atol_b):
    """One shared pass over both tolerance legs. a64 = default reference in fp64, x = fp64 anchor,
    c = candidate fp32. b_e = atol + rtol*|c| (candidate-scaled, evaluator-exact)."""
    c64 = c.double()
    shape = a64.shape
    if a64.ndim == 0:
        a64 = a64.reshape(1); x = x.reshape(1); c64 = c64.reshape(1)
        rows, cols = 1, 1
    else:
        rows, cols = a64.reshape(-1, a64.shape[-1]).shape
        a64 = a64.reshape(rows, cols); x = x.reshape(rows, cols); c64 = c64.reshape(rows, cols)
    out = {}
    for tag, tol in (("1e-4", atol_s), ("1e-2", atol_b)):
        out[tag] = {"fail_default": 0, "fail_anchor": 0, "ref_induced": 0, "cand_induced": 0,
                    "ref_over_budget": 0, "total": int(rows * cols), "maxabs_ca": 0.0, "maxabs_cx": 0.0,
                    "maxabs_ax": 0.0, "max_ratio_ref": 0.0, "max_ratio_cand": 0.0,
                    "ratio_samples": [], "bit_identical_ca": True}
    for i in range(0, rows, CHUNK):
        ac = a64[i:i + CHUNK]; xc = x[i:i + CHUNK]; cc = c64[i:i + CHUNK]
        d_ca = (cc - ac).abs()
        d_cx = (cc - xc).abs()
        d_ax = (ac - xc).abs()
        for tag, tol in (("1e-4", atol_s), ("1e-2", atol_b)):
            s = out[tag]
            thr = tol + tol * cc.abs()
            fail_d = d_ca > thr
            fail_a = d_cx > thr
            refover = d_ax > thr
            s["fail_default"] += int(fail_d.sum())
            s["fail_anchor"] += int(fail_a.sum())
            s["ref_over_budget"] += int(refover.sum())
            s["ref_induced"] += int((fail_d & refover).sum())
            s["cand_induced"] += int((fail_d & ~refover).sum())
            s["maxabs_ca"] = max(s["maxabs_ca"], float(d_ca.max()))
            s["maxabs_cx"] = max(s["maxabs_cx"], float(d_cx.max()))
            s["maxabs_ax"] = max(s["maxabs_ax"], float(d_ax.max()))
            r_ref = d_ax / thr
            r_cand = d_ca / thr
            s["max_ratio_ref"] = max(s["max_ratio_ref"], float(r_ref.max()))
            s["max_ratio_cand"] = max(s["max_ratio_cand"], float(r_cand.max()))
            if d_ca.max() > 0:
                s["bit_identical_ca"] = False
            if len(s["ratio_samples"]) < RATIO_CAP:
                s["ratio_samples"].extend(r_ref.reshape(-1)[:RATIO_CAP - len(s["ratio_samples"])].tolist())
    for tag in out:
        s = out[tag]
        smp = s.pop("ratio_samples")
        smp.sort()
        def q(p):
            return smp[min(len(smp) - 1, max(0, int(round(p * (len(smp) - 1)))))] if smp else None
        s["ratio_ref_median"] = q(0.5)
        s["ratio_ref_q90"] = q(0.9)
        s["ratio_ref_q99"] = q(0.99)
        s["n_ratio_samples"] = len(smp)
        s["verdict_default"] = "fail" if s["fail_default"] > 0 else "pass"
        s["verdict_anchor"] = "fail" if s["fail_anchor"] > 0 else "pass"
        s["ref_share_of_fails"] = (s["ref_induced"] / s["fail_default"]) if s["fail_default"] else None
        s["cand_share_of_fails"] = (s["cand_induced"] / s["fail_default"]) if s["fail_default"] else None
    return out


def main():
    torch.cuda.init()
    dev = "cuda"
    t_start = time.time()
    prereg_sha = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()

    rows71 = [json.loads(l) for l in open(S71)]
    strict = [r for r in rows71 if r.get("exceed_strict_default")]
    admissible = [r for r in rows71 if r.get("anchor") == "admissible"]
    controls = [r for r in admissible if not r.get("exceed_strict_default")][::13][:N_CONTROLS]
    pop = [("exceed", r) for r in strict] + [("control", r) for r in controls]
    print("population", len(pop), "exceed", len(strict), "controls", len(controls), flush=True)

    out_rows, excluded = [], []
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
               "s71_output_shape": r71.get("output_shape"), "s71_ref_max_abs": r71.get("ref_max_abs")}
        t_task = time.time()
        try:
            torch.manual_seed(0)
            torch.cuda.empty_cache()
            mod = load_task(path)
            init_args = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
            inputs = mod.get_inputs()
            torch.manual_seed(0)
            model = mod.Model(*init_args)

            signal.alarm(CFG_ALARM_S)
            t0 = time.time()
            set_cfg(True, False)
            A = run_model(model, inputs, torch.float32, dev)
            rec["t_ref_s"] = round(time.time() - t0, 3)
            signal.alarm(0)
            if A is None:
                raise RuntimeError("no tensor output from reference")

            signal.alarm(CFG_ALARM_S)
            t0 = time.time()
            set_cfg(False, False)
            X = run_model(model, inputs, torch.float64, dev)
            rec["t_anchor_s"] = round(time.time() - t0, 3)
            signal.alarm(0)
            if X is None or X.shape != A.shape:
                raise RuntimeError("anchor missing or shape mismatch")
            model.float()

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

            a64 = A.double()

            # C1: torch.compile candidate
            signal.alarm(CFG_ALARM_S)
            t0 = time.time()
            set_cfg(True, False)
            try:
                cmod = torch.compile(model, dynamic=False)
                C1 = run_model(cmod, inputs, torch.float32, dev)
                rec["compile_ok"] = True
            except Exception as e:
                rec["compile_ok"] = False
                rec["compile_error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
                C1 = None
            rec["t_compile_s"] = round(time.time() - t0, 3)
            signal.alarm(0)
            if C1 is not None and C1.shape != A.shape:
                raise RuntimeError("compile candidate shape mismatch")

            # C2: tf32-off candidate (different arithmetic realization)
            signal.alarm(CFG_ALARM_S)
            t0 = time.time()
            set_cfg(False, False)
            C2 = run_model(model, inputs, torch.float32, dev)
            rec["t_tf32off_s"] = round(time.time() - t0, 3)
            signal.alarm(0)
            if C2 is None or C2.shape != A.shape:
                raise RuntimeError("tf32-off candidate missing or shape mismatch")

            for arm, C in (("c1_compile", C1), ("c2_tf32off", C2)):
                if C is None:
                    continue
                signal.alarm(CFG_ALARM_S)
                d = decomp(a64, X, C, ATOL_RTOL_STRICT, ATOL_RTOL_BENCH)
                signal.alarm(0)
                rec["decomp_" + arm] = d
            del a64, X, A, C1, C2
            rec["status"] = "ok"
        except Timeout as e:
            rec["status"] = "timeout"
            excluded.append({"problem_id": pid, "name": fname, "reason": str(e)})
        except Exception as e:
            rec["status"] = "error:%s: %s" % (type(e).__name__, str(e)[:200])
            excluded.append({"problem_id": pid, "name": fname, "reason": rec["status"]})
        finally:
            signal.alarm(0)
        rec["t_task_s"] = round(time.time() - t_task, 2)
        out_rows.append(rec)
        with open(OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        brief = {"pid": pid, "kind": kind, "st": rec.get("status"), "t": rec.get("t_task_s")}
        for arm in ("c1_compile", "c2_tf32off"):
            dd = rec.get("decomp_" + arm, {}).get("1e-4", {})
            brief[arm] = {"v": dd.get("verdict_default"), "n": dd.get("fail_default"),
                          "rs": dd.get("ref_share_of_fails"), "rmed": dd.get("ratio_ref_median")}
        print(json.dumps(brief), flush=True)
        torch.cuda.empty_cache()

    ok = [r for r in out_rows if r.get("status") == "ok"]
    ex = [r for r in ok if r["kind"] == "exceed"]
    ct = [r for r in ok if r["kind"] == "control"]

    def agg(arm, tag):
        rows = [r["decomp_" + arm][tag] for r in ex if "decomp_" + arm in r]
        fail_def = sum(1 for r in rows if r["verdict_default"] == "fail")
        tot_fail = sum(r["fail_default"] for r in rows)
        tot_ref = sum(r["ref_induced"] for r in rows)
        tot_cand = sum(r["cand_induced"] for r in rows)
        meds = sorted(x for x in (r["ratio_ref_median"] for r in rows) if x is not None)
        return {"tasks": len(rows), "tasks_fail_default": fail_def,
                "tasks_fail_anchor": sum(1 for r in rows if r["verdict_anchor"] == "fail"),
                "verdict_flips_ref_swap": sum(1 for r in rows if r["verdict_default"] != r["verdict_anchor"]),
                "failing_elements_default": tot_fail, "ref_induced_elements": tot_ref,
                "cand_induced_elements": tot_cand,
                "pooled_ref_share_of_fails": (tot_ref / tot_fail) if tot_fail else None,
                "pooled_cand_share_of_fails": (tot_cand / tot_fail) if tot_fail else None,
                "median_of_task_ratio_ref_median": meds[len(meds) // 2] if meds else None,
                "median_of_task_ratio_ref_q90": (lambda v: v[len(v) // 2] if v else None)(
                    sorted(x for x in (r["ratio_ref_q90"] for r in rows) if x is not None)),
                "max_ratio_ref": max((r["max_ratio_ref"] for r in rows), default=None),
                "tasks_with_ref_over_budget": sum(1 for r in rows if r["ref_over_budget"] > 0)}

    ctrl_frac = [r["decomp_c1_compile"]["1e-4"]["ref_over_budget"] / max(1, r["decomp_c1_compile"]["1e-4"]["total"])
                 for r in ct if "decomp_c1_compile" in r]
    summary = {
        "prereg": "docstring of this file, frozen pre-run (amendment v2 declared before the completed run)",
        "prereg_sha256": prereg_sha,
        "population": {"exceed_rows": len(strict), "controls": len(controls), "attempted": len(out_rows),
                       "ok": len(ok), "excluded": excluded, "truncated": truncated},
        "c1_compile": {"1e-4": agg("c1_compile", "1e-4"), "1e-2": agg("c1_compile", "1e-2")},
        "c2_tf32off": {"1e-4": agg("c2_tf32off", "1e-4"), "1e-2": agg("c2_tf32off", "1e-2")},
        "controls": {"n": len(ct), "max_ref_over_budget_frac": max(ctrl_frac) if ctrl_frac else None,
                     "instrument_ok": bool(ctrl_frac and max(ctrl_frac) <= 0.10)},
        "compile": {"attempted": len([r for r in out_rows if "compile_ok" in r]),
                    "ok": sum(1 for r in out_rows if r.get("compile_ok"))},
        "wall_s": round(time.time() - t_start, 1), "torch": torch.__version__, "kills": {},
    }
    s1 = summary["c1_compile"]["1e-4"]; s2 = summary["c2_tf32off"]["1e-4"]
    summary["kills"]["K1_vacuous"] = bool(s1["tasks_fail_default"] == 0 and s2["tasks_fail_default"] == 0)
    shares = [x for x in (s1["pooled_ref_share_of_fails"], s2["pooled_ref_share_of_fails"]) if x is not None]
    summary["kills"]["K2_ref_not_material"] = bool(shares and max(shares) < 0.05)
    summary["kills"]["K3_compile_coverage"] = bool(summary["compile"]["attempted"] > 0 and
                                                  summary["compile"]["ok"] < 0.75 * summary["compile"]["attempted"])
    json.dump(summary, open(SUMOUT, "w"), indent=1)
    print(json.dumps(summary, indent=1)[:4000])


if __name__ == "__main__":
    main()
