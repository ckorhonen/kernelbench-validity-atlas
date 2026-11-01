#!/usr/bin/env python
"""s78_excluded_decomp.py — candidate-error decomposition on the 8 rows s71 excluded from the anchor audit.

PREREG (frozen before the run; do not edit after).

Why: s76 closed coverage to 200/200 and found that the only new strict 1e-4 exceedance in that set is L1/76
conv_standard_1D_dilated_strided (263.65M/1.43B elements = 18.42%, rho 1.835, entirely TF32-caused). That is a
REFERENCE-side fact, exactly like the 54/192 figure was. The plan's next action is the join s73 performed for the
admissible population but never for these eight: does the reference's deviation change a CANDIDATE's verdict on
these shapes, and when a candidate fails, how much of its margin is reference-induced vs candidate-induced?

Population (rule, not outcome): the 8 rows of results/s71_full_audit.jsonl with anchor == "inadmissible:working_set"
(L1 34/35/36/37/38/39/45/76). Arms per task on one model instance and one input draw (s70 lesson):
  A   default fp32 reference (cudnn tf32 on, matmul tf32 off) — the benchmark's shipped default
  X   fp64 anchor (exact widening from the same fp32 weights, flags off)
  C1  torch.compile(model, dynamic=False), fp32, default flags — a candidate sharing the reference engine class
  C2  same model, fp32, tf32 off — a candidate with a different arithmetic realization
Predicates: evaluator-exact direction (allclose(reference, candidate) scales rtol by the CANDIDATE),
b_e = atol + rtol*|candidate|; legs 1e-4 and 1e-2 (the shipped fp32 gate).

TIE-OUT GATE (must pass before any new number): recompute the A-vs-anchor strict count under s71's own
reference-scaled predicate (thr = atol + rtol*|X|) and require exact equality with s76's recorded
tie_out.s71_strict for that task. Mismatch => harness defect, task excluded from new numbers (declared, not
silently repaired).

Measurements: m1 verdict_default/m3 verdict_anchor (elements + task verdict) per arm per leg; m4 failing-element
split into REFERENCE-INDUCED (d_ref > b_e, would fail with an exact candidate) and CANDIDATE-INDUCED; m5 bit
identity C vs A, compile wall; m2 reference budget share distribution (median/q90/q99/max of d_ref/b_e).

Kill clauses (predeclared):
  K1  if neither candidate arm fails any task at 1e-4 under the default reference => decomposition vacuous on this
      set; report the null and keep the 1e-4 exceedance a reference-side property only.
  K2  if the pooled reference-induced share of failing elements at 1e-4 is < 5% => the reference error is not a
      material contributor to candidate margins here.
  K3  instrument: if >25% of tasks fail to produce both candidates, narrow the population and state it.
  K4  (100x-within-one-instance falsifier, policy carried from s75b) if within this 8-row set the reference budget
      ratio spans >=100x, the per-shape budget column must not be replaced by a set-level mean anywhere.

Expected-but-not-required reads (recorded, no threshold): L1/76 should reproduce s73's pattern — compile 0/1
(engine-class sharing inherits a pass) and tf32-off 1/1 with ~100% reference-induced share; the seven norm/pool
rows are reference-clean at 1e-4 per s76 so their candidate arms should pass on both legs.
"""
import importlib.util, json, os, signal, time, hashlib
import torch

KB = os.path.expanduser("~/kernelbench/KernelBench")
LAB = os.path.expanduser("~/kb-lab")
OUT = os.path.join(LAB, "results", "s78_excluded_decomp.jsonl")
SUMOUT = os.path.join(LAB, "results", "s78_excluded_decomp_summary.json")
S71 = os.path.join(LAB, "results", "s71_full_audit.jsonl")
S76 = os.path.join(LAB, "results", "s76_anchor_coverage.jsonl")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

ATOL_RTOL_STRICT = 1e-4
ATOL_RTOL_BENCH = 1e-2
CFG_ALARM_S = 300
GLOBAL_DEADLINE_S = 2100
CHUNK = 1 << 22


def load_s73():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s73_cand_ref_decomp.py")
    spec = importlib.util.spec_from_file_location("s73mod", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S73 = load_s73()


def ref_scaled_count(A, X, atol, rtol):
    """s71/s76 predicate: thr = atol + rtol*|X| (anchor-scaled). Used only for the tie-out gate."""
    a64 = A.double()
    x64 = X.double()
    fa = a64.reshape(-1, a64.shape[-1])
    fx = x64.reshape(-1, x64.shape[-1])
    cnt = 0
    for i in range(0, fa.shape[0], CHUNK):
        d = (fa[i:i + CHUNK] - fx[i:i + CHUNK]).abs()
        thr = atol + rtol * fx[i:i + CHUNK].abs()
        cnt += int((d > thr).sum())
    del a64, x64
    return cnt


def main():
    torch.cuda.init()
    dev = "cuda"
    t_start = time.time()
    prereg_sha = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()

    rows71 = [json.loads(l) for l in open(S71)]
    rows76 = [json.loads(l) for l in open(S76)]
    s76_by = {(r["level"], r["problem_id"]): r for r in rows76}
    pop = [r for r in rows71 if r.get("anchor") == "inadmissible:working_set"]
    print("population", len(pop), flush=True)

    out_rows, excluded = [], []
    for r71 in pop:
        if time.time() - t_start > GLOBAL_DEADLINE_S:
            excluded.append({"problem_id": r71["problem_id"], "name": r71["name"], "reason": "global deadline"})
            continue
        level, pid, fname = r71["level"], r71["problem_id"], r71["name"]
        path = os.path.join(KB, "KernelBench", "level%d" % level, fname)
        r76 = s76_by.get((level, pid), {})
        rec = {"level": level, "problem_id": pid, "name": fname, "op_class": r71.get("op_class"),
               "s76_output_shape": r76.get("output_shape"), "s76_ref_max_abs": r76.get("ref_max_abs"),
               "s76_s71_strict": (r76.get("tie_out") or {}).get("s71_strict")}
        t_task = time.time()
        try:
            torch.manual_seed(0)
            torch.cuda.empty_cache()
            mod = S73.load_task(path)
            init_args = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
            inputs = mod.get_inputs()
            torch.manual_seed(0)
            model = mod.Model(*init_args)

            signal.alarm(CFG_ALARM_S)
            t0 = time.time()
            S73.set_cfg(True, False)
            A = S73.run_model(model, inputs, torch.float32, dev)
            rec["t_ref_s"] = round(time.time() - t0, 3)
            signal.alarm(0)
            if A is None:
                raise RuntimeError("no tensor output from reference")

            signal.alarm(CFG_ALARM_S)
            t0 = time.time()
            S73.set_cfg(False, False)
            X = S73.run_model(model, inputs, torch.float64, dev)
            rec["t_anchor_s"] = round(time.time() - t0, 3)
            signal.alarm(0)
            if X is None or X.shape != A.shape:
                raise RuntimeError("anchor missing or shape mismatch")
            model.float()

            tie = ref_scaled_count(A, X, ATOL_RTOL_STRICT, ATOL_RTOL_STRICT)
            rec["tie_out_count"] = tie
            rec["tie_out_ok"] = bool(rec["s76_s71_strict"] is not None and tie == rec["s76_s71_strict"])
            if not rec["tie_out_ok"]:
                rec["status"] = "tie_out_mismatch"
                excluded.append({"problem_id": pid, "name": fname,
                                 "reason": "tie-out %s != s76 %s" % (tie, rec["s76_s71_strict"])})
                out_rows.append(rec)
                with open(OUT, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                del A, X
                torch.cuda.empty_cache()
                continue

            a64 = A.double()
            del A

            signal.alarm(CFG_ALARM_S)
            t0 = time.time()
            S73.set_cfg(True, False)
            try:
                cmod = torch.compile(model, dynamic=False)
                C1 = S73.run_model(cmod, inputs, torch.float32, dev)
                rec["compile_ok"] = True
            except Exception as e:
                rec["compile_ok"] = False
                rec["compile_error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
                C1 = None
            rec["t_compile_s"] = round(time.time() - t0, 3)
            signal.alarm(0)
            if C1 is not None and C1.shape != a64.shape:
                raise RuntimeError("compile candidate shape mismatch")

            signal.alarm(CFG_ALARM_S)
            t0 = time.time()
            S73.set_cfg(False, False)
            C2 = S73.run_model(model, inputs, torch.float32, dev)
            rec["t_tf32off_s"] = round(time.time() - t0, 3)
            signal.alarm(0)
            if C2 is None or C2.shape != a64.shape:
                raise RuntimeError("tf32-off candidate missing or shape mismatch")

            for arm, C in (("c1_compile", C1), ("c2_tf32off", C2)):
                if C is None:
                    continue
                signal.alarm(CFG_ALARM_S)
                rec["decomp_" + arm] = S73.decomp(a64, X, C, ATOL_RTOL_STRICT, ATOL_RTOL_BENCH)
                signal.alarm(0)
            del a64, X, C1, C2
            rec["status"] = "ok"
        except S73.Timeout as e:
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
        brief = {"pid": pid, "st": rec.get("status"), "t": rec.get("t_task_s"), "tie": rec.get("tie_out_ok")}
        for arm in ("c1_compile", "c2_tf32off"):
            dd = rec.get("decomp_" + arm, {}).get("1e-4", {})
            brief[arm] = {"v": dd.get("verdict_default"), "n": dd.get("fail_default"),
                          "rs": dd.get("ref_share_of_fails"), "rmed": dd.get("ratio_ref_median")}
        print(json.dumps(brief), flush=True)
        torch.cuda.empty_cache()

    ok = [r for r in out_rows if r.get("status") == "ok"]

    def agg(arm, tag):
        rows = [r for r in ok if "decomp_" + arm in r]
        tasks_fail = sum(1 for r in rows if r["decomp_" + arm][tag]["verdict_default"] == "fail")
        tasks_fail_anchor = sum(1 for r in rows if r["decomp_" + arm][tag]["verdict_anchor"] == "fail")
        fails = sum(r["decomp_" + arm][tag]["fail_default"] for r in rows)
        refi = sum(r["decomp_" + arm][tag]["ref_induced"] for r in rows)
        candi = sum(r["decomp_" + arm][tag]["cand_induced"] for r in rows)
        bits = sum(1 for r in rows if r["decomp_" + arm][tag]["bit_identical_ca"])
        meds = sorted(r["decomp_" + arm][tag]["ratio_ref_median"] for r in rows
                      if r["decomp_" + arm][tag]["ratio_ref_median"] is not None)
        return {"n_tasks": len(rows), "tasks_fail_default": tasks_fail, "tasks_fail_anchor": tasks_fail_anchor,
                "fail_elements": fails, "ref_induced": refi, "cand_induced": candi,
                "pooled_ref_share_of_fails": (refi / fails) if fails else None,
                "bit_identical_tasks": bits,
                "ratio_ref_median_min": meds[0] if meds else None,
                "ratio_ref_median_max": meds[-1] if meds else None,
                "ratio_ref_median_span_x": (meds[-1] / meds[0]) if meds and meds[0] else None}

    summary = {"prereg_sha256": prereg_sha, "population": len(pop), "ok": len(ok),
               "excluded": excluded,
               "c1_compile": {"1e-4": agg("c1_compile", "1e-4"), "1e-2": agg("c1_compile", "1e-2")},
               "c2_tf32off": {"1e-4": agg("c2_tf32off", "1e-4"), "1e-2": agg("c2_tf32off", "1e-2")},
               "wall_s": round(time.time() - t_start, 1), "torch": torch.__version__, "kills": {}}
    s1 = summary["c1_compile"]["1e-4"]; s2 = summary["c2_tf32off"]["1e-4"]
    summary["kills"]["K1_vacuous"] = bool(s1["tasks_fail_default"] == 0 and s2["tasks_fail_default"] == 0)
    shares = [x for x in (s1["pooled_ref_share_of_fails"], s2["pooled_ref_share_of_fails"]) if x is not None]
    summary["kills"]["K2_ref_not_material"] = bool(shares and max(shares) < 0.05)
    summary["kills"]["K3_instrument"] = bool(len(ok) < 0.75 * len(pop))
    span = max([x for x in (s1["ratio_ref_median_span_x"], s2["ratio_ref_median_span_x"]) if x], default=None)
    summary["kills"]["K4_instance_span_100x"] = bool(span is not None and span >= 100)
    json.dump(summary, open(SUMOUT, "w"), indent=1)
    print(json.dumps(summary, indent=1)[:4000])


if __name__ == "__main__":
    main()
