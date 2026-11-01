#!/usr/bin/env python
"""s73b_oom_rerun.py — complete the s73 candidate x reference decomposition on the four OOM tasks.

PREREG (frozen before the run; do not edit after).

Deviation from s73 declared here, measurement-neutral: s73 held the default reference, the fp64 anchor and
both candidate outputs simultaneously and materialized full-size fp64 copies (a64, c64) inside decomp(),
which exceeded the 121 GB unified pool on four tasks (level1/63, level2/87, level1/21, level1/33). This
producer keeps the SAME arms, predicates, tolerance legs and kill clauses but (a) streams the decomposition
chunk-by-chunk from the fp32 outputs so no full-size fp64 copy is ever materialized, and (b) releases each
candidate immediately after its decomposition. The tie-out gate and every quantity are computed by the same
formulas as s73; no threshold, population or predicate changed.

Population (rule): exactly the four problem_ids named in results/s73_cand_ref_decomp_summary.json
["population"]["excluded"], all of kind "exceed". Merged with s73's 52 completed exceed rows this yields the
56-row set.

Arms per task, one model instance, one input draw, seed 0 (identical to s73):
  A   default fp32 reference (cudnn tf32 on, matmul tf32 off)
  X   fp64 anchor (model.to(float64) from the same fp32 weights)
  C1  torch.compile(model, dynamic=False), fp32, default flags
  C2  same model, fp32, tf32 off

Predicates: evaluator-exact, b_e = atol + rtol*|candidate| (eval.py:804), at 1e-4 and 1e-2.
Quantities: ref_budget_share distribution (median/q90/q99/max/frac over budget), verdict_default,
verdict_anchor, failing-element split (reference-induced = d_ref > b_e, candidate-induced otherwise),
bit-identity C vs A, compile wall.

TIE-OUT GATE: chunked recompute of s71's recorded A_vs_anchor count under s71's reference-scaled predicate
(thr = atol + rtol*|anchor|) must equal the recorded count exactly; otherwise the task is excluded as
tie_out_mismatch and no new number from it is used.

Kill clauses carried verbatim from s73: K1 no candidate arm fails any task at 1e-4 => vacuous; K2 pooled
reference-induced share < 5% => reference not material; K3 instrument if compile fails >25% of attempts.
Controls: none new (s73's 6 controls stand); this run adds no control task.

Memory protocol: A, X, C1 decomp, del C1, empty_cache, C2 decomp, del C2, empty_cache; decomp iterates
chunks of CHUNK elements and never materializes a full fp64 copy of A or C. A per-task SIGALRM guards a
single configuration; a global deadline bounds the run.
"""
import importlib.util, json, os, re, signal, time, hashlib
import torch

KB = os.path.expanduser("~/kernelbench/KernelBench")
LAB = os.path.expanduser("~/kb-lab")
OUT = os.path.join(LAB, "results", "s73b_oom_rerun.jsonl")
SUMOUT = os.path.join(LAB, "results", "s73b_oom_rerun_summary.json")
S73SUM = os.path.join(LAB, "results", "s73_cand_ref_decomp_summary.json")
S73ROWS = os.path.join(LAB, "results", "s73_cand_ref_decomp.jsonl")
S71 = os.path.join(LAB, "results", "s71_full_audit.jsonl")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

ATOL_RTOL_STRICT = 1e-4
ATOL_RTOL_BENCH = 1e-2
CFG_ALARM_S = 600
GLOBAL_DEADLINE_S = 1800
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


def _rows_cols(t):
    if t.ndim == 0:
        return 1, 1
    return t.reshape(-1, t.shape[-1]).shape


def ew_count_ref_scaled_chunked(a, b, atol, rtol):
    """s71 predicate thr = atol + rtol*|b| (b = anchor). Chunked: a is fp32, b is fp64."""
    rows, cols = _rows_cols(a)
    af = a.reshape(rows, cols) if a.ndim else a.reshape(1, 1)
    bf = b.reshape(rows, cols) if b.ndim else b.reshape(1, 1)
    cnt = 0
    for i in range(0, rows, CHUNK):
        ac = af[i:i + CHUNK].double()
        bc = bf[i:i + CHUNK]
        d = (ac - bc).abs()
        thr = atol + rtol * bc.abs()
        cnt += int((d > thr).sum())
    return cnt


def decomp_stream(a, x, c, atol_s, atol_b):
    """Streaming decomposition, identical quantities to s73's decomp(). a, c fp32; x fp64."""
    rows, cols = _rows_cols(a)
    af = a.reshape(rows, cols) if a.ndim else a.reshape(1, 1)
    xf = x.reshape(rows, cols) if x.ndim else x.reshape(1, 1)
    cf = c.reshape(rows, cols) if c.ndim else c.reshape(1, 1)
    out = {}
    for tag in ("1e-4", "1e-2"):
        out[tag] = {"fail_default": 0, "fail_anchor": 0, "ref_induced": 0, "cand_induced": 0,
                    "ref_over_budget": 0, "total": int(rows * cols), "maxabs_ca": 0.0,
                    "maxabs_cx": 0.0, "maxabs_ax": 0.0, "max_ratio_ref": 0.0, "max_ratio_cand": 0.0,
                    "ratio_samples": [], "bit_identical_ca": True}
    for i in range(0, rows, CHUNK):
        ac = af[i:i + CHUNK].double()
        xc = xf[i:i + CHUNK]
        cc = cf[i:i + CHUNK].double()
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
            if float(d_ca.max()) > 0:
                s["bit_identical_ca"] = False
            if len(s["ratio_samples"]) < RATIO_CAP:
                s["ratio_samples"].extend(r_ref.reshape(-1)[:RATIO_CAP - len(s["ratio_samples"])].tolist())
        del ac, cc, d_ca, d_cx, d_ax
    for tag in out:
        s = out[tag]
        smp = sorted(s.pop("ratio_samples"))

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

    s73 = json.load(open(S73SUM))
    excluded = s73["population"]["excluded"]
    ids = {e["problem_id"] for e in excluded}
    rows71 = [json.loads(l) for l in open(S71)]
    pop = [r for r in rows71 if r["problem_id"] in ids and r.get("exceed_strict_default")]
    print("population", len(pop), sorted(ids), flush=True)

    out_rows = []
    for r71 in pop:
        level, pid, fname = r71["level"], r71["problem_id"], r71["name"]
        path = os.path.join(KB, "KernelBench", "level%d" % level, fname)
        rec = {"kind": "exceed", "level": level, "problem_id": pid, "name": fname,
               "op_class": r71.get("op_class"), "s71_strict_count": r71["A_vs_anchor"]["ew_1e-4"]["count"],
               "s71_output_shape": r71.get("output_shape"), "s71_ref_max_abs": r71.get("ref_max_abs"),
               "producer": "s73b_oom_rerun (streaming decomp, sequential release)"}
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
            torch.cuda.empty_cache()

            tie = ew_count_ref_scaled_chunked(A, X, ATOL_RTOL_STRICT, ATOL_RTOL_STRICT)
            rec["tie_out_count"] = tie
            rec["tie_out_ok"] = bool(tie == rec["s71_strict_count"])
            if not rec["tie_out_ok"]:
                rec["status"] = "tie_out_mismatch"
                out_rows.append(rec)
                with open(OUT, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(json.dumps({"pid": pid, "status": rec["status"]}), flush=True)
                del A, X
                torch.cuda.empty_cache()
                continue

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
            if C1 is not None:
                signal.alarm(CFG_ALARM_S)
                rec["decomp_c1_compile"] = decomp_stream(A, X, C1, ATOL_RTOL_STRICT, ATOL_RTOL_BENCH)
                signal.alarm(0)
            del C1
            torch.cuda.empty_cache()

            # C2: tf32-off candidate
            signal.alarm(CFG_ALARM_S)
            t0 = time.time()
            set_cfg(False, False)
            C2 = run_model(model, inputs, torch.float32, dev)
            rec["t_tf32off_s"] = round(time.time() - t0, 3)
            signal.alarm(0)
            if C2 is None or C2.shape != A.shape:
                raise RuntimeError("tf32-off candidate missing or shape mismatch")
            signal.alarm(CFG_ALARM_S)
            rec["decomp_c2_tf32off"] = decomp_stream(A, X, C2, ATOL_RTOL_STRICT, ATOL_RTOL_BENCH)
            signal.alarm(0)
            del A, X, C2
            torch.cuda.empty_cache()
            rec["status"] = "ok"
        except Timeout as e:
            rec["status"] = "timeout"
        except Exception as e:
            rec["status"] = "error:%s: %s" % (type(e).__name__, str(e)[:200])
        finally:
            signal.alarm(0)
        rec["t_task_s"] = round(time.time() - t_task, 2)
        out_rows.append(rec)
        with open(OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        brief = {"pid": pid, "st": rec.get("status"), "t": rec.get("t_task_s"),
                 "tie": rec.get("tie_out_ok")}
        for arm in ("c1_compile", "c2_tf32off"):
            dd = rec.get("decomp_" + arm, {}).get("1e-4", {})
            brief[arm] = {"v": dd.get("verdict_default"), "n": dd.get("fail_default"),
                          "rs": dd.get("ref_share_of_fails")}
        print(json.dumps(brief), flush=True)
        torch.cuda.empty_cache()

    ok = [r for r in out_rows if r.get("status") == "ok"]
    new_ids = {r["problem_id"] for r in ok}

    # merged table: s73's completed exceed rows + these
    merged = [json.loads(l) for l in open(S73ROWS)]
    merged = [r for r in merged if r.get("status") == "ok" and r["kind"] == "exceed"]
    merged += [r for r in ok]
    with open(os.path.join(LAB, "results", "s73c_merged_56.jsonl"), "w") as fh:
        for r in merged:
            fh.write(json.dumps(r) + "\n")

    def agg(rows, arm, tag):
        sel = [r["decomp_" + arm][tag] for r in rows if "decomp_" + arm in r]
        tot_fail = sum(r["fail_default"] for r in sel)
        tot_ref = sum(r["ref_induced"] for r in sel)
        tot_cand = sum(r["cand_induced"] for r in sel)
        meds = sorted(x for x in (r["ratio_ref_median"] for r in sel) if x is not None)
        return {"tasks": len(sel), "tasks_fail_default": sum(1 for r in sel if r["verdict_default"] == "fail"),
                "tasks_fail_anchor": sum(1 for r in sel if r["verdict_anchor"] == "fail"),
                "verdict_flips_ref_swap": sum(1 for r in sel if r["verdict_default"] != r["verdict_anchor"]),
                "failing_elements_default": tot_fail, "ref_induced_elements": tot_ref,
                "cand_induced_elements": tot_cand,
                "pooled_ref_share_of_fails": (tot_ref / tot_fail) if tot_fail else None,
                "pooled_cand_share_of_fails": (tot_cand / tot_fail) if tot_fail else None,
                "median_of_task_ratio_ref_median": meds[len(meds) // 2] if meds else None,
                "median_of_task_ratio_ref_q90": (lambda v: v[len(v) // 2] if v else None)(
                    sorted(x for x in (r["ratio_ref_q90"] for r in sel) if x is not None)),
                "median_of_task_ratio_ref_q99": (lambda v: v[len(v) // 2] if v else None)(
                    sorted(x for x in (r["ratio_ref_q99"] for r in sel) if x is not None)),
                "max_ratio_ref": max((r["max_ratio_ref"] for r in sel), default=None),
                "tasks_with_ref_over_budget": sum(1 for r in sel if r["ref_over_budget"] > 0),
                "tasks_bit_identical": sum(1 for r in sel if r.get("bit_identical_ca"))}

    summary = {
        "prereg": "docstring of this file, frozen pre-run (streaming memory deviation declared)",
        "prereg_sha256": prereg_sha,
        "population": {"target_ids": sorted(ids), "attempted": len(out_rows), "ok": len(ok),
                       "ok_ids": sorted(new_ids),
                       "failed": [{"problem_id": r["problem_id"], "status": r.get("status")}
                                  for r in out_rows if r.get("status") != "ok"]},
        "new_rows": {"c1_compile": {"1e-4": agg(ok, "c1_compile", "1e-4"), "1e-2": agg(ok, "c1_compile", "1e-2")},
                     "c2_tf32off": {"1e-4": agg(ok, "c2_tf32off", "1e-4"), "1e-2": agg(ok, "c2_tf32off", "1e-2")}},
        "merged_56": {"n_exceed": len(merged),
                      "c1_compile": {"1e-4": agg(merged, "c1_compile", "1e-4"),
                                     "1e-2": agg(merged, "c1_compile", "1e-2")},
                      "c2_tf32off": {"1e-4": agg(merged, "c2_tf32off", "1e-4"),
                                     "1e-2": agg(merged, "c2_tf32off", "1e-2")}},
        "tie_out": {"n_ok": sum(1 for r in ok if r.get("tie_out_ok")), "n_rows": len(ok)},
        "wall_s": round(time.time() - t_start, 1), "torch": torch.__version__, "kills": {},
    }
    s1 = summary["merged_56"]["c1_compile"]["1e-4"]
    s2 = summary["merged_56"]["c2_tf32off"]["1e-4"]
    summary["kills"]["K1_vacuous"] = bool(s1["tasks_fail_default"] == 0 and s2["tasks_fail_default"] == 0)
    shares = [x for x in (s1["pooled_ref_share_of_fails"], s2["pooled_ref_share_of_fails"]) if x is not None]
    summary["kills"]["K2_ref_not_material"] = bool(shares and max(shares) < 0.05)
    comp_att = len([r for r in out_rows if "compile_ok" in r])
    comp_ok = sum(1 for r in out_rows if r.get("compile_ok"))
    summary["kills"]["K3_compile_coverage"] = bool(comp_att > 0 and comp_ok < 0.75 * comp_att)
    json.dump(summary, open(SUMOUT, "w"), indent=1)
    print(json.dumps(summary, indent=1)[:6000])


if __name__ == "__main__":
    main()
