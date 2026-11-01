#!/usr/bin/env python
"""s78b_l176_decomp.py — AMENDMENT v2: candidate decomposition on L1/76 alone, the one strict 1e-4 exceedance
in the s76 coverage closure. Prereg frozen before this run; s78's population and gate are unchanged.

WHY THE AMENDMENT (declared pre-run, measurement-neutral): s78 ran the 8 s71 anchor-inadmissible rows and its
tie-out gate excluded L1/76 because my recomputed A-vs-anchor strict count (263,654,458) did not equal the
s76 record field tie_out.s71_strict (263,654,070). But 263,654,458 IS s76's own reported A_vs_anchor.ew_1e-4
count for that task -- so s78 reproduced s76's headline number exactly and the gate compared against a
different field, not against a different measurement. The 388-element difference between those two fields is
itself a bookkeeping defect worth resolving, so this run computes the exceedance count under THREE scaling
conventions (|A|-scaled, |X|-scaled, |C|-scaled) and reports which one reproduces each recorded number.
No threshold, arm, population or kill clause changes; the only new object is that three-way count.

Population: L1/76 conv_standard_1D_dilated_strided (output 64x128x174758 = 1,431,617,536 elements).
Arms on one model instance and one input draw: A default fp32 reference (cudnn tf32 on, matmul tf32 off),
X fp64 anchor, C1 torch.compile(dynamic=False) fp32 default flags, C2 fp32 tf32 off.
Predicates: evaluator-exact (b_e = atol + rtol*|candidate|); legs 1e-4 and 1e-2.

Tie-out: the run is accepted if the |X|-scaled count equals 263,654,458 (s76's reported A_vs_anchor) OR the
|A|-scaled count equals 263,654,070 (s76's tie_out.s71_strict). Both are recorded either way; a run matching
neither is a harness defect and the task is reported as such.

Reads (recorded, no threshold): does compile inherit a pass at 1e-4 (engine-class sharing), does tf32-off fail
with ~100% reference-induced share (s73's pattern), and does the shipped 1e-2 gate clean the population.
Kill clauses: K1 vacuous (no arm fails at 1e-4), K2 reference-induced share <5% at 1e-4, K5 harness (neither
count reproduced). K3/K4 from s78 are not repeated here (single task).
"""
import importlib.util, json, os, signal, time, hashlib
import torch

KB = os.path.expanduser("~/kernelbench/KernelBench")
LAB = os.path.expanduser("~/kb-lab")
OUT = os.path.join(LAB, "results", "s78b_l176_decomp.json")
S71 = os.path.join(LAB, "results", "s71_full_audit.jsonl")
S76 = os.path.join(LAB, "results", "s76_anchor_coverage.jsonl")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

ATOL = RTOL = 1e-4
ATOL_B = RTOL_B = 1e-2
CFG_ALARM_S = 420
CHUNK_ELEMS = 1 << 22
REC_S76_A = 263654458
REC_S76_TIEOUT = 263654070


def chunk_rows(cols):
    return max(1, CHUNK_ELEMS // max(1, cols))


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S73 = load_mod("s73mod", os.path.join(os.path.dirname(os.path.abspath(__file__)), "s73_cand_ref_decomp.py"))


def counts_three_way(A, X, C):
    """Exceedance counts for |A-X| under three scaling conventions, plus |C-A| and |C-X| under |C| scaling."""
    a64 = A.double(); x64 = X.double(); c64 = C.double()
    rows, cols = a64.reshape(-1, a64.shape[-1]).shape
    cr = chunk_rows(cols)
    fa = a64.reshape(rows, cols); fx = x64.reshape(rows, cols); fc = c64.reshape(rows, cols)
    out = {"ax_by_A": 0, "ax_by_X": 0, "ax_by_C": 0, "ca_by_C": 0, "cx_by_C": 0,
           "ca_by_C_1e-2": 0, "cx_by_C_1e-2": 0, "refover_by_C": 0, "ref_induced": 0, "cand_induced": 0,
           "maxabs_ca": 0.0, "maxabs_ax": 0.0, "bit_identical_ca": True, "total": int(rows * cols)}
    for i in range(0, rows, cr):
        ac = fa[i:i + cr]; xc = fx[i:i + cr]; cc = fc[i:i + cr]
        d_ax = (ac - xc).abs(); d_ca = (cc - ac).abs(); d_cx = (cc - xc).abs()
        out["ax_by_A"] += int((d_ax > ATOL + RTOL * ac.abs()).sum())
        out["ax_by_X"] += int((d_ax > ATOL + RTOL * xc.abs()).sum())
        thr = ATOL + RTOL * cc.abs()
        out["ax_by_C"] += int((d_ax > thr).sum())
        fail_d = d_ca > thr
        refover = d_ax > thr
        out["ca_by_C"] += int(fail_d.sum())
        out["cx_by_C"] += int((d_cx > thr).sum())
        out["refover_by_C"] += int(refover.sum())
        out["ref_induced"] += int((fail_d & refover).sum())
        out["cand_induced"] += int((fail_d & ~refover).sum())
        thrb = ATOL_B + RTOL_B * cc.abs()
        out["ca_by_C_1e-2"] += int((d_ca > thrb).sum())
        out["cx_by_C_1e-2"] += int((d_cx > thrb).sum())
        out["maxabs_ca"] = max(out["maxabs_ca"], float(d_ca.max()))
        out["maxabs_ax"] = max(out["maxabs_ax"], float(d_ax.max()))
        if float(d_ca.max()) > 0:
            out["bit_identical_ca"] = False
        del ac, xc, cc, d_ax, d_ca, d_cx, thr, thrb, fail_d, refover
    out["verdict_default"] = "fail" if out["ca_by_C"] > 0 else "pass"
    out["verdict_anchor"] = "fail" if out["cx_by_C"] > 0 else "pass"
    out["ref_share_of_fails"] = (out["ref_induced"] / out["ca_by_C"]) if out["ca_by_C"] else None
    del a64, x64, c64, fa, fx, fc
    return out


def main():
    torch.cuda.init()
    dev = "cuda"
    t0_all = time.time()
    prereg_sha = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    rows76 = [json.loads(l) for l in open(S76)]
    r76 = [r for r in rows76 if r["problem_id"] == 76][0]
    rec = {"problem_id": 76, "name": r76["name"], "output_shape": r76["output_shape"],
           "s76_reported_A_vs_anchor": r76["A_vs_anchor"]["ew_1e-4"]["count"],
           "s76_tieout_s71_strict": r76["tie_out"]["s71_strict"],
           "prereg_sha256": prereg_sha}
    path = os.path.join(KB, "KernelBench", "level1", r76["name"])
    torch.manual_seed(0)
    mod = S73.load_task(path)
    init_args = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
    inputs = mod.get_inputs()
    torch.manual_seed(0)
    model = mod.Model(*init_args)

    signal.alarm(CFG_ALARM_S)
    S73.set_cfg(True, False)
    A = S73.run_model(model, inputs, torch.float32, dev)
    signal.alarm(0)
    signal.alarm(CFG_ALARM_S)
    S73.set_cfg(False, False)
    X = S73.run_model(model, inputs, torch.float64, dev)
    signal.alarm(0)
    model.float()
    if A is None or X is None or X.shape != A.shape:
        raise RuntimeError("reference/anchor missing or shape mismatch")

    signal.alarm(CFG_ALARM_S)
    S73.set_cfg(True, False)
    try:
        C1 = S73.run_model(torch.compile(model, dynamic=False), inputs, torch.float32, dev)
        rec["compile_ok"] = True
    except Exception as e:
        rec["compile_ok"] = False
        rec["compile_error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
        C1 = None
    signal.alarm(0)
    signal.alarm(CFG_ALARM_S)
    S73.set_cfg(False, False)
    C2 = S73.run_model(model, inputs, torch.float32, dev)
    signal.alarm(0)

    for arm, C in (("c1_compile", C1), ("c2_tf32off", C2)):
        if C is None:
            continue
        signal.alarm(CFG_ALARM_S)
        rec["three_" + arm] = counts_three_way(A, X, C)
        signal.alarm(0)
        torch.cuda.empty_cache()

    t = rec.get("three_c2_tf32off", {})
    rec["tie_out_ok"] = bool(t.get("ax_by_X") == REC_S76_A or t.get("ax_by_A") == REC_S76_TIEOUT)
    s1 = rec.get("three_c1_compile", {}); s2 = rec.get("three_c2_tf32off", {})
    rec["kills"] = {
        "K1_vacuous": bool(s1.get("ca_by_C", 0) == 0 and s2.get("ca_by_C", 0) == 0),
        "K2_ref_not_material": bool(max([x for x in (s1.get("ref_share_of_fails"), s2.get("ref_share_of_fails"))
                                        if x is not None], default=0) < 0.05),
        "K5_harness": bool(not rec["tie_out_ok"]),
    }
    rec["wall_s"] = round(time.time() - t0_all, 1)
    rec["torch"] = torch.__version__
    json.dump(rec, open(OUT, "w"), indent=1)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
