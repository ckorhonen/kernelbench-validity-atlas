"""D3 preregistration -- conv margin decomposition + fp64-accum arm.

Frozen 2026-10-20 ~23:50Z, BEFORE any d3 measurement (tick #30, per
d2_conv_followup.md registered branches). Constants derive from committed
artifacts (results/d2_freeze.json, results/d2_conv.jsonl, guard_v1.json)
plus policy constants fixed here ex ante. NO d3 numbers are consulted;
the runner (lab/d3_margin.py) may not alter any constant below.

UNIVERSE: d2 final row set verbatim -- build_rows_contingency (ordmag
48 log-scales 1e0..1e8 x seeds 52,53) + d2 cancel arm (10 x 2) = 116
rows, cell C64_K576_Cout128_8x8_N4; tie-out asserted per
(arm,seed,si) on max_budget_ratio (rel 1e-9) vs results/d2_conv.jsonl.

METRICS (per element; ATOL=RTOL=1e-4 d2 label convention):
  ratio_i = |T_i-R_i| / (ATOL + RTOL*|R_i|);  fail_i := ratio_i > 1
  near_i  := |R_i| < 1e-3   [ex ante: rtol*|R| <= 1e-7 = 1e-3 x atol
             floor throughout the band => relative lens uninformative]
  PRIMARY   phi_cond    = P(near | fail), pooled over all 116 rows
  descriptive: phi_abs (P(near)), phi_argmax (near frac among per-row
    argmax-ratio elements), control phi_pass = P(near | !fail), and
    median |R| per side.

BRANCH RULE (frozen):  A "phi-heavy" iff phi_cond >= 0.50
                       B "flat"      otherwise
  A => fp64-accum is the registered remedy candidate (followup branch A).
  B => family exits: '1e-4 elementwise-all is not a conv-reassociation
  lens' negative numerics-axis entry; d3 fp64 result descriptive only.

ARM (same pass regardless of branch; role set by the branch rule):
  cand64 := d2 cand_fwd path in float64 (unfold/permute/reshape gemm +
  bias), result cast to fp32; label64 := elementwise-all vs fp32
  conv2d at ATOL=RTOL=1e-4.  remedy green iff all 116 rows label64;
  partial iff >= 50%; else fail.  T*_row := max_i d_i/(1+|R_i|) (exact
  minimal symmetric atol=rtol, no search) per arm for cand32/cand64 --
  descriptive lens axis, quantiles min/q25/q50/q75/q90/max.

DECLARED RE-EVALUATION (verbatim d2 verdict machinery, label64):
  if >= 1 arm two-class under label64: per-unit Youden t* (plateau
  midpoint), T_law = a_dag + gamma*d_cen + b_conv_pred, K-a/K-b/K-c/G,
  b_meas = mean t* over two-class units - a_dag, rider pass iff
  |b_meas - b_conv_pred| <= bound.  b_conv_pred/bound CARRY from
  d2_freeze (geometry unchanged => prediction carries, per #29 note).
  All units single-class => nulls recorded, no kill fires (d2 conv).

OUTPUT: results/d3_margin.jsonl (per-row) + d3_margin.json (verdict
block); freeze sha pins d2_freeze.json + d2_conv.jsonl + guard_v1.json.
Run the runner on spark ONLY after this file + d3_freeze.json are
committed (P3: harness commits carry data).
"""
import json, os, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

ATOL = RTOL = 1e-4            # d2 label convention, verbatim
NEAR = 1e-3                   # near-zero band, ex ante this tick
PHI_BRANCH = 0.50             # A iff phi_cond >= this
REMEDY_GREEN = 1.00           # frac rows label64 for "green"
REMEDY_PARTIAL = 0.50

def sha(p):
    return hashlib.sha256(open(os.path.join(RES, p), "rb").read()).hexdigest()

def main():
    fz = json.load(open(os.path.join(RES, "d2_freeze.json")))
    rows = [json.loads(l) for l in open(os.path.join(RES, "d2_conv.jsonl"))]
    arms = {}
    for r in rows:
        arms[r["arm"]] = arms.get(r["arm"], 0) + 1
    assert len(rows) == 116 and arms == {"ordmag": 96, "cancel": 20}, (len(rows), arms)
    n_elem = None
    for r in rows:
        if n_elem is None:
            n_elem = r["B"] * r["Hrole"]
        else:
            assert n_elem == r["B"] * r["Hrole"]
    out = dict(
        when__frozen="2026-10-20T23:50Z (before any d3 run); see module docstring",
        universe=dict(n_rows=len(rows), arms=arms, elem_per_row=n_elem,
                      cell=rows[0]["cell"]),
        atol=ATOL, rtol=RTOL, near_band=NEAR, phi_branch=PHI_BRANCH,
        remedy_green=REMEDY_GREEN, remedy_partial=REMEDY_PARTIAL,
        metrics=dict(primary="phi_cond=P(near|fail) pooled",
                     descriptive=["phi_abs", "phi_argmax", "phi_pass",
                                  "median_absR_fail", "median_absR_pass",
                                  "tstar_row_cand32", "tstar_row_cand64"]),
        branch_A="phi-heavy: fp64-accum remedy candidate; re-eval if two-class",
        branch_B="flat: family exits, negative numerics-axis entry; fp64 descriptive",
        cand64="d2 cand path float64 accum, cast fp32, label64 vs fp32 conv2d",
        carried=dict(b_conv_pred=fz["b_conv_pred_F1"], bound=fz["bound_2rmse"],
                     a_dag=fz["a_dag"], gamma=fz["gamma"],
                     t_frozen_d2=fz["t_frozen"]),
        reeval="d2 verdict machinery verbatim on label64; single-class => nulls",
        shas=dict(d2_freeze=sha("d2_freeze.json"), d2_conv=sha("d2_conv.jsonl"),
                  guard_v1=sha("guard_v1.json")),
        runner="lab/d3_margin.py (spark); tie-out rel 1e-9 vs d2_conv.jsonl",
    )
    json.dump(out, open(os.path.join(RES, "d3_freeze.json"), "w"), indent=1)
    print(json.dumps(out, indent=1)[:1200])

if __name__ == "__main__":
    main()
