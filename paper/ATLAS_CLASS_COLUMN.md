# ATLAS — attributable column, class granularity (s84)

Prereg: `lab/s84_class_column.py` docstring, commit `697f208`, frozen before the first computation.
Population: the 54 exceed rows of `s73c_merged_56.jsonl` (28 conv_transpose, 25 conv, 1 gemm) plus 8 controls;
engine class joined from `s75_engine_class_s73pop.jsonl` (47 TF32_TENSOROP, 7 FP32_SIMT_OR_OTHER).
CPU only, committed artifacts, no new measurement. Artifacts: `results/s84_class_column.json`,
`results/s84_class_column.csv` (108 arm-rows), `results/s84_class_column_rows.csv` (cell table).

## What the column can carry at class granularity

**The share is class-level.** Across the 53 rows where the tf32-off arm fails the 1e-4 default-reference
gate, 2,178,740,305 of 2,182,959,627 failing elements are reference-induced (99.807%), candidate-induced
4,219,322 (0.193%). Per-shape reference share: median 0.9993, minimum 0.7769. Frozen verdict
**SHARE_REFERENCE_DOMINATED** (pooled median 0.9993 >= 0.95; every n>=3 cell median >= 0.996). The single
material exception is L2/11 conv_transpose, 22.31% candidate-induced over 11.7M failing elements — named,
not smoothed.

**The magnitude is not.** Inside one engine class the admission-relevant tail spans more than a decade:
TF32_TENSOROP conv n=23 tail 1.14x -> 12.78x (spread 11.20x), conv_transpose n=24 tail 1.19x -> 23.68x
(spread 19.88x). The prereg's kill fired: **TAIL_PER_SHAPE**. No cell-level magnitude claim is made; class
labels annotate the share only.

**Draw noise is not the explanation.** For the 28 arm-rows on 14 probed tuples, median draw spread is
1.077x against a smallest cell spread of 11.2x (**DRAW_NOISE_BELOW_CELL_SPREAD**), so the within-class
spread is genuine shape dependence, not input sampling.

**The verdict is class-conditional on identical shapes.** 53/54 tasks: compile passes and tf32-off fails the
same 1e-4 default-reference gate. 27 of the 54 compile rows are bit-identical to the reference (|C-A| = 0),
so those passes are inherited by construction; all 27 still fail against the fp64 anchor. At the shipped
1e-2 gate compile passes 54/54 and tf32-off 53/54, and the one 1e-2 failure is candidate-induced (s73b).
Control gate: 8/8 controls at ref_over_budget_frac 0.0 (**CONTROL_GATE_PASS**).

## Deliverable shape

The column ships as: per-shape tail magnitude (max_ratio_ref, frac_over_ref), per-shape reference share,
engine class as an annotation, draw interval where probed, and a single class-level share constant for the
reference-dominated population. Per-arm enumeration is dropped (s83); the class key enters the share, never
the magnitude.

## s85 — the tail does not compress in geometry either

Prereg frozen pre-run (`lab/s85_tail_geometry.py`). Target: log10(max_ratio_ref) for the C1 arm on all 54 rows
(no exclusions), five declared shape-derived features, within-op LOO OLS, bar 0.50.
Result: **TAIL_GEOMETRY_FAILS** — conv LOO R2 -1.078 (n=25, in-sample 0.105), conv_transpose -0.374 (n=28,
in-sample 0.238); rank checks 0.418 and 0.064. The derived coordinates that failed to predict the median bias
(s74/s81) also fail to predict the admission-relevant tail, so the per-shape magnitude column is final for this
population and no geometry surface is claimed for either statistic.

## s86 — the binned reproducible key fails too (magnitude question parked)

Prereg frozen pre-run. Cell = engine class x op_class x output-extent band (<1e6, 1e6-1e8, >=1e8 elements), target
identical to s85, eta^2 bar 0.50 plus a 10x cell bound on cells with n>=5 and a >=3 qualifying-cell rule.
Result **TAIL_KEY_FAILS**: eta^2 0.358; two n>=5 cells exceed the bound (TF32 conv_transpose 1e6-1e8 14.56x,
>=1e8 15.41x). The one tight cell is TF32 conv at >=1e8 elements (n=18, 4.20x spread, log10 median 0.684).
Three keys have now failed for the tail (engine class, continuous geometry, binned key), so the magnitude question
is parked for this population and only a new population (fresh shapes, or a peer's margins) reopens it.

## s87/s87b/s87c — the share constant outside the 54 (generalizes; class axis untestable here)

Preregs frozen pre-run (`lab/s87_fresh_shape_share.py`, `s87b_share_class_join.py`, `s87c_winograd_share.py`;
commits a529342, 7da0a87, b01a461). This is the previous reflection's named follow-up: the share is the only
surviving compressible quantity, so the question worth spending effort on was whether it survives outside the
population it was measured on, not whether a fourth magnitude key exists.

s87: 17 constructed conv/convT configurations outside the 54, extent-stratified 10/4/3, seeds declared.
**V1_SHARE_GENERALIZES** — pooled 0.99850 (153,126,865 of 153,357,153 failing elements), every extent band
>= 0.9968, minimum per-configuration share 0.99573, maximum candidate-induced 0.43%; zero failures at the
shipped 1e-2 gate on all 17. So the constant is not an artifact of the benchmark's own shapes and holds from
4e5 to 1.05e8 output elements.

s87b: the reference call profiled on all 17 with s75's frozen patterns. **CLASS_CONFINED** — 17/17
TF32_TENSOROP (xmma/cutlass s1688 fprop and dgrad). The generalization is therefore across shape and extent
*inside* one engine class; the class axis stays open.

s87c: 8 small-extent configurations bracketing the A7 winograd regime, plus 2 declared TF32 controls.
**NO_NON_TF32_DISPATCH** — 3 configurations did label WINOGRAD (H=W=14), but all three are reference-clean
(ref_over_budget_frac exactly 0.0, fail_default 0), so no failing elements exist to split and the class axis
cannot be tested by extent alone. Reading: the exceedance population is *selected* by the reference being
TF32-dispatched, so "99.8% of failing elements are reference-induced" is a statement about reference-dominated
populations, not a law about kernels. Controls labelled TF32 2/2 (instrument gate clean).

Scope of the constant as it now stands: GB10 sm_120, rev 423217d, conv/convT at the benchmark's default flags,
TF32_TENSOROP dispatch, extents 4e5 to 2.1e9, one input draw per shape (s79a/s84 establish draw-stability at
1.08x-1.14x). Not established: any non-TF32 class, and the magnitude of the tail (per-shape only; the one tight
cell remains TF32 conv >=1e8, n=18, 4.20x, and is the only place a magnitude cell bound can be quoted).

Limits. Engine class is measured on this population and replicates at fresh-process scope (s72, 26/26), but
class membership does not bound magnitude; the tail bound would need a finer reproducible key, and the
obvious candidate (kernel-instance identity) is not reproducible across passes (s80c, 12/14). The gemm cell
has n=1 and carries no cell statistic. Resolution conventions (s62, s83) apply to any behavioural count.

## s88/s88b — the winograd class has no reference-side exceedance to split (population rule falsified; compile's inheritance is extent-conditional)

Prereg frozen pre-run (`lab/s88_class_share.py`, commit 1477773), population rule = the 19 rows of s63's 56-row
r1 population whose default-flags reference call s63 labelled WINOGRAD (13 m3m2, 3 w4, 3 a2, conv2d 3x3 s1 p1
fp32), plus 3 declared TF32 controls (first in file order). Constructed replication at the same tuples; per-row
instrument gate = the constructed conv must label WINOGRAD.

**The registered population rule is falsified.** All 19 tuples label WINOGRAD (19/19 gate clean) and all 19 are
reference-clean: frac_over_ref exactly 0.0, rho_max 0.047-0.300 (30-2000x inside the budget). There are no
reference-induced failing elements to split, so `wino/C2` has n_fail = 0 and the share is undefined (K2
UNDERPOWERED fires at <500 failing elements; K1's CI bound is degenerate for the same reason). The control gate
passes: the 3 TF32 rows reproduce s84 exactly (764/764 reference-induced, share 1.0, Wilson CI low 0.995).

**s88b (post-hoc, descriptive, no prereg).** A7's winograd row (2,5,5,256,256; rho 10.195, frac_over 0.0052)
does not replicate on a fresh draw: rho 0.149, frac_over 0.0; brackets (2,6,6,256,256) 0.147 and
(2,8,8,256,256) 0.225. A7's number was a near-zero-anchor extremum (its abs_max 0.00264 divided by a ~2.6e-4
anchor magnitude), not a bulk exceedance. So the winograd class is bulk-clean at 4e3-1e5 elements on two
independent populations (s87c extents, s88 r1 tuples), and the class axis still has no exceeded population to
test. The constant remains TF32-scoped, now with a stronger non-claim: no winograd-class share exists because no
winograd reference exceeds in bulk at these extents.

**New finding, and a correction to my own §4 claim.** On 4 of the 19 winograd tuples (the smallest extents:
1x4x4x256, 4x4x4x64, 4x4x4x256, 1x7x7x64) torch.compile's candidate FAILS the 1e-4 gate against the same
reference — 1,170-5,623 failing elements, 100% candidate-induced (d_ref <= b_e), and compile dispatches
TF32_TENSOROP on 2 of them, i.e. it is NOT class-sharing there. s88b reproduces the direction at larger
small-extents (4,459-12,318 candidate-induced failures). So "compile inherits a pass" is extent-conditional:
it holds on the 54-row exceed population (27/54 bit-identical) and fails where the inductor conv picks a
different engine class than the winograd reference. tf32-off, by contrast, stays in the winograd class on
19/19 and passes 19/19 (0 failing elements).

Deliverable shape unchanged (per-shape magnitude, class-level share), with one added caveat: a candidate's
verdict inheritance depends on whether it shares the reference's engine class, and the compiler arm does not
share it at small extents.
