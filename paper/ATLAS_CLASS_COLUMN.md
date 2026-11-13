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

## s89/s89b — where the tf32-off reference itself exceeds: the share is a coverage ratio, not a class constant

Prereg v1 `lab/s89_tf32off_exceed_attrib.py` @ d357139 (frozen pre-run); amendment v2 `lab/s89b_tf32off_attrib_v2.py`
@ 2b22a4f (frozen before the v2 run, v1 artifacts kept). Population from `results/s71_full_audit.jsonl`:
targets = the 2 rows with `exceed_strict_tf32off` (L2/11 convT composite, L2/14 gemm composite); controls =
first two file-order rows with `exceed_strict_default` true and tf32-off clean (L2/54 conv3d, L2/55 conv2d).

**v1 produced no attribution, and said so.** Two instrument slips were caught before any share was quoted: the
fp64 anchor call left the model in fp64 (`run_model` mutates dtype) so every compile attempt died on a dtype
mismatch, and the ceiling was coded over all elements instead of over failing elements. v2 fixes both; no
threshold, population or selection rule changed.

**v2 results (spark, 4 rows, ~20 s; `results/s89b_tf32off_attrib.{jsonl,summary.json}`).**
- *Premise:* only L2/11 shows a measured class change (A TF32_TENSOROP -> B WINOGRAD). L2/14 is same-class
  FP32_SIMT and B is bit-identical to A, so its 4/1024 exceedance is an ordering effect, not a class test.
- *L2/11:* A is over budget on 54.95% of 18.94M elements (maxabs 1.89e-3); B on 26.36% (maxabs 2.78e-2,
  rho 58.66). compile vs B: 11,728,917 failing (61.9%), 4,085,381 reference-induced, share 0.3483
  (CI [0.3480, 0.3486]), ceiling 0.4256, share_ratio 0.8184. compile vs A: 0 failing elements and not
  bit-identical. B-as-candidate vs A: 11,728,901 failing, share 0.7769, ceiling 0.8874, ratio 0.8756.
- *Controls:* B is clean (frac_over exactly 0.0, maxabs 8.9e-7 / 2.7e-6). B-as-candidate vs A reproduces
  s84's constant on a fresh population: 0.99968 and 0.99914 (ratios 0.99968 / 0.99914). compile vs A: 0 failing
  (bit-identical on L2/54). compile vs B: 42.6M / 97.7M failing, 0% reference-induced.
- *Instrument:* in-process repeat control bit-identical and count-stable on 4/4 rows, so v1's s71 tie-out
  mismatch (19 elements of 4.99M, 3.8e-6 relative) is cross-process algorithm/allocator variability, not
  harness instability.

**Frozen verdict: BOUNDED_BY_REFERENCE_COVERAGE** (not SHARE_HOLDS, not CLASS_SCOPED). The 0.9985 figure is not
a class property; it is the coverage ratio |fail ∩ {reference over budget}| / |fail|, and it approaches 1 only
where the reference's exceedance set covers essentially the whole candidate failure set. The column therefore
carries share together with ceiling and share_ratio, and the constant's scope is "references whose exceedance
covers ~the whole failure set", not "TF32 references". Second finding: compile's pass inheritance is
reference-conditional, not merely extent-conditional — on L2/11 compile (engine OTHER) sits within budget of the
default reference on all 18.94M elements while differing from it, and fails 61.9% of elements against the
tf32-off reference.

## s90 — reference-configuration certificate at the SHIPPED gate (2026-11-12)

Question: every reference-axis number I own is at 1e-4. The benchmark ships 1e-2. On 24 fresh shapes (conv, convT, gemm,
layer_norm, reduction, softmax; disjoint from the 54-task population and s87's 17 configs), default reference vs an fp64
anchor, evaluator-exact direction:

- Shipped 1e-2, benchmark-realistic weight scale (1/sqrt(fan_in)): **0/24 exceed**. Max budget fraction 0.174 (convT
  3x3 s1 89x89), frac_over 0.0 everywhere, exact candidate fails nowhere. Non-conv max 0.0105 (gemm), 5.7e-4 (LN),
  4.3e-3 (reduction), 7e-8 (softmax). Verdict CERT_STRICT_GATE_DIAGNOSTIC, conv-only scope.
- Strict 1e-4, same population: 11/12 conv+convT configs exceed (frac .320-.614, rho 7.5-17.4) and one gemm at 1.7e-6;
  tf32-off arm clean (max frac 1.7e-6). The exceedance is TF32-class and replicates s73/s84 on fresh shapes.
- Scale arm (same shapes/seeds, raw randn weights): 11/24 conv/convT exceed at 1e-2, max rho 7.49, frac up to 0.0091;
  non-conv unchanged. So shipped-gate safety is governed by |A|max against atol (2.2e-4*|A|max vs 1e-2), a scale
  coordinate, not geometry. The declared test agreed 18/24 (over-predicts for FP32_SIMT gemm).

Validity domain of the certificate: GB10 sm_120, fp32, default cudnn-TF32 reference, conv/convT with benchmark-style
weight normalization. Scope claim: the benchmark's 0/200 at 1e-2 is conditional on its own weight normalization; the
reference axis is a strict-gate diagnostic at benchmark scale. Non-claim: no shape law; magnitude stays enumerated.

## s91 — the scale coordinate becomes a boundary, and the boundary is gate-specific (2026-11-12)

s90's scale finding was a two-point comparison, so it had a scope but no boundary. s91 swept the weight scale g over
{0.0625..64} on two fixed conv tuples from s90's population (2.92M and 14.25M output elements; 4.88x apart), with the
benchmark-realistic normalization multiplied by g, same seed and evaluator-exact predicate. Prereg `e390322` committed
pre-run; one declared post-hoc amendment extends the grid downward because its floor sat above one tuple's strict-gate
crossing (grid-layout defect, not outcome-driven). 62 rows, all ok; tie-out to s90's norm-arm rho at g=1 within 1.8e-4
and 2.7e-4 relative; repeat bit-identical 62/62; zero kill clauses fired.

- **Monotone:** log-log slope of rho vs g is 0.965 / 0.948, max log residual 4.5% / 4.1%, no monotonicity violation.
- **Crossing rho=1, 1e-4:** T1 g=0.0639 |A|max 0.308 u=0.678; T2 g=0.0599 |A|max 0.314 u=0.690 (cross-tuple 1.017).
- **Crossing rho=1, 1e-2 (shipped):** T1 g=8.354 |A|max 40.33 u=0.887; T2 g=7.336 |A|max 38.38 u=0.844 (1.051).
- **Gate shift:** per-tuple cross-gate ratio 1.308 / 1.224 => the boundary collapses across tuples at a FIXED gate but
  is gate-specific; governing verdict BRANCH_GATE_ONLY (the frozen summary printed BRANCH_COLLAPSE_ATOL through a
  min/max pairing across gates — coding slip recorded, no threshold changed). The atol-floor predictor u=1 survives
  only as an upper bound, over-predicting by 12-47%.
- **Mechanism falsified:** the argmax element is not near-zero. |x|/atol is 5.6 / 17.2 / 6.9 / 1.3 with relative error
  16.5% / 5.8% / 15.4% / 74.7% — a cancelled output whose absolute error comes from the accumulation-term scale.
- **Class/population:** tf32-off clean at every crossing (rho <=3.2e-3); every crossing tuple TF32_TENSOROP. frac_over
  is exactly 0.0 at all four crossings; at 1e-2 it stays below 0.01 up to |A|max 309, while allclose fails the tensor
  on one element. Extremum and population boundaries are an order of magnitude apart in scale.

Column note for the deliverable: the reference-conditional column now carries a boundary, not just a diagnostic —
"reference-clean" must name the statistic (allclose vs population fraction) and the gate; at the shipped gate the
reference fails on a single element once |A|max exceeds ~38-40, and benchmark-normalized conv outputs sit at 4.8-5.2,
i.e. 7.7-8.4x inside that boundary. Scope: conv/convT, GB10 sm_120, one draw per point, two tuples.

## s92 — the population boundary is a plateau, and admission/usability are separate surfaces (2026-11-13)

s91 measured the extremum boundary but left frac_over below 0.01 at the top of its sweep. s92 extends the same sweep
upward on the same two tuples (prereg committed pre-run, sha 2a4b06e1c5aa; harness imported from s91 verbatim, so the
tie-out is bit-level: 5.6e-6 / 6.5e-6 relative on rho and 1.5e-8 / 2.1e-7 absolute on frac_over at g=64; repeat
bit-identical 22/22; frac_over monotone with zero violations).

- **1% population boundary located, and it collapses across tuples:** T1 g=76.91, |A|max 371.3, u=8.17; T2 g=79.38,
  |A|max 415.3, u=9.14 (ratios 1.032 / 1.119 across a 4.88x element-count difference). The frozen summary took its
  FRAC_BOUNDARY_PARTIAL branch because the 0.5 level never occurred, so the collapse clause was computed post-hoc
  from committed rows in `results/s92_analysis.json` and labelled as such.
- **The boundary is a plateau, not a threshold:** frac_over rises from 0.0087 at g=64 to 0.0184 / 0.0183 at
  g=4096 (|A|max 19,775 / 21,431); the last 4x in g adds 6.1% / 6.0% of the total rise while rho grows 7.6 -> 327
  (log-log slope 0.904 / 0.924). The 0.5 level does not exist in range — the prereg's implicit expectation that the
  fraction would climb toward 1 is falsified by its own grid.
- **Mechanism measured:** at g=4096 the evaluator predicate and a pure-relative predicate agree (0.01842 vs 0.01861
  T1; 0.01831 vs 0.01851 T2), so atol is irrelevant there and the plateau is the reference's relative-error tail
  P(rel_err > 1e-2) = 1.83-1.84% (p50 2.9e-4, p99 1.86e-2, p99.9 0.184). The extremum is a different subpopulation:
  the argmax element sits at 4.2e-7 / 4.6e-7 of the median magnitude with absolute error ~1.15 / 0.66, i.e. deep
  cancellation against the accumulation-term scale.
- **Class control:** tf32-off frac_over exactly 0.0 and rho <=0.012 at both crossings; both tuples TF32_TENSOROP.

Column note: the column now carries two statistic-specific coordinates at the shipped gate — admission (allclose,
u*~0.87) and usability (1% population, u*~9) — and a measured ceiling of ~1.85% for the failing fraction. Scope:
conv/convT, GB10 sm_120, one draw per point, two tuples.

## s93 — the plateau's domain: a shape-independent TF32 tail constant, with a measured low-scale regime (2026-11-13)

s92 measured the plateau at one scale per tuple, so "the predicate is relative here" and "the relative tail is
scale-invariant" were indistinguishable. s93 separates them (prereg `8dfa114` frozen pre-run; a mechanical import
fix `454b5d7` landed before any row existed; tie-out to s92's g=4096 rows at 1e-7 absolute on frac_over and 1.3e-6
relative on rho; repeat bit-identical 10/10; no kill clause fired).

- **Relative-error distribution is exactly scale-invariant under power-of-two rescaling:** p50 2.9243e-4, p99
  1.862e-2, p99.9 0.1842 bit-identical at g=16/1024/4096 on T1 (2.9190e-4 / 1.8524e-2 / 0.1843 on T2). fp32
  scaling by 2^k is exponent-exact, so a TF32 accumulation's relative error cannot move. At g=1e5 the same
  quantiles shift 0.02%, consistent with weight rounding.
- **The constant is shape-independent within the class:** frac_over 0.018567 (T1, 2.92M elements), 0.018492 (T2,
  14.25M), 0.018663 on a fresh third tuple (T3, 3x3 s1 128x128 C64->64, 1.05M) — cross-tuple ratio **1.009** over
  13.6x element count, two kernel sizes, two output extents; all three TF32_TENSOROP. The constant is
  P(rel_err > 1e-2) = **1.85-1.87%**, a property of the TF32 reference, not of a shape.
- **Low-scale regime measured:** at g=16 (|A|max 77-84) frac_over falls to 2.71e-4 / 2.75e-4 while the
  pure-relative fraction stays 1.861% / 1.851%, so 98.5% of the tail is absorbed by the absolute floor. Absorbed
  mass 1.83e-2 (g=16) -> 7.9e-4 (g=1024) -> 1.1e-5 (g=1e5).
- **Mechanism reconstructed exactly:** s93b (post-hoc) recomputes the predicate as P(rel > rtol + atol/|X_e|) from
  measured per-element |X_e|; agreement 0.0 absolute on 8/8 points (MODEL_EXACT), threshold at median |X| 0.010933
  (g=16) / 0.010015 (g=1024) / 0.010000 (g=1e5). No free parameter, so the value is the decomposition it licenses.
- **Class control:** tf32-off frac_over <=7.3e-5 at every point (rho <=14.7).

Column note, replacing s92's ceiling-only wording: the usability column now ships a tail constant with a domain —
1.85-1.87% for |A|max >~ 2e4 (u >~ 4e2), approached from below (1.78% at 4.9e3), suppressed by the gate's own atol
floor by a factor of ~70 below |A|max 80. Admission stays extremum-governed at u*~0.87. Scope: conv/convT, three
tuples, one draw per point, GB10 sm_120.
