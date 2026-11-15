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

## s94 — the share does not transfer outside conv, and the depth law is not a power law (2026-11-13)

The decomposition deferred across four ticks ran on fp32 matmul (population declared by rule from A6b/A6d/A6f; prereg
in `lab/s94_nonconv_decomp.py` committed pre-run; `results/s94_nonconv_decomp.jsonl` 5/5 ok, `results/s94_analysis.json`
post-hoc). Three findings for the column:

- **The compile candidate is bit-identical to the reference on this family (5/5)**, so the decomposition is carried by
  a chunked-accumulation candidate declared in the prereg, not by Inductor.
- **Share 0.560, not 0.9985.** 300 failing elements at 1e-4 split 168 reference-induced / 132 candidate-induced;
  per-tuple 0.600 / 0.562 / 0.541 (spread 0.059). The conv constant is a property of a population where the reference
  carries a coarse engine, not of the comparison itself.
- **Carrier rule, measured on three cases:** reference carries TF32 → 0.9985 (s84/s87); both arms fp32 with comparable
  error (rho_cand/rho_ref 0.96-1.15) → 0.56; candidate carries TF32 → 7.0e-5 (3.50M failing elements, all its own).

Depth: rho_ref 0.580 / 0.559 / 1.235 / 2.449 at K=1024/2048/4096/8192, crossing bracketed 2048 < K < 4096 (A6d's
3608/3632 replicated on a fresh draw); fresh geometry M=N=512 K=8192 gives 1.652. The log-log fit is slope 0.738 with
0.32 max log residual and a −3.7% step at the bottom, so DEPTH_LAW_INTERMEDIATE: the depth effect is piecewise, not an
exponent, and the column does not ship `eps(K)` yet.

Column note addition: the reference-induced share must be printed with its carrier case. Reporting 0.9985 outside conv
is a category error; reporting a share without naming which arm holds the coarse engine is uninterpretable in the same
way that a bare "reference-clean" is (§4.5e). Scope: matmul, five tuples, one draw per tuple, fp32, GB10 sm_120.

## s95 — the low-K floor is draw noise; the exponent is draw-stable and the amplitude is not (2026-11-13)

Eight independent draws per K at M=N=1024, K in {1024, 2048, 4096, 8192} (prereg `80a91f6` frozen pre-run;
`results/s95_draw_repeat.jsonl` 33/33 ok with instrument repeat bit-identical 3/3; `results/s95_analysis.json`).
The reflection's bold conjecture — a real two-regime floor below K*~2048 — is refuted: the K=1024→2048 decrease
appears in 1 of 8 draws, and the means run 0.504 → 0.626 (+24%, 1.83 pooled sd). Branch FLOOR_NOISE.

What survives is better for the column than what s94 had. Per-draw log-log slopes are 0.692-0.900 (mean 0.771,
sd 0.081), so the depth exponent is stable across draws while the amplitude at fixed K spans 1.39x (K=1024), 1.37x
(2048), 1.20x (4096) and 1.53x (8192) with CV 12.7/11.1/6.6/14.3%. The prereg's stability kill fires above 10%, so no
exponent claim is issued; what ships is a scaling relation with a draw-distributed amplitude, `eps(K)=c(draw)·K^0.77`,
and the rule that a per-tuple rho_ref is a sample from that distribution, not a property of the tuple. frac_over_ref
at the shipped 1e-2 gate is exactly 0.0 on all 32 rows, so the matmul reference is clean at the benchmark's own gate
on every draw; at the strict gate it is 0.0 at K≤2048, 3.6e-6 at 4096 and 1.9e-4 at 8192 — three to four orders below
conv's 1.85% tail. The share is likewise a distribution, not a constant: 16 rows at K≥4096 give 0.25-0.75 (mean 0.502,
sd 0.134), SHARE_UNSTABLE, and s94's 0.560 lies inside it.

Column note addition: a per-tuple rho_ref must be printed with its draw spread, and a share with both its carrier case
and its draw spread. Scope: matmul, one geometry, fp32, eight draws per K, GB10 sm_120; eight draws resolve a 3.7%
effect but do not characterize the tail.

## s96/s96b — the amplitude has no input, engine, or location driver: it is a rounding-magnitude lottery (2026-11-13)

32 independent draws at M=N=1024, K=4096, benchmark fp32 flags (prereg `cb23e44` frozen pre-run;
`results/s96_amplitude_mechanism.jsonl` 33/33 ok with instrument repeat bit-identical on rho and frac_over;
`results/s96_analysis.json`; s96b post-hoc declared, `results/s96b_ratio_argmax.jsonl` 32/32 ok). Kills clean:
K1 instrument ok, K2 vacuous not fired, K3 degenerate not fired.

rho_ref spans 1.046-1.474 (1.41x, mean 1.226, CV 9.04%) while every predeclared covariate is nearly constant:
sigma_max(a) CV 0.20%, kappa proxy CV 0.51%, median |X| CV 0.125%, max |X| CV 4.4%. Spearman against rho_ref:
−0.192 sigma_max(a), −0.126 kappa, −0.008 median |X|, −0.217 max |X|, −0.005 max abs error, +0.050 cancellation
depth — all below the 0.40 bar, so **AMPLITUDE_LOTTERY** fires and no predictor ships. Two caveats stated rather
than smoothed: Gaussian draws leave sigma_max(a) inside 0.2% by Marchenko-Pastur, so the spectral hypothesis is
untested where conditioning actually varies; sigma_max(b) was recorded but not a predeclared verdict arm and returns
−0.558, which is a candidate for a designed-conditioning test, not a finding.

Engine identity is stable: eight profiled draws plus the instrument repeat all return one kernel set
(`cutlass_80_simt_sgemm_256x128_8x4_nn_align1` plus a device memset), so the engine-lottery explanation is rejected.
The tail ladder is 0.75% CV at p99, 1.07% at p99.9, 2.02% at p99.99, 3.15% at p99.999 against 9.04% for the maximum
(ratio 0.223 at p99.99); the prereg's tail clause needed CV(rho_ref) > 0.10 and measured 0.0904, so it fires
**TAIL_INDETERMINATE** verbatim and the ladder is reported as extreme-value-shaped, not as a mechanism.

s96b locates the variance inside the ratio: the ratio-argmax element always sits in deep cancellation (|X| 5.9e-4 to
0.41, depth 1.4e-5 to 9.5e-3 of median), the identity log rho = log e − log(1e-4(1+|X_e|)) holds to 8e-7, and the
absolute error there spans 1.075e-4-1.738e-4 (1.62x, CV 12.8%, Spearman with rho 0.661) while depth does not predict
it (−0.021). So the amplitude is an error-magnitude lottery at the worst-cancelled element.

Column note addition: a per-shape rho_ref ships with its draw distribution and never as a shape property; at K=4096
all 32 draws exceed rho_ref=1 (min 1.046), so the upper end of the 2048<K<4096 crossing is draw-robust while its
magnitude is a sample. Scope: matmul, one geometry, fp32, 32 draws, GB10 sm_120.

## s99 (2026-11-14): designed conditioning retires H1

s96's null on spectral amplification had no leverage — iid Gaussian draws hold sigma_max inside 0.2% — so s99 designs
the conditioning instead of sampling it: b = U·diag(s)·Vᵀ with orthonormal factors from QR of Gaussian draws,
log-spaced singular values, Frobenius norm fixed at unit-RMS scale, kappa swept 1→1e4 at M=N=1024, K=4096 with the
benchmark's fp32 flags, five draws per level (prereg `lab/s99_prereg.md` at `e35d699`, frozen pre-run; artifacts
`results/s99_conditioning.jsonl` 25/25, `s99_instrument.json`, `s99_conditioning_summary.json`,
`s99_analysis.json`).

Kills all clean: instrument repeat bit-identical on rho_ref and frac_over; construction check reproduces the
prescribed singular values to 7.8e-6 relative; and every level dispatches the same
`cutlass_80_simt_sgemm_256x128_8x4_nn_align1` set, so conditioning does not perturb dispatch.

Level means of rho_ref: 1.341 (kappa 1), 1.301 (10), 1.356 (100), 1.168 (1e3), 1.208 (1e4) — a 1.16x spread over four
decades of conditioning, with sigma_max(b) rising only 64.0→273.6 at fixed Frobenius norm. Row-level span is 1.92x
(1.020-1.961) and it sits entirely inside the levels: within-level CV is 9.1/11.8/22.7/8.8/5.1% (median 9.1%), the
same spread s96 measured at Gaussian inputs. Frozen verdict **P2_RETIRE_H1** (Spearman(kappa, rho_ref) −0.35 < 0.40
bar, span 1.92x clears the 1.3x clause). Descriptive: Spearman(median|X|) +0.376, sigma_max(b) −0.319, depth −0.105.
Strict-gate population fraction 3-5e-6, shipped 1e-2 fraction exactly 0.0 on all rows.

Column note addition: the amplitude lottery is now a statement that survives a designed intervention, not just a
sampled null — a per-shape rho_ref ships with its draw distribution even when conditioning is fixed. Scope: matmul,
M=N=1024 K=4096, fp32, GB10 sm_120, five draws per level.

s100/s100b/s100c (LN family, GB10): the arithmetic-order axis is now priced at the official L1/40 shape. Tree width
128 (the reference's realized block) gives rho_vs_ref 0.0132 at 8.21 ms; width 1024 gives 0.0556 at 5.75 ms, with
144/160/192/224 interpolating smoothly (0.0148/0.0172/0.0245/0.0312) — so the coordinate is continuous in tree
width, not a cliff. frac_over stays exactly 0.0 on all 88 rows, i.e. the whole axis spans 1.3%-5.6% of the 1e-4
budget and never approaches it. A Welford estimator on the identical tree moves exact_frac 0.479->0.520 and the
ratio not at all (0.01319->0.01325) at 2.3x the time, so bit identity is a tree/order property, not an estimator
property. Column consequence: where the reference margin is slack the admission rule can ignore arithmetic order
entirely; where it binds (conv/TF32 rows) the order term still has to be priced, which is the s101 leg.

s101/s101b (conv/TF32 binding regime, GB10): the order axis is now priced where the budget binds. Chunked
permutations change engine class below C_in=32 (probe: 8/16 channels -> implicit_convolve_sgemm FP32_SIMT; 32/64 ->
cutlass tensorop TF32), which makes a chunked candidate look exactly as wrong as the reference (c ~ 1.0000). With
the class held by zero-padding, the declared trees cost 0.22-0.45% of the reference's own deviation across u
0.66-680 at 1e-4, inherit none of its exceedance set (frac 0.0 vs frac_A up to 0.787), and reach the gate
themselves only near u ~ 400. Column consequence: the order term is a shallow surface in (u, depth), not a constant,
and it is negligible until the reference's own margin is hundreds of times the budget.

s102 (second tuple, GB10): the same declared fixed-class permutation on T2 (1x1, C=192->160, 211x211) reproduces the
shape of the surface and falsifies its single number. Above u=10 the ratio is flat within 1.13x (0.00096-0.00130) but
2.0-2.6x below T1's matched-u values, all 24 comparisons outside the prereg's 1.5x band, so the frozen branch fires
TUPLE_SPECIFIC; frac_vs_ref stays exactly 0.0 while frac_A reaches 0.789, and the permutation's own crossing sits at
737 < u < 1473. Column consequence: the order term ships as a per-tuple entry (negligible everywhere measured, but its
magnitude is not portable), and the dispatch probe adds that the 1x1 family at this shape does not leave FP32_SIMT
until C_in=192 — engine class is a decision over the whole tuple, not a C_in threshold.


s103/s103b (GB10, discriminating tuple): the order term's magnitude is owned by the reference's accumulation depth,
not by the engine instance. Holding T2's spatial extent, output count and engine instance fixed (the reference
dispatched the same cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align1 at C_in 192/384/576/768) and raising K from
192 to 576 moves the cost ratio from 0.0010 to 0.00220-0.00275 — 2.10-2.72x T2 at matched u, median 2.44, and within
1.2x of T1's recorded 0.00219-0.00233. A declared post-hoc leg at K=1728 (3x3, cudnn fprop instance) gives
0.00663-0.00968, i.e. 2.88-4.13x T1, and the two log-log steps (0.83, 1.19) bracket an overall slope of 1.01. Column
consequence: the order term ships as c(K), approximately linear in accumulation depth on this population
(0.001 at K=192, 0.0025 at K=576, 0.0094 at K=1728), not as a per-tuple entry and not as one number; C_in is not
separable from K in the 3x3 legs, so depth is the coordinate that carries and channel count is not claimed. frac_vs_ref
stays exactly 0.0 through u=384, so the permutation still inherits none of the reference's exceedance set.

### Column note addition (s104/s104b, 2026-11-14)

The order term ships as c(depth) with a measured membership modifier, and carries no merge-shape coordinate. E3's
merge-tree sweep (prereg lab/s104_prereg.md, runner lab/s104_topology_sweep.py, repair leg lab/s104b_topology_draws.py,
artifacts results/s104_topology_sweep.jsonl, s104_summary.json, s104b_topology_draws.jsonl, s104b_summary.json) ran four
topologies at matched depth on T1 across eight draws: interleaved chunk membership lowers the paired cost ratio by
15-19% on 8/8 draws at both scales (median 0.854 at g=1, 0.808 at g=64), while reordering or balancing the merge moves it
by <=3% with unstable sign. The same engine instance (cutlass_tensorop_s1688fprop TF32) dispatches for all four arms, so
this is not a class artifact. The quantity is draw-distributed (c spans 1.42-1.55x across eight draws at fixed
configuration) and gate-invariant (identical at 1e-4 and 1e-2, because both numerator and denominator scale with 1/atol).
Non-claims: one tuple, channel-axis chunking only, fp32 partial accumulation, no timing claim.


### Column note addition (s106/s106b/s106c/s106d, 2026-11-15)

The map's "same engine class, 80x apart" comparison is withdrawn: it was a flags-crossing join. Under one flag regime,
conv and gemm references at matched accumulation depth (K=1728), matched output count (2.917M) and matched scale are
indistinguishable — rho_1e-4 medians 13.04/13.29/13.79 for matmul, 1x1-conv and 3x3-conv arms, ratios 1.02-1.06, all
three TF32_TENSOROP, K1-K3 clean — and the ratios stay 1.03-1.13 across a scale sweep from u=0.18 to u=183 with
log-log slopes 0.946/0.961/0.973 (s91's extremum law replicated on three new arms). On the benchmark's own tasks the
gemm references run FP32_SIMT (or FP64) under s71's default set_cfg(True, False) and TF32_TENSOROP under
set_cfg(True, True), with median rho moving 0.039 -> 2.974 (10/10 rows, ratios 11-358), while conv controls do not move
at all. Column consequence: the reference's deviation is set by the arithmetic the reference actually runs (flag
regime) and by the scale u; the class column remains an observation per task, but it is not a coordinate and the 80x
statement does not survive. Non-claims: one tuple per arm, one arch, no timing claim; the flag regime is a
record-join defect in the map, not a property of the benchmark's tasks.
