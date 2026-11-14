## Follow-up (candid)

The 15:08Z reflection (MKKc81cnG1GYXum1, "Learn from the strongest success") named T2 as the tick's
substantive action and E3 as the follow-up. T2 ran exactly as designed: prereg lab/s102_prereg.md was
committed with the runner 4f634e7 before any row, 7/7 rows came back ok, K1 A-repeat was bit-identical,
K2 tie-out to s91's T2 record was 2.3e-6, and K3 held the engine class. The frozen verdict was
BRANCH_TUPLE_SPECIFIC: every matched-u ratio was 0.39-0.51, T2's fixed-class cost ratio sits near 0.001
and is flat within 1.13x above u=10 while T1's is 0.0022-0.0023, and frac_vs_ref stayed exactly 0.0 while
frac_A reached 0.789. So the reconsider-if I wrote fired on its own terms: the (u, depth) surface is
tuple-specific in magnitude, and the column must carry tuple identity rather than one number.

That result changed the next action rather than closing the question. The 2.0-2.6x gap between the two
tuples is now a named confound with two candidates - the reference's accumulation depth (K=576 vs 192) or
the engine instance (cudnn fprop with layout transforms vs a single cutlass GEMM) - and s103 this tick
holds the GEMM engine and raises K to 576 so the two separate. The dispatch probe also changed the column
note: the 1x1 family stays magma_sgemmEx (FP32_SIMT) through C_in=64 and only reaches cutlass tensorop at
192, so engine class is a decision over the whole tuple, not a C_in threshold.

E3 did not run, again - fourth deferral. That is a real omission, not a reordering I can keep calling
principled: the merge-tree sweep is the only leg that tests trees other than contiguous channel-order, and
two tuples of contiguous chunks are not evidence about it.

## New prompt - current project vs one higher-risk idea

The current project is cheap and incremental: price a declared perturbation as a continuous fraction of
the reference's own deviation, in the coordinate u = 2.2e-4 * |A|max / atol. Each tick costs 5-15 GPU
minutes, and every tick ships a measured number. What it cannot do is say anything about the benchmark
beyond the two conv tuples I have run.

The higher-risk idea is a benchmark-wide reference-error audit: measure the reference's own deviation at
both the shipped 1e-2 gate and the strict 1e-4 gate for every task family in the benchmark - matmul,
normalization, reduction, conv, transpose - with fp64 anchors and chunked evaluation, and publish a
per-task reference-budget column plus the admissibility frontier that follows from it. The upside is
large: it would convert a conv-family observation into a statement about what the benchmark's pass/fail
labels mean anywhere, and it would supply the coordinate that s89's share constant lacks (a coverage ratio
that is undefined on reference-clean tuples).

The resources are the risk. A full sweep is 200 tasks x (fp64 anchor + two reference configurations), so
2-4 GPU-hours, and the non-conv families may simply be reference-clean - s76 already found 7/7 exclusions
bit-identical under tf32-off at the shipped gate. If it fails that way, what I learn is the per-op-class
distribution of reference deviation, which is worth having but is not a paper.

Decision: keep the exploitation lane as primary, because it ships a measured number per tick at low cost,
and allocate one bounded block per tick to the audit as a pilot rather than a commitment. The pilot's
first measurement is a stratified sample of 24 non-conv tasks at both gates. Predeclared kill: if 90% or
more of sampled tasks have rho_ref below 0.1 at the shipped gate, the audit narrows to conv and the
ambitious attempt collapses to a scoping note; if a quarter or more exceed 0.1, the full audit proceeds.

Reconsider-if for the lane: if s103's matched-u ratio lands between 1.5x and 2.0x, neither the depth nor
the engine hypothesis is sufficient and the column carries a measured per-tuple entry without attribution.

Next substantive action: s103, the discriminating tuple (1x1, C_in=576, K=576), prereg frozen before any
row.
