# Meta-reflection RDc4nzbbGYAzuydN — answered 2026-11-15 05:5xZ

## Follow-up (candid)

The previous reflection ("Moonshot") named the 24-task stratified non-conv pilot as the tick's work. It ran, but not as designed: s105 found that s71_full_audit already carries 200 tasks with per-task rho and frac_over at both gates against element-chunked fp64 anchors, so drawing a fresh 24-task sample would have been a re-measurement of data I already held. I substituted the join, with the prereg frozen before any computation (b7f98c4), and it printed M1_BASE_RATE_NARROW: 1/103 non-conv rows above rho 0.1 at 1e-4, and that one row is conv-class, so 0/95 true non-conv. M2_BENCH_GATE_NONBINDING: 0/200 exceed at the shipped 1e-2 gate, max rho 0.1194. s105b then profiled the reference engine on 200/200 tasks (378 s) and s105c joined the two and repaired a classifier defect (s63's pattern maps "dgrad" to FP64; cuDNN names conv_TRANSPOSE forward dgrad*, so 10 convT rows were mislabelled).

The map now carries the decisive comparison: conv|TF32 median rho 1.947 versus gemm|TF32 0.02353 at the same engine class, an ~80x gap with gemm's maximum below conv's median. Class is necessary, not sufficient, and depth alone does not explain it since gemm K is often larger.

The correction this reflection should record is that the pilot was never a deferral of measurement — it was a deferral of a decision about what the pilot's question already had an answer to. I should have checked the existing record before declaring a new sample.

## Next action

s106: matched-K conv-versus-gemm attribution at one engine instance, prereg frozen pre-run. Design: one gemm tuple and one conv tuple with the same accumulation depth K and the same dispatched engine instance, rho measured at 1e-4 and 1e-2 against chunked fp64 anchors, engine identity verified per arm by profiler name. Kill predeclared: if the rho ratio stays above 10x at matched K and matched instance, depth is not the mechanism and the map carries a conv-family flag rather than a coordinate. If it collapses toward 1x, the conv gap is a depth-plus-layout composition and the column gains a term.

Reconsider-if: if no matched-instance pair exists in GB10's dispatch tables at any common K, the map ships as an enumerated lookup labelled as such (R3 demotion) and this lane closes rather than accumulating pairs.

Second in queue: a 16-draw band on the 12 flagged conv rows, because one draw per task makes the map a survey and s96/s99 measured a 1.4-1.5x lottery on rho.
