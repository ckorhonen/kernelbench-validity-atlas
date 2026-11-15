# s106b PREREG (declared extension) — scale sweep on the matched-K ladder

Written and committed BEFORE any row of results/s106b_scale_sweep.jsonl. Extends s106 (whose frozen branch was
BRANCH_SCALE_DEPTH_ONLY at g=1: G/C1/C3 medians 13.04/13.29/13.79, ratios 1.02/1.06/1.04, all arms TF32_TENSOROP,
K1-K3 clean). s106 measured ONE scale point, and at that point every arm is far over the 1e-4 budget (frac_over .548),
so the collapse could be a saturation artifact rather than an invariant. This leg sweeps the scale to turn the point
into a domain.

Design: the same three arms and machinery as s106 (G matmul 1708x1728x1708; C1 conv 1x1 C_in=1728; C3 conv 3x3
C_in=192; all implied K=1728, ~2.917M outputs, unit-normal x, w/sqrt(K)*g). g in {0.0156, 0.0625, 0.25, 1.0, 4.0,
16.0}, 2 draws per (arm, g) with paired seeds. Evaluator-exact rho and frac at both gates against chunked fp64
anchors. Engine class profiled at g=0.0156 and g=16 per arm.

Frozen rules:
  - BRANCH_CLASS_GAP_INVARIANT if max over all levels and both ratios of max(R1(g), R3(g)) < 2.0: no conv/gemm term
    exists anywhere in the measured scale domain (u_strict 0.18 to 182).
  - BRANCH_CLASS_GAP_AT_SCALE if any level has R1 or R3 >= 2.0: the levels are named and the gap's domain is reported.
  - Scale law: log-log slope of median rho vs g per arm; s91 measured .965/.948 for the extremum at fixed gate, so a
    slope outside [0.7, 1.3] on any arm is recorded as a deviation, not repaired.
Kill: if any arm's class is not TF32_TENSOROP at a profiled level, that level is class-crossing and is excluded from
the ratios with the exclusion counted.
