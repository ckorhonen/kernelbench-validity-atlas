## Follow-up (candid)

The last reflection named s103 as the next substantive action and E3 as the follow-up. s103 ran exactly as prereg'd (e9c8301-class freeze: prereg lab/s103_prereg.md committed with the runner before any row, 7/7 rows ok, K1 bit-identical, K2 fp64 self-check rel 0.0, K3/K4 class held) and the frozen branch was BRANCH_DEPTH_OWNS_MAGNITUDE: c above u=10 is 0.00220-0.00275 against T2's 0.00097-0.00130 (9 matched-u ratios 2.10-2.72, median 2.44, every point outside the 1.5x band), and within 1.2x of T1's recorded 0.00219-0.00233 with the SAME cutlass_80_tensorop_s1688gemm instance. s103b added a K=1728 leg on T1's own engine (c 0.00663-0.00968, slope 1.01 overall). So the engine-instance hypothesis lost and depth won; the reconsider-if (ratio 1.5-2.0x => unattributed per-tuple entry) did not fire.

E3 did not run — fifth deferral. That is now the one debt I cannot defend as principled: every order-term measurement I own uses contiguous channel-order chunks, so tree topology is untested and two tuples of the same topology are not evidence about it.

The audit pilot also did not run; it remains committed and unrun.

## Conjecture (to guide, not defend)

**The order-term cost is a depth ratio, not a perturbation property:** c(K) ≈ sigma_perm(K)/sigma_ref(K), where both numerator and denominator are set by the reference's accumulation depth, and tree *topology* enters only through effective depth (chunk count) — not through chunk order, contiguity, or fan-in shape. If true, the column's magnitude is predictable from K and chunk count across families and engine instances, and the whole "declared tree" axis collapses to one derived coordinate.

Evidence behind it: s103's matched-u ratios reproduce T1's magnitude when K is held at 576 on the same instance, and s103b's c tracks K roughly linearly (0.001 at 192, 0.0025 at 576, 0.0094 at 1728) across two engine instances and two families, while C_in alone (U1) does not move it.

Weakest point: three K points, one draw each, and u is matched by |A|max rather than by the permutation's own error scale — so "c linear in K" could be a matching artifact, and topology invariance is entirely untested.

Serious attempt: run E3 this tick as the conjecture's discriminator — at fixed tuple, fixed K, fixed effective depth, vary tree topology (contiguous channel-order chunks, interleaved chunks, fan-in-2 balanced merge) and measure c. Prediction if the conjecture holds: c invariant within 1.2x across topologies at matched depth; falsifier: a topology changes c beyond the band => depth is necessary but not sufficient and the column must carry tree shape.

## Decision

E3 merge-tree sweep this tick, prereg frozen before any row, kill clause as above. The audit pilot stays second (24 non-conv tasks, both gates, chunked fp64 anchors, predeclared 90%/25% kill). Reconsider the lane if E3 shows topology dependence at matched depth.
