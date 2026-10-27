# fp64 tuple test (stage 1)

Prereg: meta-reflection standing plan, lab/meta_reflection_plan.txt.

Grid: C=384, k=3, s=1, d in 1..4, pad=p, H=W=7, batch=2, seed 4321.

Results: 9 rows, 9 flips, all-class-match=False.

Decision rule: fp64-anchored W/T labels equal TF32-class labels on all 9 rows => reference-invariant. Result: ANCHOR-CONDITIONAL.