# Conjecture (reflection VpanUgkdgBN9z5yB, 2026-10-20): the offset law

**Conjecture.** The bind-time family offset in the decision-boundary surface
t*(z) = global law + family offset is not a free parameter per family but a
CLOSED-FORM functional of the family-constant z components alone — predictable
from the frozen corpus columns and the family's weight statistics, with zero
labels and zero probes, BEFORE any family row is scored.

**Evidence for.** D1 (commit 798ee44, results/d1_cellaffine.json): the affine
surface R² = .831 with the LN offset absorbed exactly by the intercept while
its d_u ≡ 0 — the B5 mechanism (family-constant z shifts the absolute score)
is the same quantity the offset must encode; the b5 intercept shift +0.846
(t* −0.259 vs b4 −1.105) is the signed mean of the constant-part z's to
first order. Sub-unitary slope 0.588 says the law term and offset term
interact — consistent with an offset defined on the z scale, not the score
scale.

**Weakest point.** n_families = 2, one of which has a degenerate covariate;
the b4 "constant part" (cwr = K−1 artifacts of the corpus grid) may be a
grid accident, and conv's constant-part structure (padding/geometry terms) is
different in kind. The offset could be family-idiosyncratic scribble.

**Serious attempt (item 3 prereg addition, BEFORE the GPU run).** Freeze at
prereg time: the conv family's predicted offset, computed from its
constant-part z signature on the frozen corpus columns; pass = predicted
offset within the M1 residual scale of the measured family intercept
(numeric bound frozen in the prereg); fail = free-offset reframe recorded
(the hierarchy result), lane continues per plan v2 kills (DR6: frozen-t
sufficient / repaired precision <0.80 / cell-fit R² <0.5).

**If true**, the zero-probe axiom extends to decision calibration verbatim:
bind-time shapes + weight stats determine the FULL threshold surface. **If
false**, per-family refit is load-bearing and the atlas records the
hierarchy — either branch is an entry (A4 already carries both branches).

**Status 00:05Z 10-21 (tick 30): rider carried, unevaluable 2nd tick.** Conv family
closed instrument-negative (plan v2 item 3 STATUS): the 1e-4 lens fails everywhere for
conv-as-gemm on GB10 => no two-class units => b_meas undefined; b_conv_pred=-0.570799 /
bound 0.279736 remain frozen and carry (geometry unchanged). Any future conv evaluation
under a repaired lens tests the SAME frozen prediction, no re-freeze.
