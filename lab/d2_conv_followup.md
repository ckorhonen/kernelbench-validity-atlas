# D2 follow-up -- the all-fail conv margin question (reflection quAu5aAf78r-RHSU, 20:40Z)

Surprise: conv-as-gemm fp32 vs conv2d fails the 1e-4 elementwise-all budget EVERYWHERE
(116 rows, min ratio 1.59 at scale 1, growing with scale), while the linear-fold corpus
passes 83% at the same tolerance. Assumption broken: "fp32 reordering << 1e-4 budget".

Explanatures ranked (reflection): (2) near-zero-output dominance of the all() predicate
-- TEST NEXT, no GPU, on committed results/d2_conv.jsonl: per-row margin decomposition,
ratio vs |R| elementwise proxy... rows carry only max_budget_ratio; the elementwise
margins need one CPU/GPU recomputation pass -- PREREG FIRST (frozen before the run):
  primary: fraction of failing elements with |R| < 1e-3 (near-zero band), pooled; and
  the same fraction conditional on ratio > 1. Passing-element |R| distribution as control.
  (3)/(4) discriminator: ratio-vs-scale growth already rules out pure-reorder-only.

Remedy branches, to be frozen in the #30 prereg BEFORE any GPU:
  A) phi-heavy => fp64-accumulate candidate arm + declared tolerance axis; rider
     b_conv_pred=-0.570799 CARRIES (geometry unchanged).
  B) flat margins => '1e-4 elementwise-all is not a conv-reassociation lens' -- atlas
     numerics-axis negative entry, family exits.
Reconsider-if: flat margins AND fp64-accum also failing.

Status: diag NOT run this tick (tick budget); prereg #30 opens with it.
