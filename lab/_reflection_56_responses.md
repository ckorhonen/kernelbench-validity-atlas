# Reflection 57 response — learn from the strongest success

## Strongest success
A10: crossing neighborhood-and-seed study (66 rows, 11 K values x 6 seeds). Discharged
review finding 5 and fired all three pre-registered falsifiers.

## Why it works (essential structure)
1. Falsifier written BEFORE any run: per-seed monotonicity of frac_over in K.
2. Per-seed reporting: turns one aggregate into six measurements.
3. Crossing bracket reported honestly: K=3608/3632, 4/6 and 2/6 pass (coin flip).
4. Re-entry at K=3840: structure is not noise, but not a law.

## Incidental choices
- K grid and seed set can be re-run at any time.

## What prevents extension
- Tuple test (reflection-56 leg) postponed and still not run.
- Manuscript repaired (98,399 B, 570b101, all six auditors green) but not resubmitted.
- A10 CPU-only; no GPU-level evidence.

## Substantive extension
- Run the tuple test with pre-registered falsifiers (fixed shape; vary stride/pad/dilation/
  layout/batch; class flip => 0.604 is a projection artifact; no flip => 0.604 is a real
  unpredictability result).
- Fold A10's measured row into section 2 (already done in 2e7f7c5).
- Resubmit the manuscript with the five f01MfL findings discharged.
- Continue A11 (LayerNorm reference distance) as the oracle side of the reference-order
  school (already done: a11_layernorm_reference_dist.json).
- Save decisions in lab/_reflection_56_responses.md and embed in the research plan.

## Outstanding constraint
- The tuple test requires GPU (cudnn/cuDNN engine-class discriminator). No local GPU
  access on Ceres; ssh spark alias dead. Pending re-establishment of remote GPU access.
