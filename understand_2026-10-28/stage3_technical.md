# Stage 3 — Technical brainstorming (completed 19:00Z 2026-10-28)
Verified in source: `src/kernelbench/eval.py` get_tolerance_for_precision() → fp32 1e-4, fp16/bf16 1e-2 (torchbench
cite). So the benchmark's declared fp32 gate IS 1e-4, the same order as the reference's own TF32/winograd deviation.
The inheritance claim bites at the shipped contract, not only at tightened ones. rho normalises by the gate, so
rho>1 = elements fail; frac_over must be reported alongside rho.
Chosen directions: (b) reference-swap flip table (intervention, falsifiable, cheap; producers exist) at the declared
1e-4 gate; (c) boundary bisection + position predictor (second session); (a) benchmark-wide reference-error audit
(scope, after (b) proves the gate consequence); (d) winograd bound = one derivation paragraph only; (e) two-sided
admission parked (needs candidate-side class per arm); (f) collapses into (b).
24h order: repairs+resubmit → prereg+run (b) → out-of-family class prediction (frozen, kill at majority baseline)
→ bias-vs-tail check → audit scoping doc. Change of course: few flips + baseline predictor ⇒ publish standalone,
pivot to benchmark-wide audit.
