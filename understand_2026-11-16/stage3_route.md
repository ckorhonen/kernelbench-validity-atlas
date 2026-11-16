# Stage 3 — Route and scaffold design (Deep Reflection session 2, completed 15:29:08Z)

Routes compared: (1) configuration-mixture coordinate; (2) conditional-exceedance model; (3) honest retreat to measured
bands only. Route 1 is cheap and partly refuted already (s112: one engine name on 16/16 draws for the row with the
widest swing). Route 3 has no failure mode that teaches anything. Route 2 chosen, with its object corrected.

**The correction that matters (round 2 of this session): the map's rho is an accuracy statement, the benchmark's gate is
an agreement statement.** s108's L2/14 shows the extreme case: compile under the same flags deviates from exact fp64 by
exactly the reference's amount (16.4779 to every printed digit) and passes the benchmark's own check while the
reference is 16.48x the gate from truth. So 83/191 measures reference accuracy, not verdict risk. The predictive
certificate therefore has two parts:

(a) accuracy interval for the reference — already published (82.3, Wilson 79.3-84.6), recomputable by s111_repro;
(b) agreement-risk statement for candidates, keyed on whether the candidate's engine class shares the reference's
arithmetic, with the mechanism coordinate being the per-element residual correlation between the two arms.

Scaffolds and their removal:
1. Committed arm record (r1_conv_gated.jsonl: 56 rows, 672 arm-checks, declared precision, pass/fail, error dicts) —
   ordering test on CPU today, conv fit / convT+gemm held out.
2. Declared-precision proxy -> named-engine join from committed event lists (191-row map run), keyed on
   (task_id, procedure, seed), row-count asserted before any coefficient.
3. Aggregate calibration vs s110's Wilson interval (the only well-powered coverage check available).
4. Only if 1-3 disagree: GPU leg, three rows spanning shared-engine / different-engine at matched rho_ref, per-element
   residuals both arms vs one anchor, correlation and variance ratio.

Push on: monotone ordering, out-of-class prediction inside the binomial interval, aggregate inside the Wilson interval.
Blocked: flat ordering, or reversal under rho_ref control, or a join that cannot be made row-exact.

24h plan: M1 CPU ordering test (frozen rule pre-run) -> M2 named-engine join -> M3 aggregate calibration -> M4 GPU
correlation leg if needed -> M5 ship as post-11 comment + updated public producer + ATLAS_MAP/§4.5 note.
