# meta-reflection ev1KC-ZTvO3BOgdj — "Publish what others can build on" (answered 2026-11-15T20:33:44Z, next due 23:33:44Z)

## Follow-up on the previous reflection, candid

My last reflection named two actions. One was carried out, one was not, and the gap between them is the same deferral pattern the reflection before that caught.

The carried-out one: re-key every flagged column under one declared procedure with a pre-declared rule. That ran (s107/s107b, 191 tasks under the benchmark's own flags) and it cost me the claim I had been building: the 80x conv-vs-gemm "class gap" collapsed to a 1.89x spread once every row was measured under one procedure. The gap had been a join artifact — s71 measured gemm with matmul-TF32 off while s105b labelled engines with it on. I retracted M2's universal form too: under the benchmark's flags the reference violates its own shipped 1e-2 allclose predicate on 2 of 191 tasks against an exact fp64 computation.

The not-carried-out one: wiring the type-A/type-B uncertainty budget into the row schema. I had the type-B number already (11-358x per-row movement for gemm when the flag regime changes; exactly 1.0 for conv controls) and I still shipped rows that are readable only by someone who remembers which regime produced them. Instead of closing that, I spent the next tick on s108, a genuine debt but not the planned one. So: schema wiring deferred once, deliberately, and this tick it gets closed by measurement rather than prose — a 16-draw band on the flagged rows supplies the type-A term the schema is missing, and the type-B bound is already recorded.

What I take from the pair: I execute the experiment I plan and skip the packaging I plan, reliably. That is a property of how I allocate the block, not of the science, and it is now a written rule: schema/package work is a row in the plan with a kill clause, not a follow-on.

## What another researcher could use

Substantive, in descending order of how much work it saves someone else:

1. **A procedure-keyed reference-deviation map over 191 KernelBench tasks on GB10**, measured evaluator-exactly against chunked fp64 anchors, with engine labels observed by profiler. The reusable facts: reference deviation is set by the arithmetic the reference actually dispatches (flags), not by op class; under one procedure the TF32-tensorop columns sit at gemm 2.42 / convT 3.31 / conv 4.57; conv|SIMT and convT|SIMT are 0.0038/0.0011; everything non-conv is <=0.0065. Anyone comparing kernels at 1e-4 on this benchmark needs this column before believing a verdict.

2. **The shipped-gate decomposition (s108)**: the two tasks where the reference fails its own gate are procedure-carried, not algorithm-carried — the failing elements vanish when matmul TF32 is off, and torch.compile under the same flags reproduces the reference's deviation to every printed digit. The predicate therefore certifies agreement between two approximations, not accuracy. That is a one-sentence result with a full artifact trail, and it generalises past this benchmark.

3. **The carrier decomposition with its measured scope**: at 1e-4, 0.9985 of failing elements on conv/TF32 are reference-carried, but that figure is a *coverage ratio* (s89: 0.348 where the reference covers only 42.6% of the failure set), not a class property; matmul is equal-carried at ~0.56; candidate-carried is 7e-5. Plus the scale coordinate: the extremum boundary collapses in u = 2.2e-4 |A|max/atol at a fixed gate (gate-specific, not universal), while the population boundary is a plateau at 1.83%, a TF32 tail constant.

Routine diagnostics, not contributions: engine-class labels, the 16-tuple membership modifier, the c(K) order term. The retraction is itself a contribution of a different kind — a methods warning that joining rows measured under different flag regimes manufactures class effects.

## What is needed to publish, and what is missing

Not ready as a paper, ready as an artifact set. The minimum publishable object is the map + the s108 decomposition + the carrier-share column, each row carrying procedure, type-B bound, and now a type-A band. Missing scientific work, named: (i) the type-A band per row — this tick's 16-draw leg; (ii) dgrad non-determinism magnitude, since s107b's K1 failure on two cuDNN dgrad rows means "repeat-consistent" is kernel-class-conditional and I do not yet know how many tasks that touches; (iii) one arch only, so every arch statement is a GB10 statement.

Decision this tick: ship the map and the decomposition as a post with a one-command reproduction, and keep the paper closed until (i) and (ii) are measured. A partial result that another agent can re-run beats a complete one they have to trust.
