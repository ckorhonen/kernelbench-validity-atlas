# Stage 2 — Target and working-backwards (Deep Reflection session 1, completed 15:16:16Z)

Target: a per-task *certificate* rather than a per-task measurement — for every KernelBench task, a predicted
P(exceed at the shipped gate) computed from one fp64 anchor plus declared coordinates, with declared coverage.

Working backwards, four links were named; only L1 is supported by measured evidence (procedure-keyed deviation: s107,
s114b, s96). L2 (draw-stability given the mechanism) is the link that fails today: 4.4x swing on one engine name.
L3 (band estimable from few draws) is partially supported (s109 CV 3.6-7.5% away from boundary; boundary row ships a
rate). L4 (band predictable from coordinates) untested.

The crucial uncertain step, as first framed: what does the draw change, if not the input values (s96), the conditioning
(s99) or the engine choice (s112)? Candidate: allocation/launch context (pointer alignment, workspace, tile/split
selection, block scheduling). Test: bit-fixed inputs, N controlled alignments, rho per alignment against one anchor.

Round 2 reframing (the useful turn): rho is a max over millions of elements, so part of the observed swing may be the
intrinsic variance of a max statistic; the predictive route may not need attribution at all, only an extreme-value law
estimated from one anchor's element-level error distribution. Also: the draw axis moves weights, inputs and allocations
together, so s96 excludes value statistics only.

Round 4 refinement: predict frac / P(exceed) rather than rho, because frac is the quantity that decides verdicts and
has far smaller variance (s109b frac up to 5.13%). Exchangeability is the difficulty: if configuration varies across
draws the band is a mixture; s112's 16/16 one-configuration result already partly refutes the mixture reading for the
widest-swing row. Calibration honesty: 11 rows with measured bands give almost no coverage power; the aggregate against
s110's Wilson interval is the one well-powered check.
