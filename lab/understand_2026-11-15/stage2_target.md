# Stage 2 — Target and working-backwards (reflection session 1, 5 rounds, completed 01:56Z)

Target: benchmark-wide evaluator validity map, conditional on the base rate measured today.

Claim ladder and kill conditions:
R1 (instrument): per-task reference deviation measurable with chunked fp64 anchors and the benchmark's own input
   generators at both gates. Solid on conv (200/200); untested on reductions/attention anchors.
R2 (base rate): >=25% of stratified non-conv tasks above rho 0.1 => WIDE map; >=90% below => NARROW (scoping note);
   between => PARTIAL, report the split counts with sampling weights.
R3 (predictability): (engine class, K, u) predicts held-out rho within 2x on >50% of falsifiable rows; else the map is
   an enumerated lookup and is labelled as such.
R4 (admission consequence): a declared order permutation on flagged near-budget rows consumes a measurable share of
   candidate margin; c(K) and the membership modifier are the prior.

Crucial uncertain step: the base rate outside conv. Everything downstream is only worth building after it exists.
Weak assumptions named: the 0.1 threshold is arbitrary unless stated as a budget fraction; scale must come from the
benchmark's own generators (s90); stratification must be over declared features, not expectations.
