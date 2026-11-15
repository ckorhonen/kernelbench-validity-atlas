# Stage 1 — Evidence review: distance to a decisive advance

What a transformative advance would look like on this problem: a benchmark-wide, quantitative validity map of the
evaluator itself — per task, the reference's own deviation as a fraction of the admissible budget at the shipped and
strict gates, plus a derived coordinate that predicts that fraction from declared features (op class, accumulation
depth K, |A|max/atol, engine class), plus the admission consequence (how much candidate margin a declared order
perturbation consumes). That converts a local conv finding into a statement about the benchmark as an instrument.

Evidence that it is worth attempting: conv is measured and reference-carried (55/200 exceed at 1e-4; 99.8% of 2.183B
failing elements reference-induced; s84/s89/s87); the extremum crosses at u = 2.2e-4|A|max/atol ~ 0.68 with a
gate-specific constant (s91); the population statistic saturates at a 1.83% TF32 tail with an exact reconstruction
(s92/s93); at the shipped 1e-2 gate 0/200 exceed with benchmark-realistic weights but 11/24 exceed at unit scale (s90,
scale-conditional); matmul is 0.56 reference-carried with rho_ref crossing bracketed 2048<K<4096 (s94/s95), i.e. depth
carries outside conv; engine class is categorical and stable across contexts/processes/budgets (s63/s66/s72).

Core obstacle: the population beyond conv is untested at benchmark scale. s76 found 7/7 non-conv exclusions
reference-clean and bit-identical under tf32-off; s90 found 0/24 at 1e-2 with realistic weights. Two hints that the
map may be narrow, one measured counterexample (matmul deep-K) that says it is not empty. Nobody knows the base rate.

Relationship to my work: rungs 1 (measurable per task) and partially 4 (admission) are solid; rung 2 (non-trivial
outside conv) is the open question; rung 3 (predictability) has a mechanism sketch (eps_engine x K^0.77 x u) with a
measured 10-40% amplitude lottery.
