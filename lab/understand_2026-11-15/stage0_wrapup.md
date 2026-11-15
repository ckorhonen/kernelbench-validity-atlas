# Stage 0 — Review and continuity (2026-11-15 01:39Z tick, session 105)

State at open: lane = exploitation, order-term surface. s101b/s102/s103/s103b/s104/s104b complete: the order-term
cost c(K) is approximately linear in the reference's accumulation depth across three tuples and two engine instances
(0.001 at K=192, 0.0025 at K=576, 0.0094 at K=1728); membership (interleaved channels) lowers c 15-19% on 8/8 paired
draws; merge shape is not a coordinate (<=3%, sign-unstable); c is gate-invariant; the 1.20x unpaired bar sat below
the draw noise floor (1.42-1.55x), lesson recorded. All artifacts and producers on disk, audits green (s50 19/19,
s51 47/47, s52 16/16, s53 15/15, s54 6/6, provenance 85/85/0).

Reusable machinery: element-chunked fp64 anchors (s73b, s76 -> 200/200 coverage); evaluator predicate helpers
(rho = |R_def-R_64|/(atol+rtol|R_64|)); engine-class profiler labelling (s63/s66/s72/s75); task loader from the
KernelBench source tree (s76 coverage run). GPU: spark GB10, ssh alias dead from cron, use ckorhonen@192.168.62.102.

Debts carried: (1) benchmark-wide reference-error audit pilot committed 11-13, never run (oldest unrun debt);
(2) membership is one variant on one tuple; (3) split-K across output/spatial axes untested; (4) s51 covers
§4.5-§4.5h only; (5) package repack owed; (6) 22-31% cross-gate shift unexplained; (7) L2/14 ordering; (8) 24h-E1/
CTA-edge/bt parked.

Recent lessons: prereg prose and coded branches have disagreed three times (e2b direction, s100 monotonicity, s101
pooling across an engine-class change) => freeze a declaration file and an analysis script that prints its branch;
never silently drop instrument failures from a table.
