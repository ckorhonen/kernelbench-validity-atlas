# CONTRIBUTION.md — reusable held-out generalization protocol (FINAL v1.0 2026-10-20)

Ship condition (reflection #14 decision): publish AFTER the atlas sections it points to exist.
A1 written+day-2 passed; A2 written, day-2 PASSED 10-19. Publish gate MET; submitted 2026-10-20
(see wrapup_atlas.md / DIARY for the submission receipt). Complements the fastest-kernel lane;
does not enter it. The A4 decision-boundary class (ATLAS.md) and its two guard kills are part of
the evidence record below.

## Protocol (what a claim must carry to be falsifiable)
1. FROZEN rule: the dispatcher predicate / config rule, written with its domain, BEFORE test
   shapes are drawn. No refit after seeing them.
2. Fresh disjoint shapes: sampled from the benchmark's own generator, disjoint from every
   prior fitting AND timing surface (W4 used 24+10 non-pow2 grids).
3. Comparator set, probe-charged honestly: frozen rule; budgeted autotune b in {1,2,4,8};
   torch.compile delivery; oracle post-hoc. Delivery regret AND total regret (probes + walls)
   reported separately -- they rank differently (A1 vs A2).
4. Paired per-shape statistics: regret vs oracle per shape, paired delta vs each comparator,
   t with n; numerics within benchmark tolerance, err recorded in-row.
5. P2 schema row per claim: {op_class, rule, domain_predicate, derived_coords, boundary_ref,
   variance, n_reps, date}. Runtime target: minutes (W4: 34 shapes, 12 reps, one GPU-hour
   including harness debts).

## Worked example (numbers from committed artifacts only)
- Gate-vs-budget (total regret, fresh shapes): reduction 0.0451 vs 1.53 (b1) / 29.8 (b8, probe
  charges dominate); conv 0.0201 vs 2.15 -- A1 generalized (w4_heldout.json, prereg 2026-10-18T20:41Z).
- Delivery frontier (zero-probe, correctness-qualified #39): reduction gate-vs-compile not
  separable (day-1 t=-0.92, day-2 -0.65); on conv compile loses to the gate's correctness-gated
  delivery, which is torch (52/56 rows; beats compile 24/36+7/10+7/10) -- hand only under a
  declared fp32 reference (`r1_conv_gated.jsonl`), A2.
- Population surface: (bytes/thread, waves) bandwidth-residual collapse, median 0.55% vs 35.8%
  bytes/thread alone; matched-n 1D-on-common-set 31.7%; total-bytes lookup 0.84% (partial
  size/geometry code) -- A3, `atlas_population.json` + `r39_text_support.json`.

## The counterexample half: acceptance criteria the guard kills established

A frozen rule needs an admission filter, and the lane's guard block delimits what such a filter
can claim (full record GUARD.md; atlas entry A4):
- **In-family, bind-time features rank outcomes** — variance-gated {w_mean_abs, w_max,
  collapse_work_ratio}, held-out AUC 0.9026, precision 0.9965 at accept fraction 0.353
  (b4_guard_v1.json); the ungated rank basis fails (AUC 0.636, b4_guard_v0.json).
- **Ordering transfers across families; calibration does not** — frozen v1 as-is on LayerNorm,
  repaired symmetric-reference rerun (b5_stage2r.json; repair committed 2470a62 pre-run):
  pooled AUC 0.8021 with K3 killing in BOTH the published (0.9875, b5_stage2.json) and repaired
  runs ⇒ the disposition is robust to the referent repair. Accept 48/288 = 0.167 at precision
  0.667 repaired (0.167 published) vs the 0.90 train target. Mechanism (repaired attribution):
  cwr-z is a pure constant offset (−1.7223, AUC 0.5); the load is the γ-magnitude pair —
  family-constant components preserve rank order and shift the absolute score.
- **Consequence for the protocol:** any claim of the form "rule R with admission filter A" must
  declare, per family, which features are constant and where the operating point was fit;
  a threshold fit on one family is evidence about ordering, not about the operating point.
- **Offset-repair sufficiency withdrawn (R2, #39 10-22):** a family-offset-only repair cannot
  reach 0.90 on the repaired records (max precision 2/3 over retained thresholds; randn_p1000
  96/96 fail); coverage 0.167 is reported separately from precision; the score is blind to input
  distribution at matched gain/scale, so distribution-marginal covariates are required for any
  operating-point claim; transfer estimates carry seed/gain/input-conditioned uncertainty
  (GUARD.md repair addendum 2).

## Honest limits (carry these with any publication)
Single arch (GB10 sm_120); single pool seed; warm-cache compile walls, cold v-structure
unmeasured; descriptive population bins, confirmatory prereg pending; one peer lane + 10-17
novelty sweep as basis -- re-check before publishing.
