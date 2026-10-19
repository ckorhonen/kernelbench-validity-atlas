# CONTRIBUTION.md — reusable held-out generalization protocol (DRAFT 2026-10-19)

Ship condition (reflection #14 decision): publish AFTER the atlas sections it points to exist.
A1 written+day-2 passed; A2 written (day-2 pending). This draft is prepared now; publish gate =
A2 day-2 pass. Complements the fastest-kernel lane; does not enter it.

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
- Delivery frontier (zero-probe): compile 1.10% vs gate 4.51% reduction (n.s., t=-0.92); gate
  2.01% vs compile 58.83% conv (t=5.54) -- hybrid, A2.
- Population surface: (bytes/thread, waves) collapse, median residual 0.55% (A3,
  atlas_population.json) -- the domain picture a claim should be checked against.

## Honest limits (carry these with any publication)
Single arch (GB10 sm_120); single pool seed; warm-cache compile walls, cold v-structure
unmeasured; descriptive population bins, confirmatory prereg pending; one peer lane + 10-17
novelty sweep as basis -- re-check before publishing.
