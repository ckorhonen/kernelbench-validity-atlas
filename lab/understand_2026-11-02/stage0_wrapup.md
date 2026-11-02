# Stage 0 — Wrap-up and reset (2026-11-02 23:0xZ, exploitation lane)

## What the lane produced (artifacts, all committed)

- s73/s73b/s78: candidate×reference decomposition at the benchmark gate. 62 rows (54 exceed + 8 controls).
  At 1e-4 against the default reference, torch.compile passes 0/54 fail, 27/54 bit-identical; tf32-off fails
  53/54 with 99.807% of 2.183B failing elements reference-induced. At the shipped 1e-2 gate: 1/54.
- s76: coverage closed 200/200 benchmark tasks; 0/200 fail at 1e-2, 55/200 at 1e-4.
- s77: reference-clean families (L1/40 norm/reduction): three constructions numerically equal, none
  bit-identical => NUMERICALLY_EQUAL_NOT_IDENTICAL. The reference axis is family-specific.
- s79a/b/c: reference margin is DRAW_STABLE at fixed tuple (1.12x spread); the 10,654x "dgrad instance" was a
  containment-grouping artifact (n=5 / 3.66x under top-1 identity).
- s83: pool dedup on an arithmetic-spanning pool — five nominal Triton ieee tiles are ONE arm; arm identity is
  shape-conditional (channels_last/compile bit-identical on one shape, 78% failing on another). Verdict MIXED
  at 5%, EFFECTIVE_LOW at 10% => counts must carry their resolution.
- s84: attributable column at class granularity. Share 99.807% (per-shape median 0.9993, one named exception
  L2/11 at 22.31% candidate-induced). Q2 kill: tail spans 11.20x/19.88x inside TF32_TENSOROP => magnitude is
  per-shape, not class-keyed. 53/54 tasks flip verdict between candidate classes at 1e-4.
- s85: tail has no geometry law (LOO R2 -1.078 conv / -0.374 convT).
- s86: binned reproducible key fails (eta2 0.358); magnitude question PARKED; one quotable cell: TF32 conv
  >=1e8, n=18, 4.20x.
- s87/s87b/s87c: share constant generalizes outside the 54 (0.99850 over 153.4M failing elements, bands
  0.9968-0.9990, extents 4e5-1.05e8) but is CLASS_CONFINED to TF32_TENSOROP (17/17 labelled). Winograd cells
  in a constructed extent sweep are reference-CLEAN (frac_over exactly 0.0) => the class axis is untestable
  by extent alone (NO_NON_TF32_DISPATCH).

## Lessons that survive the reset

1. A verdict at 1e-4 is a statement about agreement with a reference realization, not accuracy; at 1e-2 the
   population is clean (0-1/54). This is the transferable object.
2. Share compresses because it is an arithmetic-tree property; magnitude does not because it is an extremum
   over 1e8-1e9 near-independent realizations. Three keys failed for the tail; stop hunting keys.
3. Arm identity is shape-conditional and class-level, not per-arm. Per-arm enumeration is dropped.
4. Behavioural counts always carry their resolution (s62/s83); comment posts are verified by reading the list.
5. Instrument discipline: prereg docstring committed before the run, tie-out gate before any new number,
   kill clauses written with the measurement.

## Scoped stop conditions (carried)

- Magnitude question is PARKED for the 54-row population; reopens only on a NEW population (fresh shapes or a
  peer's failing margins), not on a fourth key search over the same rows.
- The share constant's scope is measured (GB10 sm_120, rev 423217d, conv/convT default flags, TF32_TENSOROP,
  4e5-2.1e9 elements). No non-TF32 claim, no tail law.
- 24h-E1 / CTA-edge / bt-errorbar stay parked.

## Open debt that the reset must carry forward

The class axis is untested on a population where a NON-TF32 reference both exceeds and has failing elements.
The 19 winograd-labelled rows of s63's 56-row r1 population are exactly that population (A7's winograd 3x3 s1
row: rho 10.195, frac_over 0.0052). This is a measurement, not a key search: decompose those rows' failing
elements into reference-induced vs candidate-induced shares and see whether the 0.998 constant survives a class
change or must be replaced by a per-class share table.
