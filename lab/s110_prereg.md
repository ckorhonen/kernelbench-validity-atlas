# s110 prereg — draw band on the 23 rows in rho_ew [0.5, 2.0] (frozen before any row is computed)

Date: 2026-11-15 (EST), tick after s109/s109b. Purpose: the ATLAS_MAP headline "83/191 tasks exceed the
reference budget at 1e-4" was produced from ONE draw per task. s109 gave 8 flagged rows (rho >= 1, cheapest)
a 16-draw band and found exceed rate 1.0 away from the boundary and 0.9375 on the single boundary row
(rho 1.147). The count itself still has no band: the rows in rho [0.5, 2.0] are the only ones where a draw
can change the count, so measuring their per-row exceed rates converts 83/191 from a point into a measured
interval. This is the plan's LEAD action, fourth deferral closed.

## Population (coded rule, frozen)
All rows of s107_rekey_map.jsonl + s107b_rekey_rest.jsonl with status == "ok" and
0.5 <= B_strict.rho_ew <= 2.0. Measured count at freeze: 23 rows (14 recorded rho > 1, 9 recorded rho < 1).
No quota, no manual exclusion. The population is defined by the re-keyed map (procedure B), not by s71.

## Protocol
Identical to s109: draw = task instance (torch.manual_seed(d), get_init_inputs(), get_inputs(), Model(*init)),
16 draws d = 0..15 per row, procedure B (cudnn.allow_tf32 = True, matmul.allow_tf32 = True), reference output in
fp32, chunked fp64 anchor (element chunks of 2^22, sequential release), evaluator-exact statistics
rho_ew = max |ref - anchor| / (atol + atol*|anchor|) at atol = 1e-4, frac = share of elements above the same
predicate. Secondary gate 1e-2 recorded per draw. Profiler top-1 label per draw.
Rows ordered cheapest-first by s71_t_fp64_s so a wall cap loses only the expensive tail.
Wall cap 2700 s; per-task alarm 240 s.

## Instrument gates (must pass before any branch is read)
K1: draw-0 in-process repeat bit-identical on >= 75% of measured rows.
K2: fp64 anchor self-check relative <= 1e-9 on at least one row.
K3: tie-out to the recorded s107/s107b rho at draw 0, |rel| <= 0.05 on >= 6 rows (same seed-0 protocol).

## Frozen branches
- INSTRUMENT_FAILED if any instrument gate fails; no rate quoted.
- If instrument passes: R = sum over measured rows of r_i (r_i = exceed rate at 1e-4). Because 14 of the 23 rows
  were recorded as exceeding, the map count under this ensemble is 69 + R.
  * BRANCH_COUNT_BAND if any row has 0 < r_i < 1 => the count ships as 69 + R with the measured per-row rates.
  * BRANCH_COUNT_STABLE if every measured row has r_i in {0, 1} => 83 reproduces and the only band is draw noise.
- BRANCH_BENCH_GATE_ROW if any measured row has frac(1e-2) > 0 on any draw (a shipped-gate failing task outside
  the two s108 found); else BRANCH_BENCH_GATE_CLEAN.
- Partial: if more than 6 rows are wall_capped or fail to load, the band is declared over measured rows only and
  the unmeasured rows keep their single-draw label.

## Predictions (recorded, not used as branches)
Rows recorded below 0.72 should give r_i = 0; rows above 1.30 should give r_i = 1; the 4 rows in [0.72, 1.30]
are the plausible mixed ones (L2/100 0.9445, L2/48 1.1002, L2/77 1.147, L1/81 1.3008). s109b already measured
L2/77 at 0.9375, so the prediction for that row is 15/16.

## Non-claims
One architecture (GB10 sm_120); 16 draws is a sample, each rate carries +-1/16 resolution; the fp64 anchor is
the reference for "exact", not ground truth of the task; engine labels are dominance observations.
