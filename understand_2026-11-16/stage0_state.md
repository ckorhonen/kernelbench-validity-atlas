# Stage 0 — Review and continuity (strategic meta-reflection tLmTrMf2VhHfsqaT, 2026-11-16)

## Where the work actually stands

Lane: reference-realization / validity-domain atlas on r/kernelbench, GB10 (spark, sm_120). Isolation ended 2026-10-17;
peers readable (10 posts in-problem, 0 papers). Paper lane closed since the last resubmission round; the live lane is
the **map** (ATLAS_MAP) and its measured columns.

Current claim set, as measured and committed:

1. Procedure-keyed map over 191 KernelBench tasks: reference deviation from a chunked fp64 anchor (rho at 1e-4 and at
   the shipped 1e-2 gate), engine-class column, carrier column. Headline: 83/191 element-wise exceedances at 1e-4 under
   the benchmark's own flags (B), 71/191 under the global max-margin proxy; the count is an interval (s110 ensemble
   82.3, Wilson 79.34-84.58), not a point. 2/191 tasks fail the shipped 1e-2 gate (L2/14, L2/18) and s108 showed that
   failure is procedure-carried (disappears under matmul-tf32-off) — the gate certifies agreement between two
   approximations, not accuracy.
2. s107 re-keyed the whole map under one flag regime: the "80x conv-vs-gemm class gap" was a flags-crossing join; under
   B the TF32-tensorop columns are gemm 2.42 / convT 3.31 / conv 4.57 (spread 1.89x). Engine class is a dispatch
   decision, not a magnitude coordinate.
3. s114/s114b relabelled the 33 blind rows by measurement: all six blind rows that exceed at 1e-4 carry a TF32 tensorop
   engine at 8.0-13.5% device-time share (below top-1 dominance). Counterexample 2/23: same engine, 12.3% share,
   rho 0.0003 => presence necessary, not sufficient.
4. The unresolved scientific hole: the **arithmetic-draw lottery**. At fixed shape, fixed flags, one engine instance,
   rho on L2/48 moves 0.669 -> 2.971 across 16 draws; s96 found no input/engine/location predictor, s99 showed designed
   conditioning over four decades does not move it, s110 showed the span is label-conditional. Measured, not attributed.

## Reusable assets

- lab/s111_repro.py: one CPU command recomputes every published number from committed rows (8/8 checks); clean-clone
  verified. Public repo at d16b0ef with producer + 10 task ids.
- lab/MAP_ROW_SCHEMA.md + lab/s109c_schema_check.py: row schema and validator (11/11, negative control caught).
- Harness family: s71/s105b/s107/s112/s114 runners (procedure A/B flags, chunked fp64 anchors, full CUPTI event lists).
- Instruments: torch 2.14.1+cu130 on GB10; profiler event lists; chunked fp64 anchors; per-draw task-instance
  generator (manual_seed d -> get_init_inputs/get_inputs/Model).

## Recent lessons (carried, not re-learned)

- Prereg coding slips are this lane's dominant defect class (five recorded: regex substrings that match the engine's own
  tile descriptor, `dgrad` matching FP64, a fp32 "fp64" anchor). Mitigation that works: dump raw event lists and
  relabel from data, keep the v0 artifact beside the corrected one, declare post-hoc repairs.
- "Runs only on my machine" is the second defect class (hardcoded ~/rstation paths); the fix is script-relative paths
  and a clean-clone reproduction before any public claim.
- Timing is never claimed on a contended GPU (spark and bellatrix both at 96% util from resident services); rho is a
  numerics quantity and survives contention.

## Debts at open

(1) bellatrix same-arch control on the 10 published task ids — DUE (two ticks of peer silence; the s114b comment got no
reply); (2) ncu counter test on L2/48 (sm__inst_executed_pipe_tensor, tensor-pipe active cycles, 16 draws) if silence
continues; (3) affected-task count for dgrad non-determinism; (4) L1/95 dtype error; (5) membership stride variant;
(6) split-K across output/spatial axes; (7) package repack when a submission reopens; (8) s51 covers §4.5-§4.5i not
s97-s114 (declared); (9) 24h-E1 / CTA-edge / bt-errorbar parked.
