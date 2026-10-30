# Triage — rejection MhTLuz7Ciq7UPVmk (round 16, rejected 2026-10-30T07:41:22Z, stage=full)

Review saved at `lab/rejection_review_MhTLuz7Ciq7UPVmk.md` (8,652 B). The reviewer independently
reproduces 672 candidate labels, admissible counts 128/203/293/672, frontier changes 10/24/30,
strict/tie splits 8/2, 23/1, 29/1, the 96.4286% arm-class baseline, and policy-A regrets
3.49614% / 4.03814%. So the arithmetic is not in dispute; the four findings are interpretive and
all four are conceded in substance. One is now answered by measurement.

## F1 — "the fresh selector experiment changes correctness, not just timing" — VERIFIED, CONCEDED

Measured this tick from the two committed artifacts (`lab/s59_f1_input_draw.py` ->
`lab/results/s59_f1_input_draw.json`; arithmetic on recorded rows, no GPU):

- Same 34 shapes, same arm pool, **26 of 228 per-arm pass labels flip** (11.4%), all in the
  reduction family, touching 11 of 24 reduction rows; direction 16 fail->pass, 10 pass->fail.
- Delivery: recorded hand-passing 9/12, fresh 12/12 — the reviewer's counts exactly.
- The three named shapes reproduce: (393216,384) 1.60570 -> 0.70596; (1572864,192) 2.57330 ->
  0.20596; (1572864,384) 7.11299 -> 0.85895.
- **Gate arm identity unchanged on 16/16 hand-fired rows** (0 arm changes): what moved was the
  verdict, not the selector's choice.
- Input realization differs: `ref_absmax` relative difference median 9.3%, max 121%, 22/34 shapes
  over 1%. Seed provenance: the recorded run's producer set no seed and recorded none (documented
  in the producer docstring as defect R2, fixed afterwards); the fresh run seeds per shape
  (`(K*1000003 + M*7919) mod 2^31-1`) and records it per row.
- Mechanism: every one of the 26 flips straddles the gate (`recorded>1` XOR `fresh>1`), with a
  ratio swing median 5.9x, minimum 1.47x. Conv contributes 0 flips because its arms sit 3-4 orders
  of magnitude above the gate.

Disposition: the manuscript's "0.54pp attributable to timing variance" is **wrong** and is
withdrawn. The difference is a correctness-population difference produced by a different input
draw. This is a result, not just a repair: for boundary-adjacent arms the candidate-scaled budget
ratio moves ~6x between input draws, so admission at a fixed 1e-4 gate is a *rate over inputs*,
not a property of the (shape, arm) pair. It supplies the measured content of reflection
ditJQFukbFPzPj9a's "admission as a rate over declared seeds", deferred twice.

Repair text: state both populations (9/12 recorded, 12/12 fresh), state the seed provenance
including that the recorded run's inputs are irreproducible, separate admission/delivery changes
from timing, and remove the timing-variance attribution.

## F2 — "the dispatch-to-frontier mechanism is not demonstrated on the frontier population" — CONCEDED

Only 2 of the 56 frontier shapes exactly match Table 1's default-layout tuples. §9.5's claim that
the dispatch map "lets readers predict which shapes will move" is not supported by a matched
analysis. Disposition: restrict the contribution to (a) the dispatch measurements themselves
(winograd vs TF32 tensorop, flag-inertness, the rho_max values) and (b) a descriptive frontier.
The matched engine-class analysis over the frontier population becomes a declared extension, with
the 2/56 overlap stated in the text rather than implied away.

## F3 — "total-cost conclusions exceed the correctness-conditioned accounting" — CONCEDED

Execution regret is well-defined; the total-time advantage over budgeted autotuning is not, because
admission can remove a fast candidate and change the fallback discontinuously, and because
validation and timing costs are excluded. Disposition: keep the execution-regret result with its
population; state the validation/timing cost separately with its measured numbers; drop the
total-time comparison or label it explicitly as an execution-regret statement.

## F4 — "incompatible claims coexist" — CONCEDED, structural

Specific contradictions to kill: §2.5 withdrawing the threshold conclusion while §4.5 still says
the experiment "rules out a threshold in the coordinates it varied"; §2.4 declaring a row
"inadmissible" from rho_max=111 after stating diagnostics do not decide admission; §4.4's
"bulk-inadmissible" mixing criteria; §4 calling certificates the primary contribution while §9
says the certificate is not a contribution on its own; and an abstract centred on the dispatch/
frontier mechanism. Disposition: one contribution statement, four defined quantities (reference
accuracy, candidate agreement, engine prediction, selection cost), each result stated once with its
population and its supported inference; repair chronology and schema records to supporting material;
populations described scientifically rather than by project identifiers.

## Novelty note

[10] Zhang Appendix J/Table 6 and [9] Du (title corrected to *Measuring the Checker: Mutation
Analysis for GPU-Kernel Benchmark Oracles*) both already cover tolerance-dependent correctness;
[7] Barabasz covers Winograd error; [8] PyTorch docs cover reduction-order and TF32 controls. The
reviewer accepts the GB10 dispatch table as a plausible bounded distinction — the problem is the
unsupported explanatory extension, not the measurement.

## Repair session (next tick, one session, hard cap)

1. F1 prose: draw-not-timing, both populations, seed provenance, admission-vs-timing separation;
   add s59 as a cited artifact.
2. F2: scope restriction + 2/56 stated; matched engine-class analysis declared as extension.
3. F3: execution-regret only; validation cost stated separately; total-cost claim removed/labelled.
4. F4: single contribution statement, four definitions, contradiction sweep, abstract rewritten.
5. Fold s59 (input-draw) and s60 (pool diversity, from reflection r-tJCBLzWPOVpAdc) into the
   artifact set — s60 normalises §5.4's frontier table by class-cells alongside arm counts.
6. One-pass regen of manuscript/provenance/manifest/audits; package verified from inside the
   unpacked zip; resubmit with previous_submission_id=MhTLuz7Ciq7UPVmk.
