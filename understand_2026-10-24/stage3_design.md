# Stage 3 — Design reflection (Deep Reflection 5/5, completed 2026-10-24T04:05:27Z)

## Candidates considered

**A. Reference-distance census (measurement artifact / dataset).** Insight: the reference's own distance
from an fp64 anchor, as a fraction of the task tolerance budget, is a per-family property nobody publishes
and it bounds every pass/fail verdict on that family. Decision changed: whether to claim a speedup at all,
and against which anchor, before choosing a construction direction. Controls: conv2d positive (predicted
≥2x budget), b1 negative (flip-invariant). Duplication: low — ruachtov verifies kernels, sylve verifies one
family vs one reference; nobody measures the reference itself. Overclaim: 3 points is not a census; GB10-only.

**B. Anchor-declaration protocol (measurement protocol).** Insight: the missing thing is a reporting
convention — a results row must carry reference build identity and its fp64 distance. Decision changed:
cross-lab comparability of reported speedups. Duplication: none. Overclaim: a protocol backed by one lab's
data is a proposal; without A it degrades to form-filling. **Verdict: B is A's packaging, not a rival.**

**C. Cross-arch replication of baseline admissibility (dataset).** Insight: is baseline-admissibility failure
(52/56 vs fp64) a GB10/cudnn artifact or general? Decision changed: whether the whole §2 argument is
arch-local. Duplication: none. Overclaim: none — it is the overclaim killer. **Blocked: no second arch
locally** (bellatrix is same GB class; the 3090 belongs to sylve). Becomes the first extension request.

## Primary direction (chosen): A, at its smallest falsifiable size

**Prereg (to be committed before any run).**
- Definition, single and evaluator-shaped: ρ(R) = max over output elements of |R_def − R_64| / (atol + rtol·|R_64|),
  atol=rtol=1e-4 ⇒ the largest fraction of the tolerance budget the reference itself consumes. All other
  error figures (cd tA1/tB1, d3 t*) are different quantities and must not be tabulated beside it.
- Families (declared by rule, not by record availability): conv2d 3x3 s1 p1 fp32, matmul fp32, each at one
  real benchmark shape from the pinned harness sources; plus the b1 gate-zero factored pair as flip control.
- Anchors: R_64 (fp64 inputs + fp64 accumulation), R_def (task reference at declared input precision),
  R_ieee (TF32 explicitly disabled).
- Documentation leg: read the pinned torch source for `allow_tf32` defaults on cudnn conv vs matmul.

**Predictions (basis stated, not asserted as fact).** conv ρ ≥ 2 (cudnn conv TF32 default). matmul ρ < 0.1
(basis: torch ≥1.12 defaults matmul TF32 off — to be verified in pinned source; contradiction is a result).
b1 ρ < 0.01 (margins 1.2e-7; ρ itself unmeasured — measure it, one cheap comparison, for symmetry).

**Outcome semantics.** conv ≥2 ∧ matmul <0.1 ⇒ benchmark tolerance contract is family-dependent by inherited
default and undeclared — primary claim. conv ≥2 ∧ matmul ≥1 ⇒ broader claim (reference realizations generally
exceed budget), needs a second op family. conv <1 ⇒ stage-2 conclusion retracts; 52/56 is candidate-side; say
so in the same tick.

**Kill conditions.** b1 ρ >0.1 ⇒ negative control fails, scale uncalibrated, no table. A family clean in ρ yet
flipping under anchor swap ⇒ ρ is not the mechanism; report the flip surface with no law.

**Deliverable.** ATLAS subsection A6: table (family, shape, ρ, induced flip fraction, pinned source line),
provenance column marking fresh-measured vs record-inherited, the one-line rule, and the protocol fragment
for declaring a reference build in a results row.

**Deliberately not doing.** No auditor-as-system. No cross-arch replication this tick. No re-run of the
56-row conv union (flip fractions cited from r45/cf43, not regenerated). No claim that fp64 is ground truth.

## Fixes carried from rounds 2-5
1. Family set chosen by declared rule, not by which records I happen to hold (selection-on-dependent-variable risk).
2. One definition, one anchor construction across all families — otherwise it is a comparison of instruments.
3. b1 is a flip-column control, not a ρ-column control; category slip corrected, b1 ρ measured to restore symmetry.
4. Provenance column distinguishes fresh ρ from record-inherited flip fractions (P1).
