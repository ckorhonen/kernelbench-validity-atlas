# Stage 3 — Technical brainstorming (completed 2026-10-17 02:15 UTC)

## Default overuses (named to avoid)
Per-task hero kernels; autotune ladders as `results`; H100 flash/bf16 trick imports; benchmarking only published shapes.

## Superficial vs mechanism-level
Parameter sweeps cosplaying as boundaries (no predicate/coords/transfer) VS predicates in derived coords + bisection-located flips. Difference = falsifiability + transfer.

## First-principles moves
- Representation: results live in derived-coordinate space; shapes are aliases. Coordinate superset: waves, bytes/thread, AI, occupancy, working-set/L2, grid/SM ratio — pruned by data (sample count >> coord count).
- Hypothesis (strong form): domains are low-complexity surfaces in derived space.
- Measurement: shape lattices; selector generalization error on held-out shapes = the benchmark's blind spot, quantified.
- Artifact: dispatcher + atlas, not a kernel zoo.

## Killer-experiment redesign (E1+E2 pair)
- E1 elementwise triad: instrument validation; textbook latency→bandwidth boundary must collapse in derived coords or instrument is broken.
- E2 contested boundary: split-K payoff in small-out reductions OR tensor-core/alignment validity under odd dims on sm_120. Spine wounded only if E2 no-collapse w/ E1 healthy + written falsifier.
- 2D boundaries: monotone-bisection needs 2D refinement fallback; "boundary is genuinely 2D" = interaction-term result.

## Added subquestion (absorbs cold-start angle)
Dispatcher-vs-budgeted-autotune: when does feature prediction beat cheap per-shame measurement (cold start, latency budgets)?

## Work packages
W1 instrument+calibration → E1 green. W2 E2 kill/confirm. W3 atlas wave 1 (norm/reduction, conv-as-gemm; numerics axis enters). W4 selector + generalization error. W5 write-as-you-go.

## Publishable units
U1 measurement protocol + GB10 calibration. U2 single dramatic boundary w/ mechanism. U3 coordinate-collapse across a family (spine, either polarity). U4 dispatcher + benchmark blind-spot quantification.

## Standing kill conditions
No-collapse in 2 families w/ falsifiers ⇒ reframe "domains irreducibly high-dim per arch". Drift unfixable ⇒ downgrade boundary claims, instrument paper first.
