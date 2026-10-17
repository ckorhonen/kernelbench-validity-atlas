# Stage 1 — Understanding the task (completed 2026-10-17 02:12 UTC)

## Round 1 — ground truth of the problem
- Suite: ~250 PyTorch snippets, tiers L1 single ops / L2 fused blocks / L3 networks; vision-flavored shapes, fp32/fp16; NOT covered: odd dims, stride/layout variety, serving batch=1 at scale, sub-fp16, dynamic shapes ⇒ every published speedup is a point sample in shape-space.
- Regime flips are conditional on derived quantities: launch-latency vs bandwidth for elementwise; split-K vs single-pass for reductions (output count vs SM fill); tile/wave quantization + alignment for GEMM/conv; tensor-core dim-multiple constraints; layout/alignment breaks vectorization.
- Reformulation: object of study = rules R with validity domain D_R + selector σ(features)→R. Kernel = instance; science = atlas + selector error on unseen shapes.
- Lenses: compiler (autotune = per-shape cache; failure mode = cache miss/generalization), numerics (tolerance axis; error compounds through networks), systems (GB10 unified memory inverts HBM folklore; measurement contamination).
- Sharpest form: atlas of validity domains for cost-dominant op classes + dispatcher + quantified shape-generalization error of point-sample benchmarking.

## Round 2 — assumptions attacked
- A1 "atlas worth building" ⇒ publishable object is boundary *physics* in derived coordinates, not constants; falsifiable collapse hypothesis (irregular raw-shape boundary ⇒ simple surface in derived space).
- A2 "beat the reference" ⇒ comparator must be torch.compile; the curve of interest: hand-kernel advantage vs compiler, as f(shape); atlas calibrated vs compiled or it inflates.
- A3 "correctness is a bit" ⇒ report error distributions; propagate reorderings through L3 networks before claiming fp16-accum wins.
- Difficulty: cost ⇒ bisection on boundaries (O(log) not grid), verify monotonicity, non-monotonicity = wave interference data.
- Difficulty: shared GB1 measurement ⇒ locked clocks if possible, interleaved A/B, L2 flush, variance; harness validated first.
- Difficulty: no 2nd arch locally ⇒ arch-transfer via reanalysis of published numbers, labeled.

## Round 3 — conclusions, uncertainties, behaviour plan
- Conclusions: conditional-claims framing; compiled baseline; error distributions; bisection; instrument-first. Derived-coordinate collapse hypothesis = program spine.
- Uncertainties: exact harness semantics in local rev (fast_p? comparator); GB10 clock lockability; no 2nd local arch; peer prior art (check post-isolation).
- Behaviour plan: P1 read harness/generators, validate measurement protocol. P2 pick densest-boundary op classes (normalization/reduction; conv-as-gemm), derive coordinates. P3 boundary-bisection harness. P4 rules with explicit domain predicates. Numbers carry derived coordinates or don't exist.
