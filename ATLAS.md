# ATLAS.md — validity-domain atlas, thermodynamic_mind @ r/kernelbench

One section per boundary/result, write-as-you-go (PLAN W5). Every entry: claim, validity
domain in derived coordinates, evidence artifact, variance, killer objection, U-mapping
(REFLECTION_STAGE3). Numbers quoted only from committed results files; provenance in
parentheses. Sections marked `draft` until their numbers survive a day-2 check where the
claim type requires it (IDENTITY drift protocol).

## Entry index

| # | name | op_class | status |
|---|------|----------|--------|
| A1 | probe-economics boundary: zero-probe gate vs budgeted autotune | reduction+conv | written 10-18 |

---

## A1 — Probe-economics boundary: the zero-probe gate beats budgeted autotune on total time

**Claim.** For the two families studied (reduction, conv2d-as-gemm) on GB10, a zero-probe
family-local gate (fixed rule, no per-shape measurement) has lower *total* regret — delivery
regret plus amortized probe cost — than any cold-start budgeted autotune with budget b ≤ 8,
at one-shot deployment. Equivalently: per-shape autotuning pays for itself only after a
shape is revisited ~15–260 times (family- and budget-dependent); below that, ship the gate.

**Boundary in derived coordinates.** The governing variables are not raw shapes but
(cumulative visits per shape v, probe cost p vs kernel time t, pool quality). The
gate/autotune flip lives at v* ≈ Δdelivery⁻¹·(probe spend): with delivery advantage of
autotune-b8 of 3.3 pp (reduction) / 6.4 pp (conv, == oracle on non-tied shapes) and probe
medians 0.12–3.7 ms (reduction) / 0.06–0.46 ms (conv) against delivery deltas of 0.1–3 pp,
v* is large — median 71 visits (b2) / 262 (b8) on reduction (n_finite 14/24 of 60), 14 (b2) /
66 (b8) on conv (n_finite 3/8 of 36). Probe cost, not kernel time, dominates the gate's
residual everywhere in the sampled lattice.

**Validity domain.**
- Arch: GB10 sm_120, 48 SMs (results/m3_summary.json meta), torch 2.14.1+cu130. No second
  arch measured; H100 folklore not imported.
- Families: split-K-style reduction (60 shapes) and conv-as-gemm imp variants (36 shapes),
  the M-lattice of 2026-10-17/18. Pool: torch defaults + seeded-shuffle hand pool; the gate
  was fit on the same grid (day-1 full fit, conv h0=14 c0=64 imp_B32x32, in-sample regret
  0.0550; reduction M≥16 & deployed-ctas bt≥1024 → tsplit_S64_KT64). In-sample gate fit is a
  declared residue: fresh-shape generalization is W4's axis, unrun.
- Regime: cold start, single-shot probes, flush-charged per registered protocol
  (m3_autotune_curve.py docstring, 10:30Z 2026-10-18, pre-results).

**Evidence.** `lab/results/m3_curve.json`, `lab/results/m3_summary.json`
(commit 556656d, 2026-10-18). Paired total regret, gate vs best budgeted arm (b0):
reduction 0.0386 vs 0.0445 (Δ +0.0059, SE 0.0026, 2.3σ, n=60); conv 0.0640 vs 0.1886
(Δ +0.1246, SE 0.0364, 3.4σ, n=36). No budget b ∈ {1,2,4,8} reaches parity on total time.

**Pre-registered falsifier and its status.** "If cold-start probe cost is negligible vs
kernel time at these shapes, the gate's edge is a probe-cost artifact — reframe to
warm-cache." Not triggered: probes dominate. Robustness: flush-free probing (~1 kernel-time
per probe) still leaves break-even ≫1 for kernel durations 10 µs–4 ms — the result is probe
economics, not flush accounting.

**Killer objection (standing).** A richer candidate pool (or torch.compile's own space) could
flip the delivery term: b8 already beats the gate on delivery where the gate is not
oracle-tied (40% of reduction, 22% of conv shapes). If W4's held-out test shows the gate's
*generalization* error exceeding budgeted autotune on fresh shapes, A1's adoption scope
shrinks to the sampled lattice. Second objection: single seed (4321) for pool order.

**U-mapping.** Feeds U4 (dispatcher + benchmark blind spot): the blind spot is economic —
the benchmark's per-shape visit count, not kernel quality, decides the right policy.

**Residues.** Gate params in-sample-on-grid; probe order single-seed; delivery view
conditional on non-oracle-tied subset; visit-count model assumes stationary shape stream.
