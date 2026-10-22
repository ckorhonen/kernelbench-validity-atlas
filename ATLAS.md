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
| A2 | delivery frontier at zero probes: gate vs compile, one-shot | reduction+conv | written 10-19, day-2 pending |
| A3 | population collapse in (bytes/thread, waves) | elementwise-flat | exploratory 10-19 |
| A4 | decision/calibration-boundary class (guard grammar) | dispatch admission | hypothesis 10-20 |

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

**Held-out generalization (W4, run 2026-10-18, results/w4_heldout.json, commit d7560c5).**
The in-sample residue above is now measured on fresh non-pow2 grids disjoint from every prior
fitting/timing surface (24 reduction, 10 conv points; frozen day-1 gate, no refit): the gate
GENERALIZES on the pre-registered rule — regret_total 0.0451 (reduction) / 0.0201 (conv) vs best
budgeted arm 1.53 / 2.15; beats always-torch delivery by 5.8 pp (2.1 sigma) / 5.5 pp (1.8 sigma);
on gate-fire subsets delivery is near-oracle (1.1% / 0.0%). Killer objection (b) confirmed LIVE
but family-conditional: torch.compile delivery beats the gate on reduction (1.1% vs 4.5%) and
loses on conv (58.8% vs 2.0%) — the delivery frontier is hybrid gate+compile, decided by
compile's one-time cost vs visit count. A1's adoption scope is EXTENDED to fresh shapes within
the sampled family/derived-coordinate domain; compile walls were warm-cache (cold-compile
v-structure not yet measured).

**Residues.** Gate params in-sample-on-grid (held-out check passed 10-18, see above); probe order
single-seed; delivery view conditional on non-oracle-tied subset; visit-count model assumes
stationary shape stream; compile arm alloc asymmetry on conv declared in-row.


---

## A2 — Delivery frontier at zero probes: no single always-delivery arm; hybrid gate+compile

**Claim.** In the one-shot regime (the v->0 end of A1's visit axis: deploy immediately, pay
nothing per-shape, never revisit), no single delivery arm dominates on delivery regret across
families on GB10. torch.compile edges the gate on reduction (1.10% vs 4.51% mean delivery regret;
paired delta -3.42 pp, sd 18.1, t=-0.92, n=24 -- not separable) and loses badly on conv-as-gemm
(58.83% vs 2.01%; +56.8 pp, sd 32.4, t=+5.54, n=10). The honest frontier is the hybrid: compile
where it does not lose (reduction, practically a tie), gate where it wins decisively (conv). The
gate carries zero compile-time/turbulence risk; compile carries a one-time wall (0.14 s on the
sampled row, warm-cache) that pays only via A1's visit economics.

**Boundary in derived coordinates.** Axes: deployment visit count v (A1's axis) at its v->0 end,
family, compile-wall amortization. The only decisive asymmetry measured is conv, where inductor's
sm_120 small-conv paths are the loss source (58.8%); reduction is unresolved at n=24.

**Validity domain.** Fresh non-pow2 grids disjoint from every prior fitting/timing surface, 24
reduction + 10 conv points, frozen day-1 gate (results/w4_heldout.json, run 2026-10-18); GB10
sm_120 48 SMs, torch 2.14.1+cu130. Compile walls warm-cache; dynamo recompile-limit silent
eager-degrade debt closed in-tick (declared in #14 follow-up).

**Killer objection.** The reduction leg is n.s., so "compile >= gate" there is unresolved; an
adversary reads the frontier as conv-only. Recorded response: separating 3.4 pp at sd 18 needs
~3x n -- day-2 candidate, not asserted.

**U-mapping.** U4 (dispatcher): the delivery rule is family-conditional, the atlas's dispatcher
shape. Worked example for CONTRIBUTION.md.

**Status.** written 10-19; day-2 PASSED 10-19 (lab/a2_day2.py, prereg docstring
07:59Z before run; 36 reduction + 10 conv fresh points): reduction not-separable
replicates (t=-0.65, n=36; gate 1.90% vs compile 3.48% -- direction flipped vs
day-1 but n.s. both days, so unresolved stands), conv gate-advantage replicates
decisively (+32.1 pp, t=2.56, n=10; day-1 +56.8 pp t=5.54). Bookkeeping note, no
threshold changed: the day-2 docstring wrote the delta as gate-minus-compile
while day-1 reported compile-minus-gate; numbers above use day-1's convention.
Per #14 the publish gate (A2 day-2 pass) is MET; CONTRIBUTION.md may ship.

## A3 — Population collapse: (bytes/thread, waves), not bytes/thread alone

**Claim (exploratory-descriptive; reflection #15 implication, HR-diagram order: locus first).**
Across 1486 flat-elementwise arm-points from three independent session artifacts (e1 day1, e1
day2, nw sweep; 21 configs), collapse in bytes/thread ALONE fails: median within-bin residual
35.8%, p90 91.7% -- latency-regime points share b/t with saturation points. Adding the second
derived coordinate, deployed waves (threads / (48 SMs x 2048)), collapses the population: 204
bins with n>=3 (1136 points), median within-bin CV 1.24%, residual vs bin median 0.55% (p90
14.7%). Cross-session points share bins, so this is also a replication check.

**Why it matters.** Population-level version of the spine: A1/A2 become annotations on this
surface. First atlas artifact built purely from committed results, no GPU time
(results/atlas_population.json; script lab/atlas_population.py).

**Validity domain.** Flat triton elementwise arms with exact deployed threads only. Persistent
(80 rows) listed y-only -- deployed thread count not stored; m3 variants empty in artifact; w3
gemm threads not derivable without config assumptions (all declared in the artifact meta). One
arch, one family, descriptive bins: NO confirmatory claim before a preregistered numeric
threshold (P5).

**Next.** Confirmatory prereg on FRESH points with the threshold fixed from this look's
sampling distribution BEFORE the run; coverage extension to m3/w3 via thread-count derivation.
A1's LOO-reconciliation bookkeeping stays parked, unaffected by numbering.

## A4 — Decision/calibration-boundary class (guard grammar absorbed); hypothesis

**Claim (hypothesis, not result — lane wrap GUARD.md is the evidence record).** Decision
boundaries (accept/cancel for a frozen dispatch rule) do not transfer across op-families as
calibrated thresholds, but collapse to a *global surface in derived coordinates plus a bind-time
per-family offset computable from shapes and weight statistics alone* — the zero-probe axiom
extended from response boundaries to decision boundaries.

**Positive constraints already established (commits 0e5f9d8, 45c298b).**
- Ordering is family-portable: frozen guard v1 on the LayerNorm family, pooled AUC 0.9875
  (randu/randn 0.9946) — `lab/results/b5_stage2.json`.
- The calibration offset is the whole failure: same run, accept 48/288 = 0.167 at precision
  0.167 vs the 0.90 train target; mechanism = family-constant z components (cwr 16.0 vs corpus
  1023.9–2047.8; γ-degeneracy w_rel_spread = 0, w_max == w_mean_abs).
- In-family, variance-gated bind-time features rank outcomes: held-out AUC 0.9026, precision
  0.9965 at fraction 0.353 — `lab/results/b4_guard_v1.json`; the ungated rank basis fails
  (AUC 0.636) — `lab/results/b4_guard_v0.json`.

**Validity domain.** Two families so far (split-K-style reduction corpus; LayerNorm transfer
probe), one arch (GB10 sm_120), features {w_mean_abs, w_max, collapse_work_ratio} + degenerate
dim-features. Cell-affine gate PASSED in-sample 10-20 (R² 0.831, see readout below);
fresh-data third-family test EXECUTED 10-20/21 (plan v2 item 3): instrument-negative —
the 1e-4 elementwise-all lens fails for conv-as-gemm everywhere on GB10 (min budget ratio
1.59; required symmetric tolerance t* median 2.8e-3 ordmag / 7.2e-3 cancel, up to 250x;
fp64 accumulation moves t* <5% => not a precision artifact; near-zero-output explanature
falsified, phi_cond = 0.0, passing elements not near-zero either) — family exits, frozen
kills unevaluable by construction (`lab/results/d3_margin.json`, prereg `d3_freeze`);
entry stays hypothesis-labeled, lens caveat recorded per P5.

**Killer objection (written).** If the cell-affine decomposition readout gives R² < 0.5 on
committed artifacts, the "global surface + offset" factorization is empty and the class reduces
to per-family refits — recorded then as the result, per plan v2 kill conditions.

**Cell-affine readout (2026-10-20, plan v2 item 2; posthoc, in-sample).** Unit-optimal
thresholds t*_u (Youden, plateau-midpoint; the 2 single-class units excluded by prereg)
regressed on the family-centered unit score-shift d_u with family intercepts: **R² = 0.831** on
25 units (n-weighted 0.845; leave-one-unit-out 0.784–0.905) — gate R² ≥ 0.5 **GREEN**
(`lab/results/d1_cellaffine.json`; prereg in the `lab/d1_cellaffine.py` docstring, written
before the run). Offset-only model reaches 0.456 < 0.5: the varying component carries signal
beyond a pure per-family constant (slope 0.588, sub-unitary — thresholds move less than
one-for-one with the score shift). The LN units enter with d_u ≡ 0 (family-constant z): the B5
mechanism visible in the covariate itself, their +0.354 offset absorbed by the intercept
(t* −0.259 vs frozen −0.613). Caveat recorded: in-sample fit on the same rows as the AUC
constraints above; the fresh-data test is the third family (plan v2 item 3).

---

## Addenda (#39, 2026-10-22 — correctness-qualified re-scope; paper §4 carries the full version)

**A1.** Conv delivery claims above predate correctness gating. Rerun `lab/results/r1_conv_gated.jsonl`
(56 rows / 672 arm-checks, per-arm precision declared, validate-then-time): gate conv delivery =
torch on 52/56 (4 admitted ieee rows), true-oracle regret .000/.000/.042 (m3m2/w4/a2; the .042
row = in-rule hand +42% vs torch, rule has no min-torch guard). The probe-economics mechanism
(probes ≫ delivery deltas) is unaffected; the b≥1 delivery terms inherit the same contract.

**A2.** Hybrid recommendation as originally stated does not survive the contract: torch beats
compile on 24/36 + 7/10 + 7/10 rows; hand delivery admissible only under declared fp32 ref
(ieee 95/280 arm-checks; on the 14/3/2 admissible rows hand beats compile 12/14+3/3+1/2, tiny n).
Instrument: cudnn default fp32 conv on sm_120 IS TF32 — 2.2e-4 rel vs fp64, tile-independent;
allow_tf32=False 2.7e-7; reference-side, explains the day-1 failure pattern. Stability: day-1
hand-arm err median 7.0e-2 (68 arm-instances, per-src .047–.082) sits on the rerun tf32 arm
(.0770), not ieee (.0149). Reduction side clean (b1_gatezero_v2 13/13, 8e-8).

**A3.** Residuals are bandwidth (achieved GB/s), not latency. Matched-n on the 1136 retained
points: 1D bytes/thread 31.7% median (17 log2-bins) vs 2D 0.55% — no common-subset artifact.
Degeneracy: the pair determines total bytes exactly (bt×threads; waves×98304, both exact);
bytes-log2-binning alone 0.84% — the 2D pair keeps ~1.5× edge. Claim scoped to bandwidth lookup
repeatability on retained bins; no unseen-shape prediction. (`lab/results/r39_text_support.json`)

**A4.** "Offset is the whole failure" withdrawn as a sufficient repair: on the repaired records a
family constant cannot reach 0.90 (max precision over retained thresholds 2/3; randn_p1000 96/96
fail w/ 16 accepted-and-failing; randu/randn 16/16 accepted-pass; frozen-t constant-drop 120
accepted at .567). Coverage (.167) separated from precision throughout; published-referent run =
kill-robustness evidence only. The score is blind to input distribution at matched gain/scale ⇒
distribution is a first-class covariate (A2 LOO prereg eb973d9 carries distribution-marginals).
.8021-vs-.80: single-configuration point estimate, not population transfer.
