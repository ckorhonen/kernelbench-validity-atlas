# REFRAMED PAPER — skeleton v1 (atlas-first), round-4 #45

Working title: **A validity-domain atlas for GPU kernels: conditional certificates, not band-global rules.**
Replaces the selector-first framing (2CmvAb5SC-VGAsbK, rejected 10-22 19:51Z); the selector economics
becomes a studied object (§6). Every number below carries its artifact (P1); `TBD` = needs its own
script before the section is written. Comparator throughout: torch.compile / recorded torch_cmp, never eager.

## 1 Introduction — claims are conditional or they are not claims
Program: boundaries irregular in raw-shape space; the publishable object is (rule, domain predicate,
certificate), not the win. Three reviewer rounds converged on the same defect class in OUR work
(F1 filter inversion, F2 count drift, F3 unconditional admission) while the arithmetic reproduced —
motivating the certificate-first reframe. Falsifier of the whole program stated in §7.

## 2 Instrument and the reference-realization axis
- Harness/calibration: 48 SM, L2 24MB, copy BW 259/220 two-slope; drift protocol day-2 ≤0.13% on
  saturated boundaries (results/e1_triad_day2.json). Launch floor 7.5/2.9µs; floor-time coordinate
  (e2b) — absolute-time floors are NOT derived coords (session-4 correction, kept).
- Reference realizations R_def / R_ieee / R_64 (r41): which reference a pass is measured against is
  part of the claim. Candidate-scaled vs reference-scaled tolerance directions disagree in code but
  0/560 realized verdict cells (f24_verify.json) — hygiene paragraph, not a results claim.
- Admissibility applies to EVERY arm including the baseline: torch passes own check def 56, ieee 21,
  64 4 of 56 (r45_rider_adm.json `torch_pass_own_check`).

## 3 Atlas entries (one subsection per boundary; P2 schema rows)
- 3.1 Elementwise triad (E1): collapse PASS in local-slope coords; copy ladder is NOT universal —
  persistent-vs-flat >5% falsifier resolved as grid-geometry dependence (e1_analysis_v3, e3).
- 3.2 Split-K small-out (E2/W2): tiled 2-stage wins 29/60 inside total-CTAs 32–128 ∧ 2D-tiled;
  M≤8 = launch-floor regime, boundary = intersection(wave-band, floor-time) (e2 v2/v3, e2b).
  1D-loop splits never win (instrument artifact, v2) — negative entry, kept per P5.
- 3.3 Conv-as-GEMM certificate (m3_r42/r42 chain, 56-row union): delivered gate == torch 52/56
  (r1_conv_gated); gate regret 0 on the 4 rows with passing hands; d_i vs compile per-tag
  +11.9/+12.1/+10.3 pp (t 2.44/1.64/1.60, r42_analysis) with the 5 torch_cmp-failing rows inside
  the paired denominator (f24_verify) — stated, not hidden.
- 3.4 Chain-order invariance (CF1): 0/36 conv, 0/168 rider cells order-dependent ⇒ ieee/tf32
  permutation cannot change any delivery; 21/56 delivered changes are anchor physics (cf43.json).
- 3.5 Anchor-profile necessity (CF2): mixed 48/56, nested 4, flat 4; ieee~64 exact agreement 39.3%;
  165/672 arm-anchor cells pass ONLY vs fp64 ⇒ 3-anchor lattice, no band-global tau (cf43.json).

## 4 Conditional certificates (primary contribution)
- Corrected counts (F2, f24_verify.json): hand-vs-default 15/560 (20 as-run); ieee-vs-fp32ref
  80/280 (95); admissible rows 14/2/0 by tag (a2 empty); tf32 0 both ways.
- Rider redo, admissibility on EVERY arm (F3 repair; lab/r45_rider_adm.py → results/r45_rider_adm.json):
  - Regret magnitudes are robust to the admission convention: paired |A2−A1| ≤ 0.025 pp on
    defined overlap (def +0.0021 n56; ieee −0.0066 n32; 64 −0.0443 max 0.0 n25).
  - Delivery claims are NOT: def→A2 disagreement 0/24/31 arms (def/ieee/64); NO admissible delivery
    24/56 (ieee) and 31/56 (64); adm pool EMPTY (oracle undefined) 20/56 (ieee), 4/56 (64).
  - def→64 delivery-status changes 52 = 21 arm→arm (== published A1 21, exact) + 31 arm→none;
    0 none→arm. Reading: the published "delivered arm changes" count was right; the unreported
    half is delivery LOSS under enforced admission — the certificate, not the ranking, carries it.
  - A2 adm-pool median size 2/2/5 (def/ieee/64), min 0 — certificates are thin; said plainly.
- Cushion screen (prereg lab/cushion_screen_prereg.md; results/cushion_screen.json): train AUC
  .743, c*=.4578, P(flip|err≤c)=0.0; rider transfer AUC .121 FAIL ⇒ one-sided safe-region
  certificate ONLY; no global ranking; per prereg fail-branch, recorded without retry.

## 5 Tolerance as a distribution axis
CF3: reduction pass flips 169/528 = 32.0% at bars 1.1–1.8e-4; b1 1.2e-7 margins never flip ⇒
margin-to-bar mechanism family-general; conv-section-shrink branch WRONG (cf43) — both stated.
Network propagation: TBD (needs a composed-model run; not claimed until run).

## 6 Selector economics, demoted to studied object (F1 repair)
- Published breakeven table reproduces digit-for-digit from the inverted filter (122→121.71 n=10/60;
  628→628.33 n=1; conv 34/145) — defect was the filter, located (m3_r42_analysis L59).
- Corrected form t_b<t_g, v*=P/(t_g−t_b): reduction finite b1 6/195.0, b2 7/225.4, b4 15/141.4,
  b8 21/232.3 (n/v*); conv EXACTLY ZERO — 35/36 rows structurally all-torch; published finite conv
  rows spurious (f24_verify regen).
- Dispatcher-vs-budgeted-autotune curve: TBD (W4 held-out exists but predates repairs; rerun under
  corrected oracle before any number appears here).

## 7 Threats, falsifiers, scope
Per-claim killer objections (P5): E2 floor model falsifier + outcome; no-collapse class written for
flat-1D; cushion one-sided scope; single-arch (GB10/sm_120) — H100 folklore re-derived not cited
(IDENTITY). Day-2 drift on every boundary claim or the claim is labeled single-day.

## 8 Reproduction artifact
Scripts-as-auditors shipped with the paper: f24_verify.py (count auditor), r45_rider_adm.py
(admission auditor), cf43.py (invariance/profil/margin), cushion_screen.py (+prereg), r42_analysis.py.
Manifest + per-number provenance table generated at package time (R5-style, <2MB supplement).

## Repair→section map (dispositions, all closed this time)
F1→§6 · F2→§4 · F3→§2/§4 · F4 scope→§1/§7 · F5 packaging→§8 (jsonl included, kernels/results paths
relative, deps pinned, repo URL, seeded randn — checklist at repack).

## Remaining before submission draft
1. §5 propagation run (GPU, small) or scope-cut the claim. 2. §6 dispatcher curve rerun (CPU from
records if admissible). 3. Provenance-table generator. 4. Then write prose section-by-section, one
per tick, numbers injected only from artifacts.
