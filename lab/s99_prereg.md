# s99 prereg — E2b designed-conditioning probe (H1 spectral amplification), frozen BEFORE the run

Date: 2026-11-14 ~00:35 EST. Author: thermodynamic_mind. Frozen at commit time; no threshold edited after data.

## Question
s96 measured rho_ref 1.046-1.474 (CV 9.04%) at fixed shape M=N=1024 K=4096 with NO input predictor
(|Spearman| < 0.22 on every predeclared covariate), but its negative on H1 (spectral amplification) had no
leverage: iid Gaussian draws hold sigma_max(a) inside 0.20% (Marchenko-Pastur), so conditioning never moved.
sigma_max(b) returned -0.558 descriptively at 0.22% predictor variance, never a verdict arm.

This probe DESIGNs the conditioning instead of sampling it: b is constructed with a prescribed singular
spectrum, so sigma_max(b) and kappa(b) sweep decades while shape, engine, and output scale are held.

## Design (frozen)
- Geometry: M=N=1024, K=4096, fp32, benchmark flags (matmul allow_tf32=False, float32_matmul_precision
  'highest'). One reference call per row.
- b construction: Q1,Q2 = orthonormal factors from QR of Gaussian draws; s = log-spaced singular values with
  FIXED Frobenius norm ||b||_F = sqrt(K*N) (unit-RMS entries), swept kappa = sigma_max/sigma_min in
  {1, 10, 100, 1e3, 1e4}. b = Q1 @ diag(s) @ Q2^T. a = unit-RMS Gaussian, fixed across the sweep within a draw
  block, redrawn per block.
- Draws: 5 independent (Q1,Q2,a) blocks per kappa level => 25 rows. Seeds 9900 + 10*level + draw.
- Reference A = torch.matmul(a,b) under the flags above. X = fp64 matmul (torch, .double()) as anchor.
- Measurements per row: rho_ref = max_e |A-X| / (1e-4 + 1e-4|X_e|); frac_over at 1e-4 and at 1e-2; engine
  identity from torch.profiler on draw 0 of each level (kernel name set); covariates sigma_max(b), sigma_min(b),
  kappa, sigma_max(a), ||b||_F, median|X|, max|X|, extremum cancellation depth |X_e|/median|X| at the ratio argmax,
  and the absolute error at that element.
- Controls: (a) instrument repeat of level kappa=100 draw 0 must be bit-identical on rho_ref and frac_over;
  (b) engine set must be identical across levels (else the sweep confounds dispatch and the verdict is
  ENGINE_CONFOUNDED); (c) sigma_max(b) recomputed from the constructed b by torch.linalg.svdvals must match the
  prescribed value to <=1e-3 relative (construction check).

## Verdict rule (frozen, computed mechanically)
Primary statistic: Spearman(kappa, rho_ref) over the 25 rows, two-sided.
- P1 PASS_CONDITIONING_PREDICTOR iff |rho| >= 0.40 AND rho_ref spans >= 1.3x across the sweep. The amplitude then
  gains a computable input coordinate; the column ships eps(kappa) as a hypothesis-labelled predictor.
- P2 RETIRE_H1 iff |rho| < 0.40 AND rho_ref spans >= 1.3x: conditioning is not the carrier, H1 retired entirely
  (this is the reflection's predeclared retirement condition).
- P3 NO_LEVERAGE iff rho_ref spans < 1.3x: the constructed sweep does not move the amplitude either, so the s96
  spread is draw noise rather than conditioning, and the amplitude question closes as a sampling statement.
- Secondary (descriptive, no kill): within-level CV of rho_ref at fixed kappa (s96's 9.04% baseline) and whether
  the extremum element stays deep-cancelled across levels (s96b's finding).

## Kills / instrument
- K1 instrument repeat mismatch on rho_ref or frac_over => INSTRUMENT_FAILED, no verdict quoted.
- K2 engine set differs across levels => ENGINE_CONFOUNDED, no predictor claim.
- K3 construction check fails (svdvals mismatch) => CONSTRUCTION_FAILED.

## Scope
One geometry (M=N=1024, K=4096), one family (matmul, fp32 SIMT reference on GB10 sm_120, torch 2.14.1+cu130),
5 draws per level. No timing claim (GPU may be shared; numerics only).
