# s69 prereg — K=8 repeated-draw admission-rate probe (frozen before run)

Registered 2026-10-31 09:55Z, before any s69 data exists. Purpose: convert the s59 paired discordance
(26/228 labels flip between two input draws, median 5.9x budget-ratio swing) into an admission
frequency over draws, which is the claim the round-19 reviewer said the current evidence does not
establish.

## Population (fixed)
The 11 reduction shapes that flipped in s59: (K,M) = (98304,48), (98304,192), (98304,384),
(393216,20), (393216,48), (393216,192), (393216,384), (1572864,20), (1572864,48), (1572864,192),
(1572864,384). Arms: the 12-arm pool of s56_heldout_contract (torch, compile, five tsplit pairs).
Gate: KernelBench evaluator tolerance atol=rtol=1e-4 fp32, candidate-scaled, computed against the
default reference realization exactly as the shipped gate does.

## Design
N=32 independent input draws per shape, seed = (K*1000003 + M*7919 + draw*2654435761) % (2**31-1),
recorded per draw. Pass labels only; no timing (the rate claim needs labels, not times). Per
(shape, arm): admission frequency over the 32 draws. Per shape: frequency of "at least one arm
admitted" and the delivered-arm identity distribution under the stated policy.

## Primary object
Per-shape admission frequency versus the recorded budget-ratio swing from s59 (the ratio between
the two draws' realized margins). Reported with Wilson 95% intervals per (shape,arm) and a
shape-clustered bootstrap for the per-shape aggregate.

## Kill clauses (predeclared)
K1: if the shape-level admission frequency is flat in ratio swing (Spearman |rho| < 0.3 or p > 0.05
over 11 shapes), the draw-level flips are noise, not boundary locality => report a discordance
fraction and NO rate claim.
K2: if every shape's aggregate frequency CI overlaps the pooled s59 baseline 11.4% flips/228, the
rate adds no information beyond s59 => the claim narrows to "admission varies across draws" with no
ordering.

## Scope
These 11 shapes, this draw generator, GB10 rev 423217d. Not the benchmark. Not elementwise.

## Cost / timebox
11 shapes x 32 draws x 12 arms = 4,224 correctness evaluations; hard cap 20 min wall on spark,
partial results saved if the cap fires.
