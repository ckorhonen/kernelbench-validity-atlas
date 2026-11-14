# Meta-reflection QDU_TBi8iD3a6quG — "Combine complementary results" — answered 2026-11-14T08:4xZ

**Follow-up, candidly.** The last reflection named E2b (designed conditioning) then the depth bisection.
s99 ran at the tick it opened: prereg e35d699 was frozen pre-run, 25 rows, kappa(b) swept 1..1e4 at fixed
M=N=1024 K=4096 with Frobenius norm fixed, and the amplitude did not move — level means 1.341/1.301/1.356/
1.168/1.208 (1.16x across four decades), Spearman(kappa, rho_ref) -0.35 under the 0.40 bar, within-level CV
median 9.1% against s96's 9.04% at Gaussian inputs. P2_RETIRE_H1 fired, so the spectral hypothesis is retired
on this geometry instead of carried, and the amplitude thread closes as a draw-distribution statement
(results/s99_conditioning.jsonl, s99_analysis.json; §4.5h, MANUSCRIPT §2.4, ATLAS_CLASS_COLUMN s99; s51
extended 43->47, s50/s52/s53/s54 green, provenance 85/85/0). Two debts survived that tick unchanged: the depth
bisection (still a two-point direction, 4,096 vs 32,768) and the Welford V1 arm (tree and estimator
unseparated). Both are being executed this tick, with the prereg frozen before any row.

**Two findings that supply different parts of the route.**

*Tree side — s98.* The reference's realized launch was read from Kineto (grid [16,1,1], block [32,4,1] = 128
threads, identical at both shapes), so per-thread sequential depth is a computable coordinate: 32,768 at
N=4,194,304 and 4,096 at N=524,288. The gate-ratio contrast tracks it (V1 rho_vs_ref .0132 [.0110,.0142] vs
V2 .0584 [.0554,.0616] at depth 32,768; overlapping, i.e. absent, at 4,096). What it requires: my own kernel
with a declared merge tree at the reference's shape, evaluator-exact ratios, and a depth ladder at fixed N.
What it can provide: a cost in evaluator units per unit of arithmetic-order freedom — but only where the budget
is slack, since frac_over was exactly 0.0 on 32/32 rows.

*Margin side — s91/s92/s93/s84.* The conv/TF32 reference exceeds the benchmark's own 1e-4 gate (55/200 tasks in
s76; 99.85% of failing elements reference-induced in s84), and its magnitude collapses in u = 2.2e-4|A|max/atol
at a fixed gate (crossings 40.33/38.38, ratio 1.051 across a 4.88x element-count difference), with a 1.41x draw
lottery at fixed geometry (s96) and a 1.83% population plateau that is scale-independent (s92/s93). What it
requires: a scale and an arithmetic-order perturbation on the candidate in that regime. What it can provide: the
binding regime itself.

**The missing connection.** Neither is measured where the other is non-degenerate: the depth cost exists only
where the budget never binds, and the binding regime has no declared tree perturbation. So the exchange rate is
extrapolated across regimes, which a plausible story does not license. Investigation for the next tick: take one
conv/TF32 tuple from s91's crossing population, hold the reference fixed, apply declared tree permutations
(chunked sequential at 4..1024 chunks and fan-in-2 over the same boundaries) on both sides of the rho=1 crossing
(|A|max ~0.31 at 1e-4, ~40 at 1e-2), and measure the permutation cost in the same units with frac_over. Kill: if
that cost is scale-invariant in u within 10%, the two coordinates compose and the exchange rate is one number; if
it moves with scale, the exchange rate is a surface and the column must carry (u, depth).
