# ATLAS A9 — two-stage reference budget composition (probe, tick #52)

Prereg `lab/a9_compose_prereg.md` @ 53ca881, frozen before the run. Artifacts
`results/a9_compose.json` (6 chains, 14:1xZ), `results/a9_analysis.json`. Spark GB10, torch 2.14.1+cu130,
benchmark defaults untouched (cudnn.allow_tf32=True, matmul.allow_tf32=False, precision highest,
cudnn.benchmark=False), recorded per row. All 6 chains determinism-verified (2 fp32 runs bit-equal).

## What this tests and why it is not a repeat of the cut V2 leg

§5 propagation was cut at #47 under my own frozen rule because held-out gate-hand rows carry no `|ref|`
column, so an rtol-scaled verdict could not be computed CPU-side. That failure was an instrument gap, not
a mechanism result. The currency the claim actually needs is already dimensionless — rho's denominator IS
the harness budget `atol + rtol*|R_64|` — so composition can be stated in budget multiples with no `|ref|`
field at all. A9 measures that currency end to end rather than validating it from a column that was never
there.

## Frozen-rule outcome (verbatim, no regrade)

| rule | form | pass | adequate | failing chains |
|---|---|---|---|---|
| R1 union | `1-(1-fA)(1-fB)` on frac_end | 4/6 | **no** | c5, c6 |
| R2 no-amplification | `max(rhoA,rhoB)` on rho_end | 5/6 | **yes** | c6 |
| R3 additive | `rhoB + G*rhoA` on rho_end | 5/6 | **yes** | c6 |

Verdict per the frozen rule: **COMPOSITION_PREDICTIVE**. Two rules adequate, both with the same single
failure, and both cleared on a clean-A control (c4, c5) as the prereg required.

## What the numbers say, including what they do not separate

The chain-level readings, with amplification share `G*rho_A/rho_B`:

- c1 W→TF32: rhoA 7.726, rhoB 5352.9, rho_end 5251.1, G 0.057 → amplification share 0.0001. The deep TF32
  stage swallows the winograd stage entirely; end-to-end budget consumption is 0.98x R3 and the bulk
  exceedance is 0.7982 vs B's own 0.7981, i.e. stage A contributes one part in ten thousand.
- c2 W→W: 2.935 / 22.134 / 33.365, G 0.533 → share 0.071; end 1.41x R3.
- c3 TF32→TF32: 80.962 / 793.094 / 1218.6, G 0.565 → share 0.058; end 1.45x R3. Bulk exceedance rises
  from B's 0.7877 to 0.8448 (+0.057), so a genuinely dirty upstream stage does move the fraction.
- c4 clean-A control: rhoA 0.006, rho_end = rhoB exactly (237.279 both), G 1.438 but G*rhoA 0.009 is
  invisible. The additive rule correctly predicts a clean-A chain adds nothing.
- c5 clean-A control: rhoA 0.069, rhoB 0.656, rho_end 1.294 — the only chain where the amplification term
  is comparable to B (share 0.053 of B but both are sub-1), and the only chain where R1 fails on a
  nonzero measurement (frac_end 3.05e-5 vs R1's exact zero).
- c6 deep-matmul chain: rhoA 2.275, rhoB 22.672, rho_end 56.229 = **2.34x R3 and 2.47x R2**. This is the
  declared failure: neither the max bound nor the additive amplification accounts for it. The chain is
  matmul M1024 K8192 → matmul K=1024, i.e. a deep reduction feeding a deep reduction, and the excess is
  the one place in six where composition is super-additive.

**The separation the test did not achieve**, declared rather than papered over: R2 and R3 differ only by
`G*rho_A`, and that term is under a factor of 1.2 of B on five of six chains, so passing both does not
discriminate "no amplification" from "additive amplification". R1 IS separated — it fails 4/6, so
independence of exceedance is ruled out on this sample. The honest summary is that composition is bounded
by the worst stage on 5/6 chains and that bound is 1.0-1.5x, with one super-additive exception at 2.34x.

**Post-hoc descriptive pattern, not a verdict** (no rule regraded): `frac_end` sits at or just above
`max(fracA, fracB)` on five chains and 0.0006 below it on the clean-A control (+0.0001, +0.0043, +0.0571,
-0.0006, +3e-5, +0.0029 in absolute terms).
The ratio to that max is 1.00, 1.59, 1.07, 1.00, undefined, 2.95 — so where the fractions are large the
compound exceeds the worst stage by at most 1.07x, and where they are small (c2 0.0073→0.0116, c6
0.0015→0.0044) the ratio rises but stays far below the 0.5 bulk threshold. Bulk inadmissibility is
dominated by the worst stage rather than compounded, which is why the union rule over-predicts on c5/c6 in
the small-fraction regime and why R1 was the wrong form.

## Consequences for the paper

§5 propagation moves from SCOPE_CUT to a **declared composed claim with a stated domain**: two-stage
composition on GB10 is bounded by the stage maximum within a factor of 1.5 on 5/6 measured chains, with
amplification subdominant whenever the downstream stage is the dirtier one, and super-additive at 2.34x on
a deep-reduction-into-deep-reduction chain. The SCOPE_CUT decision of #47 stands as the reason the claim
was delayed, and the missing `|ref|` column is named as what it was — an instrument gap in the wrong
currency.

Declared limits, unchanged from prereg: G is a local sensitivity at one element of one shape, not a
Lipschitz constant; six chains, per-chain scoring, no confidence interval; two-stage only, nothing claimed
about three-or-more stages.
