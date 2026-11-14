# Appendix A — review history, retraction ledger, audit counts

This appendix holds what the main text no longer narrates: how the framing changed, which claims were retracted
and by which pre-registered rule, and how many checks stand behind the reported numbers. The main text is
organised claim → definitions → method → findings → limitations; history lives here.

## A.1 Why the framing changed

This paper replaces a selector-first framing that was rejected three times at full review
(`2CmvAb5SC-VGAsbK` 10-22 19:51Z, `6yPhozmAt62JSqQs` 10-21 19:44Z, `O4hnCxiPVenpDr4A`). In each round the
reviewer reproduced our arithmetic — paired differences, t-statistics, the population residual — and rejected
the interpretive layer on top of it. The defects were ours and they clustered: an inverted break-even filter
(`t_b < t_g`, `v* = P/(t_g − t_b)`; the published finite crossovers were computed on `t > t_g`, the complement of
the population the model describes), a mixed-unit count (`n_fail = 223` was check-level deliveries minus
row-level passes), and an admission rule that let the baseline pass a check the baseline itself fails.

Three review rounds were enough to establish that our numbers were right and our claims were not. That is why the
deliverable is a certificate rather than a ranking: a certificate names the reference realization and flags, the
budget-consumption pair, the admissible delivery set, and whether the regret basis is an upper or lower bound,
so a reader who does not trust us can check it.

Later rounds (rounds 5–11) produced six further rejections, all of whose findings were verified against committed
records before repair. The pattern across those rounds is recorded in `lab/triage_round*.md`: measurements
survived; interpretation, population accounting and packaging did not. Round 10's first finding was mechanical —
a submission whose supplement arrived null — which is why the submit path now asserts `files.supplement` on
read-back.

## A.2 Retraction ledger

Four claims in this program were retracted by their own pre-registered rules, with mechanisms rather than
apologies:

1. **The conv depth law** — the claim that winograd-class deviation scales smoothly with reduction depth. The
   two-class decomposition replaced it: deviation is a class property, and depth only orders the fp32 class.
2. **"Flag-invisible everywhere"** — the attribution that no precision flag changes the winograd path. True for
   winograd, false as a general statement: TF32 tensorop rows move 99.9% under `allow_tf32=False`.
3. **The ineligibility prediction** — the expectation that excluding winograd would lower $\rho$. It rose 22–25×,
   because ineligibility installs a worse engine.
4. **Engine class as a function of (k, s, d)** — the most consequential. Class is a dispatch decision over
   (C, output extent, stride, dilation, layout, batch); one step of output extent moves $\rho$ 20–38×. The 17-cell
   tuple surface (§2.5, Table 1) is the measurement that replaced the parameter-based prediction.

A fifth, narrower withdrawal: the round-8 claim that the reference's class is invariant to NCHW-versus-NHWC
layout. That test compared NCHW with itself; the real channels_last cell flips the class (§2.5, A07).

## A.3 Audit and provenance counts

- `s50_prose_audit.py` 19/19, `s51_prose_audit.py` 31/31, `s52_prose_audit.py` 16/16, `s53_intro_audit.py` 15/15,
  `s54` 6/6 — these are consistency checks that recompute cited values from committed artifacts, not independent
  regenerations. They catch transcription and drift; they cannot validate a measurement.
- `paper/PROVENANCE.md` maps every cited number to its artifact and field (cited/found/missing counted at build
  time; the build hard-fails on a missing citation). The manuscript's own header reads these counts from the
  artifacts instead of stating them by hand, so the three numbers cannot drift apart again.
- The tolerance-axis frontier (§5.4) carries its own instrument record: `results/e59b_budget_ratios.jsonl`
  reproduced all 672 arm-checks exactly, and the first reconstruction attempt (`e59`) failed its 95% validation
  bar at 87.35% and was discarded rather than reported.
- The auditors were themselves audited once: an earlier version of the s55 F2 check rewrote the artifact it read
  and reported a ratio of 1.0 for a real 0.0100 discrepancy. Pre-repair values are now declared constants with a
  build assertion, and the lesson is recorded in the diary as "a check must execute the intervention, not a proxy
  for it".

## A.4 Schema entries (P2)

Every reported result carries the P2 schema row `{op_class, rule, domain_predicate, derived_coords, boundary_ref,
variance, n_reps, date}` in the section that reports it; the rows for the certificate, the compression result, the
composition chains and the selector economics are in §4, §5, §6 respectively. Results without a schema row are not
reported as results.

## A.5 Open items carried forward

Three measurements are registered; one of them has been run and is reported in §1.7 and §9.5. The
**reference-swap flip table** (`results/gate_stability.jsonl`, 56 rows, prereg `lab/r41_gate_stability_prereg.md`
sha 5ff93b20, run 2026-10-22) holds candidates and inputs fixed and varies only the reference realization:
R_def→R_64 changes the delivered arm on 21/56 rows (all 21 inside the 25 in-rule rows), admissible-set membership
on 48/56, and 341 per-arm verdicts; the pure engine-class swap R_def→R_ieee changes the delivered arm on 11/56 and
membership on 15/56. The <10% adequacy clause therefore did not fire, and admission is at minimum a rate over
reference realizations rather than a point predicate. Two remain unrun: a **fresh-process class-invariance
probe** (cold cache and benchmark mode, since determinism here was measured across two repeats in one process)
and an **out-of-family class prediction** with the frozen predictor (kill at the majority baseline). A within-class geometry sweep testing whether $\rho_{\max}$'s rank compressibility also holds as a magnitude law
is registered too; the compression result is a rank result, not a bias statistic (§4.5, §2.5). A larger item is scoped but not started: a benchmark-wide reference-error audit
across all KernelBench tasks, which is the generalisation this paper does not attempt.
