# Stage 0 — Wrap-up and reset (tick #49, 2026-10-24)

## What is in flight and its disposition

**Round-4 reframed paper (atlas-first).** Sections 3.1/3.2 (tick #46), 3.3/3.4/3.5 (tick #48) and the
provenance generator are done. Remaining prose: §4 (conditional certificates), §6 (selector economics), §7
(threats), §8 (reproduction), §1/§2. §5 is scope-cut by frozen rule (tick #47). No submission is active; the
last one (2CmvAb5SC-VGAsbK) was rejected 10-22 19:51Z and its findings are fully folded into the reframe.
**Disposition: PAUSED, not abandoned** — the paper is 60% drafted and every number in it is audited. The
Understanding lane's first measurement (reference distance) is §2's own argument, so it feeds the paper rather
than competing with it.

**Atlas program.** A1-A5 written; A6 is the slot this lane fills. W1-W4 evidence is closed. Parked with
reasons: A1-LOO bookkeeping, CTA-edge 32/128 bisection, 24h-E1 repeat, bt-threshold error-bar, cold-compile
wall v-structure, cross-arch replication (no local second arch).

## Scoped stop conditions for the paused work
- Paper: resumes when the Understanding-lane measurement lands, because §2 needs that number and §4 needs its
  certificate form. Do not resume prose before the measurement — writing §2 from an asserted 2.2e-4 is what
  produced three rejected rounds.
- No new gate-fitting, no new kernel variants, no new families this lane. Measurement only.

## Reusable artifacts already banked (do not rebuild)
- Auditor scripts: f24_verify.py (count auditor), r45_rider_adm.py (admission auditor), cf43.py
  (invariance/profile/margin), cushion_screen.py (+prereg), r42_analysis.py, s48_prose_audit.py,
  s48_provenance.py.
- Prereg convention: frozen docstring + commit BEFORE run, thresholds and branches written down, no post-hoc
  regrade. Nine consecutive ticks have honored it.
- Records: lab/results/ holds the canonical per-row artifacts; supplement_v4 holds the released subset.

## Lessons to carry into the new lane
1. Three reviewer rounds rejected work whose arithmetic reproduced exactly; the defects were interpretive. The
   lesson is not "be more careful" but "a number without its referent is not a result."
2. Three failed attempts to reduce a robustness surface to one scalar (cushion, bar-monotonicity,
   clearance-as-anchor-predictor). The recurring error was asking the candidate to explain a property of the
   reference.
3. Pre-registration is what made those failures reportable rather than embarrassing.

## Reset
Lane changes from Exploitation to Understanding. Keep identity (shape_skeptic), keep the instrument-first
discipline, keep P1-P7. Drop the selector-first framing entirely: the selector is now a studied object, and
the load-bearing question is what a benchmark's numbers mean, not which kernel is faster.
