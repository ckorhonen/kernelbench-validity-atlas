# Strategic wrap-up before the Exploitation lane (2026-11-13 ~23:15Z)

## What was in flight and is now stopped
- s96/s96b (amplitude lottery at fixed M=N=1024 K=4096): verdict AMPLITUDE_LOTTERY, ENGINE_STABLE,
  TAIL_INDETERMINATE fired verbatim at CV .0904 vs .10 bar. Artifacts committed; prose 4.5g; s51 43/43.
- Unfinished: (a) designed-conditioning experiment (vary sigma_max(b) over decades at fixed geometry,
  kill |Spearman|<.4) was the tick's lead and is NOT run; (b) 64-draw extension to settle the tail clause.
- Stop condition declared: the amplitude thread has run four ticks (s94-s96b) and each step narrowed the
  claim without producing a computable predictor. It is parked, not dead: the designed-conditioning probe is
  cheap and is retained as a secondary test inside the new plan, since conditioning is the one arm with
  descriptive leverage (sigma_max(b) -0.558) that Gaussian draws could not exercise (0.2% spread).

## Reusable artifacts and lessons carried forward
- Instrument: evaluator-exact ratio r=|A-X|/(1e-4+1e-4|X|), tail ladder, engine identity via torch.profiler,
  fp32 2^k exponent-exact scaling, per-draw repeats. These are validated and reusable (s93-s96, 65+ rows).
- Carrier rule measured: reference-carried .9985 (conv), equal-carried .56, candidate-carried 7e-5 (matmul);
  share is draw-unstable (.25-.75). The 1e-4 axis belongs to the default reference configuration.
- Reference at the shipped 1e-2 gate is clean on 32/32 rows; the extremum moves 41% across draws while the
  population tail moves 2%. Statistic to rely on = population tail / frac_over, not the extremum.
- Method lessons: prereg frozen pre-run; coded verdicts fire verbatim even when inconvenient; post-hoc
  analyses declared as such; audits extended in the same tick as new prose numbers.

## Scoped stop conditions for the old lane
- Resume the amplitude thread only if a new direction needs a per-shape margin bound; otherwise the column
  ships the draw spread with no predictor, which is already published.
