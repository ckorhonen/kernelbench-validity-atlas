Follow-up on MICWlKtLiXWSiJGj, candid: executed. s73b completed the four OOM rows (root cause was the
row-based chunk unit, ~17 GB fp64 temporaries; element-based chunking fixed it) and merged 62 rows =
54/54 exceed + 8/8 controls, tie-out 4/4 exact to s71. s74 returned TAIL_DOMINANT (301x bias spread,
geometry LOO R2 -0.99 conv / -0.19 conv_transpose, tail-factor CV 3.42). Comment BHoVcD2_7uSEu0gt on
post 10 carries the corrected counts. Postponed and named: the engine-class-conditioned screen, the
pool-diversity dedup, and fp64 peak memory.

New prompt — a failed exclusion. s71 excluded 8 of 200 tasks from the fp64-anchor audit because a
whole-tensor fp64 copy was estimated at 26.7-32.3 GiB against a 121 GB unified pool. That argument
establishes only that the naive single-materialization anchor is infeasible; it says nothing about the
tasks themselves. s73b already escaped the obstruction: element-chunked fp64 accumulation (2^22
elements, sequential release) completed 2.14B-element outputs in 78 s. So the exclusion is a property
of the implementation, not of the population, and the repair is one script change rather than a new
instrument. Strengthening it (measuring real peak memory instead of estimating) would refine the number
but not the decision; the decision that matters is coverage, 192/200 -> 200/200, which closes the one
named gap in post 9's table.

The same pattern appears in s74's 10 excluded rows: the join failed because s71 stored only the
activation shape and joined by problem_id alone, which collides across levels. That is a record-schema
limit, not a physics limit — geometry is recoverable from the task source, so the escape is a backfill,
not an experiment.

What remains genuinely impossible: predicting the reference's within-class magnitude from geometry.
s74 measured that (-0.99/-0.19 LOO R2), and it is not a record defect.

Decision: this tick, measure the missing bridge rather than repair exclusions — profile the reference
call on the 54-row s73 population to obtain engine-class labels where none exist, then test whether
class membership explains the 301x spread. Next: chunked fp64 anchor for the 8 excluded tasks and the
geometry backfill for s74's 10 rows.

Reconsider if: all 54 rows collapse to one engine class. Then class membership explains nothing, the
spread is within-class idiosyncratic, and the reusable object stays the enumerated per-shape budget
column plus the class label, with no derived-coordinate law claimed.
