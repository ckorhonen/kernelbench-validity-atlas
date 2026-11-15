# Stage 3 — Route and scaffold design (reflection session 2, 5 rounds, completed 02:04Z)

Routes compared: A full 200-task audit now (~4 GPU-h, table of zeros if non-conv is clean); B stratified 24-task
pilot then extend (~25 min, failure mode is a publishable scoping result); C narrow conv+gemm law (safe, answers a
different question). Chosen: B, because it is a prefix of A (its rows seed the full audit), and because the failure
mode is itself a result.

Scaffolds named, and removal order: sample (24 -> 200, only if wide); harness (borrowed s76 anchor conventions ->
verify one non-conv family against a second anchor strategy, else anchor-conditional caveat); single draw (one draw
per task is a survey; 16-draw band only on flagged rows); reference-only (admission leg last, on flagged rows).

Honesty mechanics: frozen declaration file of task IDs with source class and K before any row; frozen analysis script
that computes the joint distribution and prints the branch with counts; instrument failures stay in the table; sampling
weights reported beside split counts; the pilot cannot be narrated as wide if the split is partial.

Milestones: pilot branch today; full column or scoping note next tick; predictor held-out test; admission leg.
Change-course triggers: degenerate population, anchor incompatibility, instrument-failure dominance.
