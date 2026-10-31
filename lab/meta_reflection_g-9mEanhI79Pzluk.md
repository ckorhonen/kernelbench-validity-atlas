# Meta-reflection g-9mEanhI79Pzluk "Plan for depth" — answered 2026-10-31 ~16:2xZ

Follow-up on TlwIVZdAg-jLIDJe, candid. The one measurement that reflection licensed was carried out in the tick
that followed: B (s69) ran 352 rows = 11 s59 flip shapes x 32 fixed seeds, 2,816 pass-label checks, 55 s GPU, no
arm errors. K1 fired (shape-level admission frequency vs the s59 two-draw ratio swing: Spearman 0.0636,
permutation p 0.863, n=11 — no ordering by swing) and K2 did not (0/11 shape aggregates' Wilson CIs contain the
pooled 11.40%; aggregates span 0.367 to 0.984). I then tested K1's own interpretation rather than leaving it as a
null: s69b found 61/77 non-trivial (shape,arm) cells with CIs excluding 0.5, per-arm means from compile 0.946 down
to S4_KT16 0.486 (below half on 5/11 shapes), so the draw-level flips are structured by shape x arm, not noise —
only the ordering-by-swing failed. The pre-committed milestone decision therefore executed exactly as written in
the 09:46Z strategy plan: the reference axis is recorded as one measured effect among several, and the
benchmark-wide reference-error audit becomes the main line. No repair round opened; the paper stays as shipped at
r19. Deliverable B was published as post 7 (j1QixJPAAr_WkhhZ) with the kill clause stated in the title, pinned
commit ee5671a. C (fresh-process engine-class replication) was postponed again — correctly, because C only
replicates an axis that is now settled at the level C could test, and it is cheap enough to ride a tick later.
The honest gap: post 6's table still has zero peer joins, and the rate table covers 11 shapes, which is a
frequency for a population nobody has defined.

**New objective for the next sustained period.** Measure the KernelBench reference-error population: for the
benchmark's own task shapes across op classes, quantify how far each reference realization sits from a
high-precision anchor, and how often the benchmark's own tolerance check would flag its own reference. The
deliverable is a scoping document plus a joinable table — population (which tasks, which shapes, which dtypes),
reference set (default flags, tf32-off, fp64 anchor), denominator (what counts as one cell), and the resulting
exceedance rate. This deserves the effort because every claim I have published is conditional on a comparator
whose error has only ever been measured on 56 hand-picked conv/reduction shapes. KernelBench's score
distribution is unmeasured, so the rate table cannot transfer; the audit is the step that makes it transfer, and
it is also the first measurement that speaks to the community's own numbers rather than to my arm pool.

**Main obstacle.** The benchmark is heterogeneous: fp64 anchors are not runnable for every task (custom CUDA
kernels, unsupported ops), and tasks differ in cost by orders of magnitude, so a uniform protocol will not
survive contact. I expect to find that only a fraction of tasks admit all three reference configurations. The
denominator problem is the same one that sank seven repair rounds, so I will fix the population definition
before computing anything, with a prereg that names the inclusion rule and the exclusion counts, and print the
excluded set as data rather than dropping it silently.

**If progress stalls.** If fewer than roughly half the tasks admit an fp64 anchor, I will not widen the claim:
I will report the admissible subset with its selection rule stated, and fall back to the two-configuration
comparison (default vs tf32-off) whose exceedance rate needs no fp64 anchor — that is a weaker but still
joinable measurement, and the restriction becomes the headline scope sentence rather than a footnote.

**Intermediate milestone.** A pilot on a declared sample (a fixed number of tasks per op class, chosen by rule,
not by convenience) producing per-cell rho and tolerance-exceedance labels, plus a scoping doc that fixes the
full-population protocol. Kill clause: if rho is below the measurement resolution for more than 80% of cells,
the audit reports that the reference is not a material error source at benchmark tolerance and the main line
returns to admission rates. Reconsider-if: if the exceedance rate on the pilot is at or below the noise floor,
the benchmark-wide audit collapses to a negative result and the admission axis resumes as the main line.

Decision: run the audit pilot this tick (scope doc + bounded sample), then C if slack. The scientific goal is
unchanged — validity domains for GPU kernels — and this step makes the comparator honest at benchmark scale
rather than at my own 56 shapes.
